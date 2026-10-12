"""Figure framework on synthetic outputs produced by the real experiment runner.

Checks that every plotted number is read from saved files (never invented), that the
statistical unit is the subject, that PLV and |ciPLV| stay separate, and that outputs
are never overwritten.
"""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

pytest.importorskip("torch")

from adgraph.visualization import (connectivity, embedding, fusion, performance, spectral,  # noqa: E402
                                   subjects)
from adgraph.visualization.demo import build_synthetic_results, synthetic_stats  # noqa: E402
from adgraph.visualization.figures import FAMILIES, make_figures  # noqa: E402
from adgraph.visualization.results import index_results, read_csv, read_summary, select_unique  # noqa: E402
from adgraph.visualization.stats import InvalidStatistics, load_stats  # noqa: E402
from adgraph.visualization.style import figure_size, load_style  # noqa: E402
from adgraph.visualization.writer import FigureWriter  # noqa: E402

STYLE = load_style("configs/figures/style.yaml", {"formats": ["pdf", "png"], "dpi": 100})
PAPER = yaml.safe_load(Path("configs/figures/paper.yaml").read_text())
EXP6_FULL = {"label": "Exp6", "experiment": "exp6_multiband_psd_strength",
             "where": {"node_features.components": ["psd", "strength"], "model.fusion": "attention",
                       "graph.bands": ["theta", "alpha1", "alpha2", "beta"]}}
EXP1_SVM = {"label": "Exp1 SVM", "experiment": "exp1_source_psd", "where": {"model.name": "svm"}}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_inputs")
    store, results = build_synthetic_results(root, cohort={"D1": (5, 5), "D2": (5, 6)})
    return store, results, index_results(results)


def spec_of(fig_id):
    return next(f for f in PAPER["figures"] if f["id"] == fig_id)


# ----------------------------------------------------------------- results index / provenance
def test_index_and_duplicates(synth, tmp_path):
    _, results, index = synth
    assert {"plv", "ciplv", "none"} <= set(index["condition"])
    assert set(index["setting"]) == {"within_D1", "within_D2", "pooled", "cross_D1_to_D2", "cross_D2_to_D1"}
    one = Path(index.iloc[0]["path"])
    dup_root = tmp_path / "dup"
    shutil.copytree(results, dup_root / "a")
    shutil.copytree(one, dup_root / "b" / "x")
    with pytest.raises(ValueError, match="several runs"):
        index_results(dup_root)
    assert len(index_results(dup_root, duplicates="latest")) == len(index)


# ----------------------------------------------------------------- family 1, 10: no invented numbers
def test_performance_values_come_from_saved_metrics(synth):
    _, _, index = synth
    spec = {**spec_of("Fig3_performance_overall"), "metrics": ["roc_auc", "balanced_accuracy"]}
    fig, data, meta, sources = performance.fig_performance(index, spec, STYLE)
    assert meta["kind"] == "descriptive" and "subject-level" in meta["statistical_unit"]
    for (label, cond, setting, metric), d in data.groupby(["label", "condition", "setting", "metric"]):
        entry = next(e for e in spec["entries"] if e["label"] == label)
        row = select_unique(index, entry, [setting]).query("condition == @cond").iloc[0]
        saved = read_csv(row, "metrics_repeats.csv").query("scope == 'all'")[metric].to_numpy()
        assert np.allclose(np.sort(d["value"]), np.sort(saved), equal_nan=True)     # one point per saved repeat
        assert np.isclose(d["mean"].iloc[0], np.nanmean(saved), equal_nan=True)
    assert {"plv", "ciplv", "none"} <= set(data["condition"])                       # conditions kept apart
    assert all(Path(s).exists() for s in sources)


def test_generalization_ci_is_centred_on_saved_estimate(synth):
    _, _, index = synth
    spec = {"metrics": ["roc_auc"], "entries": [EXP6_FULL, EXP1_SVM]}
    _, data, meta, _ = performance.fig_generalization(index, spec, STYLE)
    assert len(set(data["setting"])) == 5
    for _, r in data.iterrows():
        entry = EXP6_FULL if r["label"] == "Exp6" else EXP1_SVM
        row = select_unique(index, entry, [r["setting"]]).query("condition == @r.condition").iloc[0]
        s = read_summary(row)
        assert np.isclose(r["ci_point"], s["subject_level_repeat_averaged"]["roc_auc"], equal_nan=True)
        assert np.allclose([r["ci_lo"], r["ci_hi"]], s["bootstrap_ci_95"]["roc_auc"], equal_nan=True)


# ----------------------------------------------------------------- family 2: paired, matched folds only
def test_paired_conditions_require_identical_folds(synth, tmp_path):
    _, results, index = synth
    spec = {"metric": "roc_auc", "settings": ["within_D1"], "experiments": ["exp4_band_mean", "exp5_multiband"]}
    _, data, meta, _ = performance.fig_paired_conditions(index, spec, STYLE)
    assert set(data["experiment"]) == {"exp4_band_mean", "exp5_multiband"} and not meta["refused_unmatched_pairs"]
    assert np.allclose(data["diff_ciplv_minus_plv"], data["ciplv"] - data["plv"])
    # tamper with the folds of one |ciPLV| result -> the pair must be refused
    root = tmp_path / "tampered"
    shutil.copytree(results, root)
    idx2 = index_results(root)
    row = idx2[(idx2["experiment"] == "exp4_band_mean") & (idx2["condition"] == "ciplv")
               & (idx2["setting"] == "within_D1")].iloc[0]
    f = pd.read_csv(Path(row["path"]) / "folds.csv")
    f.loc[f.index[:2], "fold"] = (f.loc[f.index[:2], "fold"] + 1) % 3
    f.to_csv(Path(row["path"]) / "folds.csv", index=False)
    _, data2, meta2, _ = performance.fig_paired_conditions(idx2, spec, STYLE)
    assert "exp4_band_mean" not in set(data2["experiment"]) and meta2["refused_unmatched_pairs"]


# ----------------------------------------------------------------- families 3, 4, 12, 14
def test_band_comparison(synth):
    _, data, _, _ = performance.fig_band_comparison(synth[2], spec_of("Fig4_band_specific_exp2_vs_exp3"), STYLE)
    assert set(data["band"]) == {"theta", "alpha1", "alpha2", "beta"}
    assert set(data["condition"]) == {"plv", "ciplv"}


def test_ablation_paired_delta(synth):
    _, data, meta, _ = performance.fig_ablation(synth[2], spec_of("Fig5_exp6_ablations"), STYLE)
    assert set(data["ablation"]) == set(spec_of("Fig5_exp6_ablations")["order"])
    assert np.allclose(data.loc[data["ablation"] == "full", "delta_vs_full"], 0)
    assert set(data["condition"]) == {"plv", "ciplv"}


def test_age_sensitivity(synth):
    _, data, _, _ = performance.fig_age_sensitivity(synth[2], spec_of("S13_age_sensitivity"), STYLE)
    assert set(data["model_type"]) == set(performance.AGE_TYPES)


def test_complexity(synth):
    _, data, _, _ = performance.fig_complexity(synth[2], spec_of("S21a_complexity_parameters"), STYLE)
    assert (data["n_parameters"] > 0).all()
    gnn = data[data["label"].str.startswith("Exp6")]["n_parameters"].iloc[0]
    lr = data[data["label"] == "Exp1 source PSD · LR"]["n_parameters"].iloc[0]
    assert gnn != lr


# ----------------------------------------------------------------- families 8, 9, 11: subject level
def test_roc_matches_saved_subject_auc(synth):
    _, data, meta, _ = subjects.fig_roc(synth[2], {"setting": "within_D1", "entries": [EXP6_FULL],
                                                   "bootstrap_band": 50}, STYLE)
    assert meta["statistical_unit"].startswith("subject")
    curves = data[data["kind"] == "curve"]
    for model, d in curves.groupby("model"):
        cond = "ciplv" if "ciPLV" in model else "plv"
        row = select_unique(synth[2], EXP6_FULL, ["within_D1"]).query("condition == @cond").iloc[0]
        assert np.isclose(d["auc"].iloc[0], read_summary(row)["subject_level_repeat_averaged"]["roc_auc"])
        assert d["n_subjects"].iloc[0] == 10                           # subjects, never windows
    assert (data["kind"] == "bootstrap_band_95").any()


def test_confusion_matches_saved_summary(synth):
    _, data, _, _ = subjects.fig_confusion(synth[2], {"setting": "within_D2", "entries": [EXP1_SVM]}, STYLE)
    row = select_unique(synth[2], EXP1_SVM, ["within_D2"]).iloc[0]
    saved = read_summary(row)["confusion_repeat_averaged"]
    got = {f"{'t' if t == p else 'f'}{'p' if p == 'AD' else 'n'}": c
           for t, p, c in data[["true", "pred", "count"]].itertuples(index=False)}
    assert got == saved and data["count"].sum() == 11
    assert (data["threshold"] == 0.0).all() and (data["score_type"] == "decision_function").all()
    norm = data.groupby("true")["row_fraction"].sum()
    assert np.allclose(norm, 1)


def test_subject_scores_one_point_per_subject_anonymized(synth):
    _, data, meta, _ = subjects.fig_subject_scores(synth[2], {"setting": "within_D1", "entries": [EXP6_FULL],
                                                              "label_subjects": True}, STYLE)
    for _, d in data.groupby("condition"):
        assert d["subject"].is_unique and len(d) == 10
    assert data["subject"].str.match(r"S\d\d$").all()
    assert not data.astype(str).apply(lambda c: c.str.contains(r"D[12]-\d")).any().any()   # no real keys


# ----------------------------------------------------------------- family 7: fusion weights
def test_fusion_weights_and_group_split_guard(synth, tmp_path):
    spec = {"setting": "within_D1", "entries": [EXP6_FULL]}
    _, data, meta, _ = fusion.fig_fusion_weights(synth[2], spec, STYLE)
    assert meta["kind"] == "descriptive" and set(data["band"]) == {"theta", "alpha1", "alpha2", "beta"}
    per_subject = data.groupby(["model", "subject_index"])["weight"].sum()
    assert np.allclose(per_subject, 1, atol=1e-4)                     # softmax weights per subject
    with pytest.raises(InvalidStatistics):
        fusion.fig_fusion_weights(synth[2], {**spec, "split_by_diagnosis": True}, STYLE)
    labels = [f"Exp6 [{c}]" for c in ("PLV", "|ciPLV|")]
    stats = load_stats(synthetic_stats(tmp_path / "s.csv",
                                       [f"fusion:{lab}:{b}" for lab in labels
                                        for b in ("theta", "alpha1", "alpha2", "beta")]))
    _, _, meta2, _ = fusion.fig_fusion_weights(synth[2], {**spec, "split_by_diagnosis": True}, STYLE, stats)
    assert meta2["kind"] == "inferential"
    with pytest.raises(ValueError, match="no learned"):
        fusion.fig_fusion_weights(synth[2], {"setting": "within_D1", "entries": [
            {"label": "mean", "experiment": "exp6_multiband_psd_strength",
             "where": {"model.fusion": "mean", "graph.bands": ["theta", "alpha1", "alpha2", "beta"]}}]}, STYLE)


# ----------------------------------------------------------------- families 5, 6: from the graph run
def test_psd_heatmap_subject_level_and_stats_guard(synth, tmp_path):
    store = synth[0]
    _, data, meta, _ = spectral.fig_psd_heatmap(store, {"kind": "rel"}, STYLE)
    assert meta["kind"] == "descriptive" and not meta["significance_from_stats_file"]
    w = store.windows
    subj_mean = pd.DataFrame(store.arrays["psd_rel"][:, 0, 5]).assign(key=w["key"], y=w["y"]) \
        .groupby("key").agg(v=(0, "mean"), y=("y", "first"))
    expected = subj_mean[subj_mean["y"] == 1]["v"].mean()               # mean of subject means, not windows
    got = data[(data["roi"] == store.roi_names[5]) & (data["band"] == "theta")]["AD_mean"].iloc[0]
    assert np.isclose(got, expected, rtol=1e-5)
    assert data["n_AD"].iloc[0] + data["n_HC"].iloc[0] == 21
    stats = load_stats(synthetic_stats(tmp_path / "p.csv", [f"psd:rel:theta:{store.roi_names[0]}"]))
    _, _, meta2, _ = spectral.fig_psd_heatmap(store, {"kind": "rel"}, STYLE, stats)
    assert meta2["kind"] == "inferential" and len(meta2["significance_from_stats_file"]) == 1
    _, lob, _, _ = spectral.fig_psd_lobes(store, {"kind": "log", "dataset": "D2"}, STYLE)
    assert (lob["n_subjects"] > 0).all() and set(lob["group"]) == {"AD", "HC"}


def test_connectivity_matrices_and_network(synth):
    store = synth[0]
    _, data, meta, _ = connectivity.fig_connectivity_matrices(store, {"connectivity": "ciplv", "bands": ["alpha1"]},
                                                             STYLE)
    assert len(data) == 68 * 67 // 2 and meta["kind"] == "descriptive"
    assert np.allclose(data["AD_minus_HC"], data["AD_mean"] - data["HC_mean"])
    _, net, meta_n, _ = connectivity.fig_connectivity_network(store, {"connectivity": "plv", "display_top_n": 10},
                                                              STYLE)
    assert meta_n["display_only_threshold"]["statistical"] is False
    assert net.groupby("band").size().eq(10).all()
    with pytest.raises(ValueError, match="plv"):
        connectivity.fig_connectivity_matrices(store, {"connectivity": "ciplv_signed"}, STYLE)
    no_coords = store.subset(np.arange(store.n_windows))
    no_coords.roi_coords = None
    with pytest.raises(ValueError, match="roi_coords"):
        connectivity.fig_connectivity_network(no_coords, {"connectivity": "plv"}, STYLE)


# ----------------------------------------------------------------- family 13: exploratory
def test_embeddings(synth):
    store, _, index = synth
    _, data, meta, _ = embedding.fig_psd_embedding(store, {"kinds": ["log", "rel"]}, STYLE)
    assert data["subject"].is_unique and len(data) == 21 and "exploratory" in meta["kind"]
    _, g, meta_g, _ = embedding.fig_graph_embedding(index, spec_of("S20_graph_embedding_pca"), STYLE)
    assert set(g["side"]) == {"train", "test"} and g.groupby("side")["subject"].apply(lambda s: s.is_unique).all()
    try:
        import umap  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            embedding.fig_psd_embedding(store, {"method": "umap"}, STYLE)


# ----------------------------------------------------------------- writer, stats, driver
def test_writer_never_overwrites_and_records_provenance(synth, tmp_path):
    fig, data, meta, sources = spectral.fig_psd_lobes(synth[0], {"kind": "rel"}, STYLE)
    w = FigureWriter(tmp_path / "figs", STYLE, {"x": 1}, synthetic=True)
    rec = w.save(fig, "lobes", data, meta, sources)
    assert {p.suffix for p in (tmp_path / "figs").iterdir()} >= {".pdf", ".png", ".csv", ".json"}
    fig2, *_ = spectral.fig_psd_lobes(synth[0], {"kind": "rel"}, STYLE)
    with pytest.raises(FileExistsError):
        w.save(fig2, "lobes", data, meta, sources)
    man = json.loads(w.write_manifest().read_text())
    assert man["synthetic"] and man["git_commit"] and man["figures"][0]["id"] == "lobes"
    assert rec["statistical_unit"].startswith("subject")
    with pytest.raises(FileExistsError):
        w.write_manifest()


def test_stats_validation(tmp_path):
    good = synthetic_stats(tmp_path / "g.csv", ["a", "b"])
    assert len(load_stats(good)) == 2
    df = pd.read_csv(good)
    for col, val in (("correction_method", ""), ("unit", "window"), ("p_corrected", np.nan)):
        bad = df.copy()
        bad.loc[0, col] = val
        bad.to_csv(tmp_path / "b.csv", index=False)
        with pytest.raises(InvalidStatistics):
            load_stats(tmp_path / "b.csv")
    bad = df.copy()
    bad.loc[0, "p_corrected"] = bad.loc[0, "p_value"] / 10
    bad.to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(InvalidStatistics):
        load_stats(tmp_path / "b.csv")


def test_performance_significance_only_from_stats(synth, tmp_path):
    spec = {"settings": ["within_D1"], "metrics": ["roc_auc"], "entries": [EXP6_FULL, EXP1_SVM]}
    _, _, meta, _ = performance.fig_performance(synth[2], spec, STYLE)
    assert meta["kind"] == "descriptive" and meta["significance_from_stats_file"] == []
    stats = load_stats(synthetic_stats(tmp_path / "s.csv", ["within_D1:roc_auc:Exp6[plv]~Exp1 SVM[none]"]))
    _, _, meta2, _ = performance.fig_performance(synth[2], spec, STYLE, stats)
    assert meta2["kind"] == "inferential" and len(meta2["significance_from_stats_file"]) == 1


def test_make_figures_full_inventory(synth, tmp_path):
    store, results, _ = synth
    w = make_figures(PAPER, tmp_path / "all", results, store, style_path=None, synthetic=True,
                     on_missing="skip")
    made = {f["id"] for f in w.figures}
    skipped = {s["id"] for s in w.skipped}
    assert made | skipped == {f["id"] for f in PAPER["figures"]}
    assert skipped == {"S07_band_specific_other_encoders"}       # encoders not run in the synthetic plan
    assert {f["family"] for f in PAPER["figures"]} == set(FAMILIES)
    for f in w.figures:
        assert f["status"] in ("main", "supplementary") and f["synthetic"]
        assert (tmp_path / "all" / f["data_file"]).exists()
    man = json.loads((tmp_path / "all" / "figure_manifest.json").read_text())
    assert man["config"]["skipped"] and man["sources"]
    with pytest.raises(ValueError):
        make_figures(PAPER, tmp_path / "err", results, store, synthetic=True, on_missing="error")


def test_style_sizes():
    w, _ = figure_size(STYLE, "single")
    assert np.isclose(w * 25.4, 85.0)
    assert np.isclose(figure_size(STYLE, "double")[0] * 25.4, 178.0)


# ----------------------------------------------------------------- frozen pre-result design decisions
def test_frozen_main_figure_set_and_selections():
    main = [f["id"] for f in PAPER["figures"] if f["status"] == "main"]
    assert main == ["Fig2a_source_psd_relative", "Fig2b_plv_group_matrices", "Fig3_performance_overall",
                    "Fig4_band_specific_exp2_vs_exp3", "Fig5_exp6_ablations", "Fig6_generalization",
                    "Fig7_multiband_fusion"]
    fig3 = spec_of("Fig3_performance_overall")
    assert fig3["metrics"] == ["balanced_accuracy", "roc_auc"]
    assert {e["experiment"] for e in fig3["entries"]} == {"exp0_sensor_psd", "exp1_source_psd", "exp4_band_mean",
                                                          "exp5_multiband", "exp6_multiband_psd_strength"}
    # Exp 2/3 never get a representative band: they appear only in band-specific figures
    for f in PAPER["figures"]:
        entries = f.get("entries", []) + ([f["entry"]] if "entry" in f else []) + f.get("groups", [])
        if f["family"] != "band_comparison":
            assert not {e.get("experiment") for e in entries} & {"exp2_conn_minimal", "exp3_conn_psd"}, f["id"]
    roc = [f for f in PAPER["figures"] if f["family"] == "roc"]
    assert roc and all(f["status"] == "supplementary" for f in roc)
    assert [e["label"] for e in roc[0]["entries"]] == ["Exp0 SVM (sensor PSD)", "Exp1 SVM (source PSD)", "Exp6 full"]
    for fam in ("confusion", "subject_scores", "connectivity_network", "psd_embedding", "graph_embedding", "complexity"):
        assert all(f["status"] == "supplementary" for f in PAPER["figures"] if f["family"] == fam)


def test_fit_time_requires_one_environment(synth, tmp_path):
    from adgraph.visualization.performance import IncomparableTiming

    _, results, index = synth
    spec = spec_of("S21b_complexity_fit_time")
    _, data, meta, _ = performance.fig_complexity(index, spec, STYLE)
    assert data["environment"].nunique() == 1 and meta["compute_environment"]["fingerprint"]
    assert {"device", "cpu_model", "versions"} <= set(meta["compute_environment"])
    root = tmp_path / "env"
    shutil.copytree(results, root)
    idx2 = index_results(root)
    row = select_unique(idx2, spec["entries"][0], ["within_D1"]).iloc[0]
    man_path = Path(row["path"]) / "manifest.json"
    man = json.loads(man_path.read_text())
    man["compute_environment"]["fingerprint"] = "other-machine"
    man_path.write_text(json.dumps(man))
    with pytest.raises(IncomparableTiming, match="different computational environments"):
        performance.fig_complexity(idx2, spec, STYLE)
    performance.fig_complexity(idx2, spec_of("S21a_complexity_parameters"), STYLE)    # parameters still fine
    del man["compute_environment"]
    man_path.write_text(json.dumps(man))
    with pytest.raises(IncomparableTiming, match="no compute_environment"):
        performance.fig_complexity(idx2, spec, STYLE)


def test_graph_embedding_requires_one_fitted_model(synth):
    spec = dict(spec_of("S20_graph_embedding_pca"))
    for bad in ({k: v for k, v in spec.items() if k != "repeat"}, {**spec, "fold": "all"}):
        with pytest.raises(ValueError, match="one explicit fitted model"):
            embedding.fig_graph_embedding(synth[2], bad, STYLE)
    _, data, meta, _ = embedding.fig_graph_embedding(synth[2], spec, STYLE)
    assert meta["single_fitted_model"] and (meta["repeat"], meta["fold"]) == (0, 0)
