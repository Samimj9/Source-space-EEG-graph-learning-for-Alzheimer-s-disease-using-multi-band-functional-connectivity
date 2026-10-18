"""The 18-region investigation and the decision gates (docs/final/parcellation.md, FINAL_RUN_GUIDE.md §3-4):
anatomical checks of the atlas, the window-by-window match of a coarser-atlas store to its 68-ROI store, store
verification, QC and runs on r18 stores with the folds and seeds of dk68, the controlled dk68 vs r18 comparison
(development CV only), the author's atlas decision, the graph decision tied to its atlas, and the holdout refused
outside the frozen final analysis. Synthetic stores only."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from adgraph import release
from adgraph.experiments import decisions
from adgraph.experiments.config import experiment_configs
from adgraph.experiments.plan import run_experiment
from adgraph.source import atlas as atl
from adgraph.stores import open_store, store_dir, verify_store


@pytest.fixture
def both_atlases(tmp_path):
    from adgraph.synthetic import write_release_stores

    for a in ("dk68", "r18"):
        write_release_stores(tmp_path, n_windows=3, psd_effect=0.8, conn_effect=0.8, band_direction=0.5, atlas=a)
    return tmp_path


def _short(monkeypatch, epochs=2):
    from adgraph.models import trainer

    orig = trainer.GNNModel.__init__

    def init(self, model, training, *a, **k):
        orig(self, model, {**training, "epochs": epochs}, *a, **k)
    monkeypatch.setattr(trainer.GNNModel, "__init__", init)


# ------------------------------------------------------------------------------------------- anatomy
@pytest.mark.parametrize("name", atl.ATLASES)
def test_every_atlas_passes_the_anatomical_checks(name):
    checks, rows = atl.verify_atlas(name)
    assert all(c["passed"] for c in checks.values()), {k: c for k, c in checks.items() if not c["passed"]}
    assert len(rows) == atl.N_REGIONS[name] and all(r["contiguous"] for r in rows)


def test_r18_regions_are_the_documented_unions():
    checks, rows = atl.verify_atlas("r18")
    t = pd.DataFrame(rows).set_index("region")
    assert set(t["hemisphere"]) == {"lh", "rh"} and (t["hemisphere"] == "lh").sum() == 9
    assert t.loc["sensorimotor-lh", "member_lobes"] == "frontal+parietal"          # functional grouping, reported
    assert t.loc["insula-rh", "members"] == "insula" and t.loc["parietal-lh", "n_members"] == 4
    assert (t.loc[[r for r in t.index if r.endswith("-lh")], "centroid_x_mm"] < 0).all()
    assert atl.aggregation_matrix("r18").sum(0).tolist() == [1.0] * 68               # each DK label exactly once


def test_a_disconnected_or_cross_hemisphere_region_is_detected(monkeypatch):
    g = {k: (tuple(v[0]), v[1]) for k, v in atl.GROUPINGS["r18"].items()}
    g["insula"] = (("insula", "cuneus"), "insula")                                   # insula + cuneus: not contiguous
    g["occipital"] = (("lateraloccipital", "lingual", "pericalcarine"), "occipital")
    monkeypatch.setitem(atl.GROUPINGS, "r18", g)
    checks, _ = atl.verify_atlas("r18")
    assert not checks["every region is one contiguous piece of cortex"]["passed"]
    assert "insula-lh" in checks["every region is one contiguous piece of cortex"]["detail"]


# ------------------------------------------------------------------------------------------- atlas match
def _r18_copy(tmp_path, monkeypatch):
    """A validated-style 68-ROI store (reference) and an r18 copy of it (target): same windows, participants, split
    and sensor features; 18 ROIs."""
    import test_source_parity as sp

    ref, tgt, arrays = sp._store_pair(tmp_path, monkeypatch)
    shutil.rmtree(tgt)
    tgt = tmp_path / "graphsLX-r18_v1_x"
    shutil.copytree(ref, tgt)
    m = json.loads((tgt / "manifest.json").read_text())
    m["config"] = {**m["config"], "store_names": ["graphsLX-r18", "graphsLX10-r18"],
                   "source": {**m["config"].get("source", {}), "atlas": "r18"}}
    m["graph_run_id"] = "r18-1"
    (tgt / "manifest.json").write_text(json.dumps(m))
    with np.load(ref / "graphs.npz") as z:
        arrays = {k: z[k] for k in z.files}
    r = len(arrays["roi_names"])
    for k in list(arrays):                                # source arrays with 18 ROIs (wrap-padded / cut)
        a = arrays[k]
        if a.ndim >= 2 and a.shape[-1] == r and k != "roi_names" and not k.startswith("sensor"):
            square = a.ndim >= 3 and a.shape[-2] == r
            pad = [(0, 0)] * (a.ndim - (2 if square else 1)) + [(0, max(0, 18 - r))] * (2 if square else 1)
            a = np.pad(a, pad, mode="wrap")[..., :18]
            arrays[k] = a[..., :18, :] if square else a
    arrays["roi_names"] = np.array(atl.region_names("r18"))
    np.savez(tgt / "graphs.npz", **arrays)
    return ref, tgt, arrays


def test_atlas_match_passes_for_the_same_windows_and_sensor_features(tmp_path, monkeypatch):
    from adgraph.sourcegraphs.match import match_atlas

    monkeypatch.syspath_prepend("tests")
    ref, tgt, _ = _r18_copy(tmp_path, monkeypatch)
    res = match_atlas(tgt, ref, "r18")
    assert res["passed"], {k: c for k, c in res["checks"].items() if not c["passed"]}
    assert res["arrays"]["plv"]["shape"][-1] == 18 and res["arrays"]["plv"]["reference_shape"][-1] != 18


@pytest.mark.parametrize("change", ["window", "label", "sensor", "config", "roi_names", "inverse"])
def test_atlas_match_detects_every_difference(tmp_path, monkeypatch, change):
    from adgraph.sourcegraphs.match import match_atlas

    monkeypatch.syspath_prepend("tests")
    ref, tgt, arrays = _r18_copy(tmp_path, monkeypatch)
    m = json.loads((tgt / "manifest.json").read_text())
    if change == "window":
        w = pd.read_csv(tgt / "windows.csv")
        w.loc[3, "stop_sample"] += 250
        w.to_csv(tgt / "windows.csv", index=False)
    elif change == "label":
        s = pd.read_csv(tgt / "subjects.csv")
        s.loc[0, "y"] = 1 - s.loc[0, "y"]
        s.to_csv(tgt / "subjects.csv", index=False)
    elif change == "sensor":
        arrays["sensor_psd_rel"] = arrays["sensor_psd_rel"] * 1.01
        np.savez(tgt / "graphs.npz", **arrays)
    elif change == "config":
        m["config"]["preprocessing"] = {**m["config"].get("preprocessing", {}), "sfreq": 200.0}   # sampling
        (tgt / "manifest.json").write_text(json.dumps(m))
    elif change == "roi_names":
        arrays["roi_names"] = arrays["roi_names"][::-1]
        np.savez(tgt / "graphs.npz", **arrays)
    else:
        m["source_operator"]["method"] = "wmne"
        (tgt / "manifest.json").write_text(json.dumps(m))
    assert not match_atlas(tgt, ref, "r18")["passed"]


def test_r18_stores_verify_only_with_their_atlas_match(both_atlases):
    for inv in ("lcmv", "wmne"):
        for mode in ("B", "A"):
            r = verify_store(both_atlases, inv, mode, "r18")
            assert r["passed"], {k: c for k, c in r["checks"].items() if not c["passed"]}
    d = store_dir(both_atlases, "wmne", "B", "r18")
    assert "graphsWX-r18_v1_" in d.name and d.parent.name == "D3_WMNE_XU"         # outside FINAL_v1/
    assert store_dir(both_atlases, "lcmv", "B", "dk68").name.endswith("_corr1")    # the 68-ROI stores are unchanged
    (d / "atlas_match.json").unlink()
    r = verify_store(both_atlases, "wmne", "B", "r18")
    assert not r["passed"] and not r["checks"]["atlas_match.json passed against the 68-ROI store of the same "
                                               "inverse and mode"]["passed"]


# ------------------------------------------------------------------------------------------- QC, runs on r18
def test_qc_of_an_r18_store_reports_everyone_and_excludes_no_one(both_atlases):
    from adgraph.qc.report import require_qc

    store, _ = open_store(both_atlases, "lcmv", "B", "r18")
    rep = require_qc(store, both_atlases, "lcmv", "B", "r18")
    assert rep["summary"]["n_participants"] == 65 and not rep["summary"]["category_A_participants"]
    assert (release.root(both_atlases) / "qc" / "lcmv_B_r18" / "qc.done.json").exists()


def test_r18_runs_use_the_same_folds_and_seeds_as_dk68(both_atlases, monkeypatch):
    _short(monkeypatch)
    for exp, var in (("exp1", "logreg__psd"), ("exp4", "plv__psd")):
        dirs = {}
        for a in ("dk68", "r18"):
            t = run_experiment(experiment_configs()[exp], both_atlases, "lcmv", "B", a,
                               None if exp == "exp1" else "dense", [var], ["xu_dev_cv"])
            assert (t["state"] == "completed").all() and set(t["setting"]) == {"xu_dev_cv"}
            g = "none" if exp == "exp1" else "dense"
            dirs[a] = next((release.results_dir(both_atlases) / "lcmv" / "mode_B" / a / g).glob(f"*/{var}/xu_dev_cv"))
        f = {a: pd.read_csv(d / "folds.csv").sort_values(["fold", "side", "key"]).reset_index(drop=True)
             for a, d in dirs.items()}
        pd.testing.assert_frame_equal(f["dk68"][["fold", "side", "key"]], f["r18"][["fold", "side", "key"]])
        s = {a: pd.read_csv(d / "training.csv")["seed"].tolist() for a, d in dirs.items()}
        assert s["dk68"] == s["r18"]
        assert not (dirs["r18"].parent / "xu_holdout").exists()                      # development CV only


# ------------------------------------------------------------------------------------------- the comparison
def test_roi_comparison_is_paired_development_only_and_never_forces_a_winner(both_atlases, monkeypatch):
    from adgraph.experiments import roi_comparison as rc
    from adgraph.experiments.report import collect

    cfg = {**rc.config(), "inverses": ["lcmv"], "models": {"exp1_source_psd_de": ["logreg__psd"]}, "n_bootstrap": 200}
    for r in rc.runs(cfg):
        run_experiment(experiment_configs()[r["exp"]], both_atlases, r["inverse"], "B", r["atlas"], None,
                       [r["variant"]], ["xu_dev_cv"])
    table = collect(release.results_dir(both_atlases))
    cls, missing = rc.classification(table, cfg)
    assert not missing and len(cls) == 1
    row = cls.iloc[0]
    assert row["same_participants"] and row["n_participants"] == 52
    assert row["auc_diff_boot_low"] <= row["auc_diff_r18_minus_dk68"] <= row["auc_diff_boot_high"]
    assert row["auc_diff_r18_minus_dk68"] == pytest.approx(row["auc_r18"] - row["auc_dk68"], abs=1e-9)
    summ = rc.summary(cls, missing, cfg)
    assert summ["complete"] and "superior" not in summ["statement"]
    fake = table.assign(setting="xu_holdout", auc=0.99)                              # holdout rows are never read
    assert rc.classification(pd.concat([table, fake]), cfg)[0]["auc_r18"].iloc[0] == row["auc_r18"]
    incomplete = rc.summary(*rc.classification(table, {**cfg, "inverses": ["lcmv", "wmne"]}),
                            {**cfg, "inverses": ["lcmv", "wmne"]})
    assert not incomplete["complete"] and "incomplete" in incomplete["statement"]
    md = rc.report_markdown(cls, summ, {}, cfg)
    assert "Source localization" in md and "Classification" in md and "holdout not evaluated" in md


def test_descriptive_summary_wording():
    from adgraph.experiments import roi_comparison as rc

    cfg = rc.config()
    cells = [(e, v, i) for e, vs in cfg["models"].items() for v in vs for i in cfg["inverses"]]

    def table(diffs):
        return pd.DataFrame([{"experiment": e, "variant": v, "inverse": i, "auc_diff_r18_minus_dk68": d,
                              "auc_diff_boot_low": d - 0.1, "auc_diff_boot_high": d + 0.1}
                             for (e, v, i), d in zip(cells, diffs)])
    assert "no material" in rc.summary(table([0.01] * len(cells)), [], cfg)["statement"]
    assert "tendency towards r18" in rc.summary(table([0.05] * len(cells)), [], cfg)["statement"]
    assert "inconsistent" in rc.summary(table([0.2, -0.1] * (len(cells) // 2)), [], cfg)["statement"]


# ------------------------------------------------------------------------------------------- decisions
def test_the_atlas_is_recorded_by_the_author_once(tmp_path):
    with pytest.raises(decisions.DecisionMissing):
        decisions.selected_atlas(tmp_path)
    with pytest.raises(ValueError):
        decisions.record_atlas(tmp_path, "r18", "")                                  # a reason is required
    decisions.record_atlas(tmp_path, "r18", "no forced redundancy; cross-talk halved")
    assert decisions.selected_atlas(tmp_path) == "r18"
    assert decisions.record_atlas(tmp_path, "r18", "the same decision again")["atlas"] == "r18"
    with pytest.raises(RuntimeError, match="never overwritten"):
        decisions.record_atlas(tmp_path, "dk68", "changed my mind about it")


def test_the_graph_decision_belongs_to_its_atlas(tmp_path):
    from adgraph.checkpoint import atomic_write_json

    atomic_write_json(decisions.graph_path(tmp_path), {"selected": "knn_union_f20", "atlas": "r18", "reason": "x"})
    assert decisions.graph_decision(tmp_path, "r18")["selected"] == "knn_union_f20"
    with pytest.raises(decisions.DecisionMissing, match="atlas"):
        decisions.graph_decision(tmp_path, "dk68")


def test_selection_runs_on_the_recorded_atlas():
    from adgraph.experiments.selection import rules, select

    r = rules()["graph_construction"]
    rows = [{"inverse": inv, "mode": "B", "atlas": a, "graph": g, "experiment": e, "variant": v,
             "setting": "xu_dev_cv", "auc": 0.7 + (0.1 if (g == "knn_union_f10" and a == "r18") else 0)}
            for a in ("dk68", "r18") for g in [r["default"]] + r["candidates"] + r["reported_only"]
            for e, vs in r["reference"].items() for v in vs for inv in r["inverses"]]
    t = pd.DataFrame(rows)
    with pytest.raises(ValueError):
        select(t)
    assert select(t, atlas="dk68")["selected"] == "dense"
    d = select(t, atlas="r18")
    assert d["selected"] == "knn_union_f10" and d["atlas"] == "r18"


def test_planned_holdout_units_exclude_exploratory_and_sensitivity_runs():
    u = pd.DataFrame(decisions.planned_units("r18", "knn_union_f10"))
    assert "exp7" not in set(u["exp"]) and set(u["mode"]) == {"B"}
    assert set(u.loc[u["exp"] == "exp0", "atlas"]) == {"dk68"} and set(u.loc[u["exp"] == "exp0", "inverse"]) == {"sensor"}
    assert set(u.loc[u["exp"] == "exp1", "graph"]) == {"none"} and set(u.loc[u["exp"] == "exp5", "graph"]) == {"knn_union_f10"}
    assert set(u.loc[u["exp"] != "exp0", "atlas"]) == {"r18"}
    assert len(u) == 4 + 2 * (4 + 10 + 10 + 4 + 8 + 10 + 10)


def test_freeze_needs_the_decisions_and_complete_development_cv(tmp_path):
    from adgraph.checkpoint import atomic_write_json

    empty = pd.DataFrame(columns=["setting"])
    with pytest.raises(decisions.DecisionMissing):
        decisions.freeze(tmp_path, empty, "abc")
    decisions.record_atlas(tmp_path, "dk68", "resolution evidence inconclusive; keep the reference")
    atomic_write_json(decisions.graph_path(tmp_path), {"selected": "dense", "atlas": "dk68", "reason": "kept"})
    units = decisions.planned_units("dk68", "dense")
    dev = pd.DataFrame([{**u, "setting": "xu_dev_cv", "auc": 0.7} for u in units])
    with pytest.raises(decisions.DecisionMissing, match="development-CV"):
        decisions.freeze(tmp_path, dev.iloc[1:], "abc")
    d = decisions.freeze(tmp_path, dev, "abc")
    assert len(d["units"]) == len(units) and d["git_commit"] == "abc"
    assert decisions.freeze(tmp_path, dev, "abc")["frozen_at"] == d["frozen_at"]      # written once


def test_the_holdout_of_a_real_store_needs_the_frozen_final_analysis(release_root):
    from adgraph.checkpoint import atomic_write_json
    from adgraph.qc.report import require_qc

    store, rec = open_store(release_root, "lcmv", "B")
    qc = require_qc(store, release_root, "lcmv", "B", "dk68")
    store.manifest = {**store.manifest, "synthetic": False}
    cfg = experiment_configs()["exp1"]
    kw = dict(variants=["logreg__psd"], store=store, store_record=rec, qc_report=qc)
    with pytest.raises(decisions.HoldoutNotFrozen):
        run_experiment(cfg, release_root, "lcmv", "B", settings=["xu_holdout"], **kw)
    with pytest.raises(decisions.HoldoutNotFrozen):                                   # default: both settings
        run_experiment(cfg, release_root, "lcmv", "B", **kw)
    assert (run_experiment(cfg, release_root, "lcmv", "B", settings=["xu_dev_cv"], **kw)["state"]
            == "completed").all()                                                      # development CV is always fine
    unit = {"exp": "exp1", "experiment": "exp1_source_psd_de", "variant": "svm__psd", "inverse": "lcmv",
            "mode": "B", "atlas": "dk68", "graph": "none"}
    atomic_write_json(decisions.final_path(release_root), {"units": [unit], "git_commit": None})
    with pytest.raises(decisions.HoldoutNotFrozen, match="not part"):
        run_experiment(cfg, release_root, "lcmv", "B", settings=["xu_holdout"], **kw)
    atomic_write_json(decisions.final_path(release_root), {"units": [{**unit, "variant": "logreg__psd"}],
                                                           "git_commit": None})
    t = run_experiment(cfg, release_root, "lcmv", "B", settings=["xu_holdout"], **kw)
    assert (t["state"] == "completed").all()


# ------------------------------------------------------------------------------------------- merged labels
def test_r18_regions_are_merged_labels_extracted_on_their_own_not_averages_of_dk_signals():
    """The store builder extracts a coarse region with mean_flip over the MERGED label (``atlas.merged_labels``):
    its sign flips come from the merged label's own normals and every source has equal weight. The equal-weight
    average of the member DK signals (each with its own flip) is a different signal; it is never used."""
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(0)
    names = atl.region_names("dk68")
    sizes = {n: int(rng.integers(3, 9)) for n in names}
    off, labs = {"lh": 0, "rh": 0}, []
    for n in names:
        h = n.rsplit("-", 1)[1]
        labs.append(mne.Label(np.arange(off[h], off[h] + sizes[n]), hemi=h, name=n))
        off[h] += sizes[n]
    merged = atl.merged_labels(labs, "r18")
    assert [m.name for m in merged] == atl.region_names("r18")
    for m in merged:
        mem = atl.members("r18")[m.name]
        assert m.hemi == m.name.rsplit("-", 1)[1]
        want = np.sort(np.concatenate([lab.vertices for lab in labs if lab.name in mem]))
        assert np.array_equal(np.sort(m.vertices), want)                       # union of whole member labels
    assert atl.merged_labels(labs, "dk68") == labs                              # dk68: the stores' labels, unchanged
    # two members with opposite normals and unequal sizes: the merged extraction is not the average of members
    n_a, n_b = 6, 18
    src = [{"vertno": np.arange(n_a + n_b), "nn": np.r_[np.tile([0, 0, 1.0], (n_a, 1)), np.tile([0, 0, -1.0], (n_b, 1))],
            "type": "surf", "rr": rng.normal(size=(n_a + n_b, 3))} for _ in range(2)]
    a = mne.Label(np.arange(n_a), hemi="lh", name="a-lh")
    b = mne.Label(np.arange(n_a, n_a + n_b), hemi="lh", name="b-lh")
    x = rng.standard_normal((n_a + n_b, 500))
    sig = lambda lab: (mne.label_sign_flip(lab, src)[:, None] * x[lab.vertices]).mean(0)
    merged_signal = sig(a + b)
    averaged = (sig(a) + sig(b)) / 2
    assert not np.allclose(np.abs(np.corrcoef(merged_signal, averaged)[0, 1]), 1.0, atol=1e-3)
