"""Regression tests for issues found in the pre-real-data audit (synthetic data only)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adgraph.experiments import runner
from adgraph.experiments.config import expand_variants, load_experiment, set_dotted
from adgraph.experiments.plan import variant_matches
from adgraph.graphs.dataset import build_dataset
from adgraph.io.eeg import normalize_channel_name
from adgraph.io.participants import normalize_participants
from adgraph.synthetic import make_synthetic_store
from test_checkpoint import CFG, cohort  # noqa: F401  (fixture)

EXP = Path("configs/experiments")


def _variant(path, name, **over):
    v = next(x for x in expand_variants(load_experiment(EXP / path)) if x["variant"] == name)
    for k, val in over.items():
        v = set_dotted(v, k, val)
    return v


# ----------------------------------------------------------------- experiments
def test_recurring_outer_split_does_not_collide_with_cached_inner_trials(store, tmp_path):
    """D1 has only 45 x 45 possible 5-fold test folds, so splits recur across repeats; each
    recurrence is a different unit (different seed) and must not be mistaken for another."""
    v = _variant("exp1_source_psd.yaml", "logreg", **{"evaluation.n_repeats": 4, "evaluation.n_bootstrap": 20})
    data = build_dataset(store, v)
    meta = data.meta[data.meta["dataset"] == "D1"].drop_duplicates("key")
    _, _, splits = runner._splits(data.meta, v, "within_D1", "y")
    keys = [runner.split_id(sp.train_keys, sp.test_keys) for sp in splits]
    assert len(keys) > len(set(keys)), "fixture must contain a recurring outer split"
    assert len(meta) == 20
    runner.run_variant(store, v, "within_D1", tmp_path, data=data)   # used to raise IncompatibleCheckpoint
    again = runner.run_variant(store, v, "within_D1", tmp_path, data=data)
    assert np.isfinite(again["subject_level"]["all"]["roc_auc"]["mean"])


def test_dataset_probe_permutation_changes_labels(store):
    v = _variant("probe_dataset.yaml", "default")
    data = build_dataset(store, v)
    perm = runner.permute_subject_labels(data, np.random.default_rng(0))
    before = data.meta.drop_duplicates("key").set_index("key")[data.target]
    after = perm.meta.drop_duplicates("key").set_index("key")[data.target]
    assert (before != after.loc[before.index]).any()
    assert sorted(after) == sorted(before)                           # a permutation, same class sizes


def test_result_dir_separates_non_primary_analyses(tmp_path):
    v = _variant("exp6_multiband_psd_strength.yaml", "plv__full")
    assert runner.result_dir(tmp_path, v, "pooled") == tmp_path / "exp6_multiband_psd_strength" / "plv__full" / "pooled"
    probe = set_dotted(v, "target", "dataset")
    loso = set_dotted(v, "evaluation.scheme", "loso")
    assert runner.result_dir(tmp_path, probe, "pooled").parent.name == "plv__full__target-dataset"
    assert runner.result_dir(tmp_path, loso, "pooled").parent.name == "plv__full__scheme-loso"


def test_variant_filter_matches_whole_tokens():
    vs = expand_variants(load_experiment(EXP / "exp6_multiband_psd_strength.yaml"))
    plv = [v["variant"] for v in vs if variant_matches(v, ["plv"])]
    assert plv and all(n.startswith("plv__") for n in plv)
    assert [v["variant"] for v in vs if variant_matches(v, ["plv__full"])] == ["plv__full"]
    assert {v["variant"] for v in vs if variant_matches(v, ["full"])} == {"plv__full", "ciplv__full"}


def test_gnn_loss_gives_each_batch_weight_by_its_windows():
    """Loss = sum(w * l) / batch size, with weights of mean 1 over the training set: a batch of
    low-weight windows (a subject with many windows) contributes less, as one vote per subject requires."""
    import torch
    from torch.nn import functional as F

    logits, y = torch.zeros(4), torch.tensor([1.0, 1.0, 0.0, 0.0])
    w_low, w_high = torch.full((4,), 0.5), torch.full((4,), 1.5)
    loss = lambda w: (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / 4  # noqa: E731
    assert loss(w_high) == pytest.approx(3 * loss(w_low))
    src = Path("src/adgraph/models/trainer.py").read_text()
    assert ".sum() / len(idx)" in src and "/ w[idx].sum()" not in src


def test_exp6_full_saves_embeddings_for_figure_s20():
    vs = {v["variant"]: v for v in expand_variants(load_experiment(EXP / "exp6_multiband_psd_strength.yaml"))}
    assert vs["plv__full"]["evaluation"]["save_embeddings"] and not vs["plv__no_psd"]["evaluation"]["save_embeddings"]


# ----------------------------------------------------------------- inputs
def test_unrecognised_group_is_an_error_but_footers_are_skipped():
    ok = normalize_participants(pd.DataFrame({"ID": ["Sub1", "Sub2", "Total", None, "Sub3"],
                                              "Group": ["AD", "hc", None, None, None],
                                              "Gender": ["F", None, None, None, None]}), "D1")
    assert ok["key"].tolist() == ["D1-1", "D1-2"] and ok["sex"].tolist() == ["F", ""]
    with pytest.raises(ValueError, match="unrecognised Group"):
        normalize_participants(pd.DataFrame({"ID": ["Sub1", "Sub2"], "Group": ["AD", "Control"]}), "D1")


@pytest.mark.parametrize("label", ["Fp1", "EEG Fp1-REF", "FP1-A1", "Fp1-M2", "fp1_A1A2", "Fp1."])
def test_reference_suffixes_are_recognised(label):
    assert normalize_channel_name(label, ["Fp1", "M1", "M2"]) == "Fp1"
    assert normalize_channel_name("M1", ["Fp1", "M1", "M2"]) == "M1"


def test_graph_run_refuses_recordings_without_label(cohort):
    pl, paths, counts, tmp = cohort
    shutil.copy(Path(paths["dataset1"]["eeg_dir"]) / "sub1.edf", Path(paths["dataset1"]["eeg_dir"]) / "sub7.edf")
    with pytest.raises(ValueError, match=r"no AD/HC label.*D1-7"):
        pl.run(paths, CFG, tmp / "out", tmp / "sd")
    assert counts["process"] == []                                  # refused before any processing


# ----------------------------------------------------------------- figures
@pytest.fixture(scope="module")
def results(synthetic_results):
    return synthetic_results                           # shared session fixture (conftest)


def _paper():
    import yaml

    return yaml.safe_load(Path("configs/figures/paper.yaml").read_text())


def test_missing_results_are_reported_not_silent(results, tmp_path):
    from adgraph.visualization.figures import make_figures

    store, res = results
    partial = tmp_path / "partial"
    shutil.copytree(res, partial)
    for d in partial.glob("exp5_multiband/*"):                      # Exp5 "still running"
        shutil.rmtree(d)
    w = make_figures(_paper(), tmp_path / "figs", partial, store, synthetic=True, only=["Fig3_performance_overall"])
    assert [f["id"] for f in w.figures] == ["Fig3_performance_overall"]
    inc = {f["id"]: f for f in w.incomplete}
    missing = inc["Fig3_performance_overall"]["missing_inputs"]
    assert {m["entry"] for m in missing} == {"Exp5 multi-band · attn"}
    meta = json.loads((tmp_path / "figs" / "Fig3_performance_overall_meta.json").read_text())
    assert meta["missing_inputs"] == missing


def test_code_errors_are_not_reported_as_missing_inputs(results, tmp_path, monkeypatch):
    from adgraph.visualization import figures

    store, res = results
    fn, kind = figures.FAMILIES["performance"]
    monkeypatch.setitem(figures.FAMILIES, "performance", (lambda *a: (_ for _ in ()).throw(IndexError("bug")), kind))
    with pytest.raises(RuntimeError, match="code errors.*IndexError"):
        figures.make_figures(_paper(), tmp_path / "f", res, store, synthetic=True,
                             only=["Fig3_performance_overall", "Fig2a_source_psd_relative"])
    assert any((tmp_path / "f").rglob("Fig2a*.png"))                 # other figures still written


def test_figures_refuse_results_from_another_graph_run(results, tmp_path):
    from adgraph.visualization.figures import make_figures

    store, res = results
    other = make_synthetic_store({"D1": (5, 5), "D2": (5, 6)}, windows_per_subject=(3, 3), seed=7)
    with pytest.raises(ValueError, match="graph run"):
        make_figures(_paper(), tmp_path / "f", res, other, synthetic=True, only=["Fig3_performance_overall"])


def test_index_excludes_non_primary_runs(results, tmp_path):
    from adgraph.visualization.results import index_results

    _, res = results
    root = tmp_path / "r"
    shutil.copytree(res, root)
    idx = index_results(root)
    d = Path(idx["path"].iloc[0])
    man = json.loads((d / "manifest.json").read_text())
    man["variant_config"]["target"] = "dataset"
    (d / "manifest.json").write_text(json.dumps(man))
    assert len(index_results(root)) == len(idx) - 1
    assert len(index_results(root, primary_only=False)) == len(idx)
    assert set(idx["graph_run_id"]) != {""}
