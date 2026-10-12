"""End-to-end runs of every experiment config on synthetic data (tiny CV, few epochs)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from adgraph.experiments.config import expand_variants, load_experiment, set_dotted
from adgraph.experiments.runner import dry_run, run_variant
from adgraph.synthetic import make_synthetic_store

EXP = Path("configs/experiments")
FAST = {"evaluation.n_repeats": 1, "evaluation.k": 3, "evaluation.inner_k": 2, "evaluation.n_bootstrap": 50,
        "training.epochs": 2, "model.hidden": 8}


def fast(v, **extra):
    for k, val in {**FAST, **extra}.items():
        if k.split(".")[0] in v or k.startswith("evaluation"):
            v = set_dotted(v, k, val)
    v["hparams"] = {k: val[:2] for k, val in (v.get("hparams") or {}).items()}
    return v


@pytest.fixture(scope="module")
def small_store():
    return make_synthetic_store({"D1": (5, 5), "D2": (5, 6)}, windows_per_subject=(3, 4), psd_effect=0.8,
                                conn_effect=0.8, seed=1)


def first_variant(name):
    return expand_variants(load_experiment(EXP / name))[0]


def _check_outputs(out, store, setting):
    pred = pd.read_csv(out / "predictions_windows.csv")
    subj = pd.read_csv(out / "predictions_subjects.csv")
    folds = pd.read_csv(out / "folds.csv")
    summary = json.loads((out / "summary.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    for col in ("window_id", "key", "dataset", "subject_id", "y", "score", "repeat", "fold"):
        assert col in pred
    # each test window predicted exactly once per repeat; never a training subject
    assert not pred.duplicated(["repeat", "window_id"]).any()
    test_side = folds[folds["side"] == "test"]
    assert (subj.groupby("repeat")["key"].nunique() == test_side.groupby("repeat")["key"].nunique()).all()
    merged = pred.merge(folds[folds["side"] == "train"], on=["repeat", "fold", "key"], how="inner")
    assert merged.empty
    assert "roc_auc" in summary["subject_level"]["all"] and "bootstrap_ci_95" in summary
    assert manifest["setting"] == setting
    return pred, summary


@pytest.mark.parametrize("config", ["exp0_sensor_psd.yaml", "exp1_source_psd.yaml", "baseline_age.yaml"])
@pytest.mark.parametrize("setting", ["within_D1", "within_D2", "pooled", "cross_D1_to_D2", "cross_D2_to_D1"])
def test_classical_experiments_all_settings(small_store, tmp_path, config, setting):
    v = fast(first_variant(config))
    run_variant(small_store, v, setting, tmp_path)
    pred, summary = _check_outputs(tmp_path / v["experiment"] / v["variant"] / setting, small_store, setting)
    if setting == "pooled":
        assert {"D1", "D2"} <= set(summary["subject_level"])                  # per-dataset breakdown


@pytest.mark.parametrize("config", ["exp2_conn_minimal.yaml", "exp3_conn_psd.yaml", "exp4_band_mean.yaml",
                                    "exp5_multiband.yaml", "exp6_multiband_psd_strength.yaml"])
def test_gnn_experiments(small_store, tmp_path, config):
    pytest.importorskip("torch")
    for v in expand_variants(load_experiment(EXP / config))[:2]:
        v = fast(v)
        for setting in ("within_D2", "cross_D1_to_D2"):
            run_variant(small_store, v, setting, tmp_path)
            pred, _ = _check_outputs(tmp_path / v["experiment"] / v["variant"] / setting, small_store, setting)
            if v["graph"]["band_mode"] == "multi":
                assert len([c for c in pred if c.startswith("band_weight_")]) == len(v["graph"]["bands"])


def test_classical_learns_planted_effect(tmp_path):
    store = make_synthetic_store(psd_effect=1.0, seed=3)
    v = fast(first_variant("exp1_source_psd.yaml"))
    summary = run_variant(store, v, "within_D1", tmp_path)
    assert summary["subject_level"]["all"]["roc_auc"]["mean"] > 0.9


def test_probe_and_permutation(small_store, tmp_path):
    v = fast(first_variant("probe_dataset.yaml"), **{"evaluation.permutation": {"n": 2, "n_repeats": 1}})
    summary = run_variant(small_store, v, "pooled", tmp_path)
    assert summary["permutation"]["n"] == 2 and 0 < summary["permutation"]["p_value"] <= 1
    with pytest.raises(ValueError):
        run_variant(small_store, v, "within_D1", tmp_path)


def test_cross_dataset_gate_blocks_unverified_runs(small_store, tmp_path):
    store = make_synthetic_store({"D1": (3, 3), "D2": (3, 3)}, windows_per_subject=(2, 3), seed=2)
    store.manifest["source_operator"]["shared_across_datasets"] = False
    v = fast(first_variant("exp1_source_psd.yaml"))
    with pytest.raises(RuntimeError, match="gate"):
        run_variant(store, v, "cross_D1_to_D2", tmp_path)
    run_variant(store, v, "within_D1", tmp_path)                            # within-dataset unaffected


def test_outputs_never_overwritten(small_store, tmp_path):
    v = fast(first_variant("exp0_sensor_psd.yaml"))
    first = run_variant(small_store, v, "within_D1", tmp_path)
    again = run_variant(small_store, v, "within_D1", tmp_path)                 # resumed: reused, not rewritten
    assert json.dumps(again, sort_keys=True) == json.dumps(first, sort_keys=True)
    with pytest.raises(FileExistsError):
        run_variant(small_store, v, "within_D1", tmp_path, resume=False)


def test_dry_run_reports_shapes(small_store):
    vs = [fast(v) for v in expand_variants(load_experiment(EXP / "exp6_multiband_psd_strength.yaml"))[:3]]
    vs.append(fast(first_variant("exp0_sensor_psd.yaml")))
    table = dry_run(small_store, vs, ["within_D1", "pooled"])
    shapes = json.loads(table.iloc[0]["shapes"])
    assert shapes["x"][1:] == [4, 68, 3] and shapes["adj"][1:] == [4, 68, 68]
    assert json.loads(table.iloc[-1]["shapes"])["X"][1] == 2 * 4 * 30


@pytest.mark.parametrize("config", ["age/exp1_source_psd_plus_age.yaml", "age/exp0_sensor_psd_plus_age.yaml",
                                    "age/exp6_multiband_psd_strength_plus_age.yaml"])
def test_eeg_plus_age_runs(small_store, tmp_path, config):
    v = fast(first_variant(config))
    if v["input"] == "graph":
        pytest.importorskip("torch")
    run_variant(small_store, v, "within_D1", tmp_path)
    m = json.loads((tmp_path / v["experiment"] / v["variant"] / "within_D1" / "manifest.json").read_text())
    assert m["data_info"]["covariates"] == ["age"]


def test_svm_variant_reports_decision_scores(small_store, tmp_path):
    vs = {v["model"]["name"]: v for v in expand_variants(load_experiment(EXP / "exp1_source_psd.yaml"))}
    v = fast(vs["svm"])
    summary = run_variant(small_store, v, "within_D1", tmp_path)
    assert summary["score"] == {"type": "decision_function", "threshold": 0.0,
                                "subject_aggregation": "mean of window scores"}
    pred = pd.read_csv(tmp_path / v["experiment"] / v["variant"] / "within_D1" / "predictions_windows.csv")
    assert (pred["score"] < 0).any()                                    # margins, not probabilities
    vc = fast(set_dotted(vs["svm"], "model.calibration", "grouped_platt"))
    vc["variant"], vc["hparams"] = "svm_grouped_platt", {}   # tiny cohort: inner folds too small to calibrate
    s2 = run_variant(small_store, vc, "within_D1", tmp_path)
    assert s2["score"]["type"] == "probability" and s2["score"]["threshold"] == 0.5
    lr = run_variant(small_store, fast(vs["logreg"]), "within_D1", tmp_path)
    assert lr["score"]["type"] == "probability"
