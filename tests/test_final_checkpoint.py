"""Resumable, never-silently-reused execution of the final framework (methodology log entries 31-32, 58): resume
after interruption, corrupted units, incompatible configuration or code, permutations, storage guard, incomplete
stores, failure recording, variant filters, saved model state and the command line. Synthetic stores only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from adgraph import release
from adgraph.checkpoint import IncompatibleCheckpoint
from adgraph.experiments import runner
from adgraph.experiments.config import experiment_configs
from adgraph.experiments.plan import experiment_variants, run_experiment, variant_matches
from adgraph.stores import open_store, verify_store

REPO = Path(__file__).resolve().parents[1]
EXP1 = experiment_configs()["exp1"]


def _variant(root, name="logreg__psd", exp=EXP1, **ev):
    v = next(x for x in experiment_variants(exp, "lcmv", "B") if x["variant"] == name)
    if ev:
        v = {**v, "evaluation": {**v["evaluation"], **ev}}
    store, rec = open_store(root, "lcmv", "B")
    return store, v


def _count_fits(monkeypatch):
    from adgraph.models import classical

    calls = []
    orig = classical.ClassicalModel.fit

    def fit(self, view):
        calls.append(view.meta["key"].nunique())                          # participants of this fit
        return orig(self, view)
    monkeypatch.setattr(classical.ClassicalModel, "fit", fit)
    return calls


def test_resume_after_an_interrupted_fold(release_root, monkeypatch):
    store, v = _variant(release_root)
    out = release.results_dir(release_root)
    real = runner.select_hparams
    state = {"n": 0}

    def flaky(data, idx, vv, seed):
        state["n"] += 1
        if state["n"] == 3:
            raise RuntimeError("simulated interruption")
        return real(data, idx, vv, seed)
    monkeypatch.setattr(runner, "select_hparams", flaky)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.run_variant(store, v, "xu_dev_cv", out)
    d = runner.result_dir(out, v, "xu_dev_cv")
    st = pd.read_csv(d / "status.csv").set_index("unit")["state"]
    assert list(st[["fold_r000_f00", "fold_r000_f01"]]) == ["completed"] * 2 and st["fold_r000_f02"] == "failed"
    monkeypatch.setattr(runner, "select_hparams", real)
    calls = _count_fits(monkeypatch)
    m = runner.run_variant(store, v, "xu_dev_cv", out)
    outer = [c for c in calls if c > 37]                  # outer fits: ~41 participants; inner CV fits: ~33
    assert len(outer) == 3 and m["folds"]["n_folds"] == 5                 # folds 0-1 reused, 2-4 computed
    calls.clear()
    assert runner.run_variant(store, v, "xu_dev_cv", out) == m and not calls   # complete: nothing recomputed


def test_corrupted_fold_is_recomputed_alone(release_root, monkeypatch):
    store, v = _variant(release_root)
    out = release.results_dir(release_root)
    runner.run_variant(store, v, "xu_dev_cv", out)
    d = runner.result_dir(out, v, "xu_dev_cv")
    (d / "units" / "fold_r000_f03.pred.csv").write_text("corrupted")
    (d / "final.done.json").unlink()
    calls = _count_fits(monkeypatch)
    runner.run_variant(store, v, "xu_dev_cv", out)
    assert len([c for c in calls if c > 37]) == 1


def test_incompatible_configuration_or_code_is_never_reused(release_root, monkeypatch):
    store, v = _variant(release_root)
    out = release.results_dir(release_root)
    runner.run_variant(store, v, "xu_dev_cv", out)
    changed = {**v, "model": {**v["model"], "params": {**v["model"]["params"], "max_iter": 50}}}
    with pytest.raises(IncompatibleCheckpoint):
        runner.run_variant(store, changed, "xu_dev_cv", out)
    from adgraph import checkpoint

    real = checkpoint.code_identity
    monkeypatch.setattr(runner, "code_identity", lambda scope: {**real(scope), "git_commit": "another-commit"})
    with pytest.raises(IncompatibleCheckpoint):
        runner.run_variant(store, v, "xu_dev_cv", out)
    monkeypatch.setattr(runner, "code_identity", real)
    runner.run_variant(store, changed, "xu_dev_cv", out, force=True)      # deliberate: old outputs superseded
    assert any((Path(out) / "_superseded").rglob("metrics.json"))


def test_evaluation_only_settings_reuse_the_folds(release_root, monkeypatch):
    store, v = _variant(release_root)
    out = release.results_dir(release_root)
    runner.run_variant(store, v, "xu_dev_cv", out)
    calls = _count_fits(monkeypatch)
    m = runner.run_variant(store, {**v, "evaluation": {**v["evaluation"], "n_bootstrap": 100}}, "xu_dev_cv", out)
    assert not [c for c in calls if c > 37] and m["participant_bootstrap_ci95"]


def test_permutations_resume(release_root, monkeypatch):
    store, v = _variant(release_root, permutation={"n": 2, "n_repeats": 1})
    out = release.results_dir(release_root)
    m = runner.run_variant(store, v, "xu_dev_cv", out)
    assert m["permutation"]["n"] == 2 and 0 < m["permutation"]["p_value"] <= 1
    (runner.result_dir(out, v, "xu_dev_cv") / "final.done.json").unlink()
    calls = _count_fits(monkeypatch)
    m2 = runner.run_variant(store, v, "xu_dev_cv", out)
    assert not calls and m2["permutation"]["null"] == m["permutation"]["null"]


def test_real_runs_refuse_ephemeral_storage(release_root, tmp_path, monkeypatch):
    from adgraph import storage
    from adgraph.storage import EphemeralStorageError

    monkeypatch.setattr(storage, "in_colab", lambda: True)                # a Colab runtime, results on the VM disk
    store, v = _variant(release_root)
    store.manifest = {**store.manifest, "synthetic": False}
    with pytest.raises(EphemeralStorageError):
        runner.run_variant(store, v, "xu_dev_cv", tmp_path / "vm_disk_results")


def test_incomplete_or_failed_stores_are_refused(release_root):
    from adgraph.stores import store_dir

    d = store_dir(release_root, "lcmv", "A")
    subj = pd.read_csv(d / "subjects.csv")
    subj.loc[0, "status"] = "error: simulated"
    subj.to_csv(d / "subjects.csv", index=False)
    r = verify_store(release_root, "lcmv", "A")
    assert not r["passed"] and not r["checks"]["complete cohort (no failed participant, not a smoke test)"]["passed"]


def test_a_failing_unit_is_recorded_and_the_others_continue(release_root, monkeypatch):
    real = runner.run_variant

    def run(store, v, setting, *a, **k):
        if setting == "xu_holdout" and v["variant"] == "logreg__psd":
            raise RuntimeError("simulated failure")
        return real(store, v, setting, *a, **k)
    monkeypatch.setattr("adgraph.experiments.plan.run_variant", run)
    t = run_experiment(EXP1, release_root, "lcmv", "B", variants=["logreg__psd", "svm__psd"])
    s = t.set_index(["variant", "setting"])["state"]
    assert s[("logreg__psd", "xu_holdout")] == "failed" and (s.drop(("logreg__psd", "xu_holdout")) == "completed").all()
    status = pd.read_csv(release.results_dir(release_root) / "lcmv" / "mode_B" / "dk68" / "none"
                         / "exp1_source_psd_de" / "_run_status.csv")
    assert "simulated failure" in status.loc[status["state"] == "failed", "message"].iloc[0]


def test_variant_filters_match_whole_tokens():
    vs = experiment_variants(experiment_configs()["exp5"], "lcmv", "B")
    v = {x["variant"]: x for x in vs}
    assert variant_matches(v["plv__separate__mean"], ["separate__mean"])
    assert not variant_matches(v["plv__separate__attention"], ["separate__mean"])
    assert not variant_matches(v["ciplv__separate__mean"], ["iplv"])
    assert {x["variant"] for x in vs if variant_matches(x, ["plv"])} == {k for k in v if k.startswith("plv__")}


def test_earm_saves_parameters_gates_and_history_per_fold(release_root, monkeypatch):
    from adgraph.models import trainer

    orig = trainer.GNNModel.__init__
    monkeypatch.setattr(trainer.GNNModel, "__init__",
                        lambda self, m, t, *a, **k: orig(self, m, {**t, "epochs": 1}, *a, **k))
    store, v = _variant(release_root, "plv__full", experiment_configs()["exp7"])
    out = release.results_dir(release_root)
    runner.run_variant(store, v, "xu_dev_cv", out)
    units = sorted((runner.result_dir(out, v, "xu_dev_cv") / "units").glob("*.model.npz"))
    assert len(units) == 5
    with np.load(units[0]) as z:
        assert z["edge_gates"].shape[-2:] == (68, 68) and len(z["history_train_loss"]) == 1
        assert "node_scaler_mean" in z.files


def test_command_line_status_run_and_report(release_root, tmp_path):
    paths = tmp_path / "paths.yaml"
    paths.write_text(yaml.safe_dump({"drive_root": str(tmp_path), "processed_dir": str(release_root),
                                     "cache_dir": str(tmp_path / "cache")}))
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}

    def cli(*a):
        p = subprocess.run([sys.executable, str(REPO / "scripts" / "final.py"), *a, "--paths", str(paths)],
                           cwd=REPO, env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        return p.stdout

    assert "lcmv" in cli("stores") and "True" in cli("stores")
    cli("qc", "--inverse", "wmne", "--mode", "A")
    assert "not started" in cli("status", "--exp", "exp1", "--inverse", "wmne", "--mode", "A")
    cli("run", "--exp", "exp1", "--inverse", "wmne", "--mode", "A", "--variants", "logreg__de",
        "--allow-ephemeral-storage")
    assert "completed" in cli("status", "--exp", "exp1", "--inverse", "wmne", "--mode", "A")
    rep = cli("report", "--exp", "exp1", "--inverse", "wmne", "--mode", "A")
    assert "2 completed results" in rep
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "final.py"), "run", "--exp", "exp5", "--graph",
                        "selected", "--paths", str(paths)], cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "no graph-construction decision" in p.stdout + p.stderr
