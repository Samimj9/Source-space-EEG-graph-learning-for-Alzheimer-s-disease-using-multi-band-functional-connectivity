"""Interruption / restart behaviour of graph construction and experiments (synthetic only)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adgraph.checkpoint import (IncompatibleCheckpoint, StatusTable, atomic_write_text, check_unit, stable_hash,
                                write_marker)
from adgraph.config import load_yaml
from adgraph.experiments.config import expand_variants, load_experiment, set_dotted
from adgraph.experiments import runner
from adgraph.synthetic import make_synthetic_store

CFG = load_yaml("configs/pipeline.yaml")
EXP = Path("configs/experiments")


class SimulatedCrash(BaseException):
    """Stands in for a Colab disconnect / kill: not caught by the per-subject error handling."""


# ======================================================================= primitives
def test_unit_states(tmp_path):
    ident = {"a": 1}
    assert check_unit(tmp_path, "u", ident)[0] == "missing"
    atomic_write_text(tmp_path / "u.txt", "payload")
    assert check_unit(tmp_path, "u", ident)[0] == "missing"            # payload without marker != complete
    write_marker(tmp_path, "u", ident, ["u.txt"])
    assert check_unit(tmp_path, "u", ident)[0] == "complete"
    assert check_unit(tmp_path, "u", {"a": 2})[0] == "incompatible"
    (tmp_path / "u.txt").write_text("payloaD")                          # corrupted payload
    assert check_unit(tmp_path, "u", ident)[0] == "corrupt"
    (tmp_path / "u.done.json").write_text("{trunc")                     # torn marker
    assert check_unit(tmp_path, "u", ident)[0] == "corrupt"
    assert not list(tmp_path.glob(".*tmp-*"))                           # no temp files left behind


def test_status_table(tmp_path):
    t = StatusTable(tmp_path / "status.csv")
    t.ensure(["s1", "s2"])
    t.set("s1", "running")
    t.set("s1", "completed")
    t2 = StatusTable(tmp_path / "status.csv")
    assert t2.get("s1") == "completed" and t2.get("s2") == "pending"
    with pytest.raises(ValueError):
        t2.set("s1", "done")


# ======================================================================= graph construction
@pytest.fixture
def cohort(tmp_path, monkeypatch):
    """4 synthetic EDF recordings + metadata, and stubbed fsaverage operator (counts builds)."""
    pytest.importorskip("edfio")
    from test_end_to_end import _write_edf

    import adgraph.pipeline as pl

    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir(), d2.mkdir()
    names = CFG["channels"]["recorded"]
    for f, unit in [(d1 / "sub1.edf", "V"), (d1 / "sub2.edf", "V"), (d2 / "2_842019_1.edf", "uV"),
                    (d2 / "17_952019_1.edf", "uV")]:
        _write_edf(f, names, unit)
    pd.DataFrame({"ID": ["Sub1", "Sub2"], "Age": [72, 68], "Gender": ["M", "F"], "Group": ["AD", "HC"],
                  "MMSE": [21, 27]}).to_excel(tmp_path / "p1.xlsx", index=False)
    pd.DataFrame({"Id": ["2_842019", "17_952019"], "Age": [71, 79], "Gender": ["M", "F"], "Group": ["HC", "AD"],
                  "MMSE (/30)": [29, 11]}).to_excel(tmp_path / "p2.xlsx", index=False)
    paths = {"dataset1": {"eeg_dir": str(d1), "participants": str(tmp_path / "p1.xlsx")},
             "dataset2": {"eeg_dir": str(d2), "participants": str(tmp_path / "p2.xlsx")}}
    counts = {"build": 0, "fetch": 0, "process": []}
    W = np.random.default_rng(2).standard_normal((68, 30))

    def build(info, cfg, d):
        counts["build"] += 1
        return {"W": W, "roi_names": [f"roi{i:02d}" for i in range(68)], "ch_names": list(info["ch_names"]),
                "n_sources": 20484, "fwd_ch_names": list(info["ch_names"]), "roi_coords": np.zeros((68, 3))}

    monkeypatch.setattr(pl, "fetch_fsaverage", lambda d: counts.__setitem__("fetch", counts["fetch"] + 1))
    monkeypatch.setattr(pl, "build_roi_operator", build)
    monkeypatch.setattr(pl, "verify_operator", lambda *a, **k: 0.0)
    real = pl.process_subject

    def counting(rec, cfg, op):
        counts["process"].append(rec.key)
        return real(rec, cfg, op)

    monkeypatch.setattr(pl, "process_subject", counting)
    return pl, paths, counts, tmp_path


def crash_on(pl, monkeypatch, key):
    real = pl.process_subject

    def crashing(rec, cfg, op):
        if rec.key == key:
            raise SimulatedCrash("disconnect")
        return real(rec, cfg, op)

    monkeypatch.setattr(pl, "process_subject", crashing)


def test_graph_run_resumes_after_interruption(cohort, monkeypatch):
    pl, paths, counts, tmp = cohort
    out, sd = tmp / "out", tmp / "subjects"
    crash_on(pl, monkeypatch, "D2-2")                 # 3rd subject in registry order
    with pytest.raises(SimulatedCrash):
        pl.run(paths, CFG, out, sd)
    assert counts["process"] == ["D1-1", "D1-2"]
    st = pl.status(paths, CFG, out)
    assert st["subjects"].set_index("key")["state"].to_dict() == {
        "D1-1": "completed", "D1-2": "completed", "D2-2": "interrupted", "D2-17": "pending"}
    assert st["assembly"] == "missing" and st["operator_cache"] == "complete"
    assert not (Path(st["run_dir"]) / "graphs.npz").exists()

    monkeypatch.undo()                                # "reconnect": same code, no crash
    from adgraph import storage

    monkeypatch.setattr(storage, "in_colab", lambda: False)   # undo() also removed conftest's tmp-dir allowance
    pl2, *_ = cohort
    counts["process"].clear()
    from adgraph import pipeline as fresh

    calls = []
    real = fresh.process_subject
    monkeypatch.setattr(fresh, "process_subject", lambda r, c, o: (calls.append(r.key), real(r, c, o))[1])
    monkeypatch.setattr(fresh, "fetch_fsaverage", lambda d: pytest.fail("fsaverage must not be fetched again"))
    monkeypatch.setattr(fresh, "build_roi_operator", lambda *a: pytest.fail("operator must be reused"))
    run_dir = fresh.run(paths, CFG, out, sd)
    assert calls == ["D2-2", "D2-17"]                 # completed subjects skipped
    from adgraph.graphs.store import GraphStore

    store = GraphStore.load(run_dir)
    assert set(store.windows["key"]) == {"D1-1", "D1-2", "D2-2", "D2-17"}
    assert store.manifest["graph_run_id"] and store.manifest["source_operator"]["from_cache"]
    st = fresh.status(paths, CFG, out)
    assert st["counts"] == {"completed": 4} and st["assembly"] == "complete"

    # third run: nothing recomputed, assembly untouched
    calls.clear()
    mtime = (run_dir / "graphs.npz").stat().st_mtime_ns
    fresh.run(paths, CFG, out, sd)
    assert calls == [] and (run_dir / "graphs.npz").stat().st_mtime_ns == mtime


def test_crash_while_writing_subject_is_not_complete(cohort, monkeypatch):
    pl, paths, counts, tmp = cohort
    real_marker = pl.write_marker

    def marker(directory, name, ident, files):
        if name == "subject" and ident["key"] == "D1-2":
            raise SimulatedCrash("killed between payload and marker")
        return real_marker(directory, name, ident, files)

    monkeypatch.setattr(pl, "write_marker", marker)
    with pytest.raises(SimulatedCrash):
        pl.run(paths, CFG, tmp / "out", tmp / "sd")
    unit = Path(pl.graph_run_dir(tmp / "out", CFG)) / "subjects" / "D1-2"
    assert (unit / "arrays.npz").exists() and not (unit / "subject.done.json").exists()
    monkeypatch.setattr(pl, "write_marker", real_marker)
    counts["process"].clear()
    pl.run(paths, CFG, tmp / "out", tmp / "sd")
    assert counts["process"] == ["D1-2", "D2-2", "D2-17"]      # partial unit recomputed


def test_corrupted_and_failed_subjects_are_recomputed(cohort, monkeypatch):
    pl, paths, counts, tmp = cohort
    real = pl.process_subject

    def failing(rec, cfg, op):
        if rec.key == "D2-17":
            raise ValueError("bad file")
        return real(rec, cfg, op)

    monkeypatch.setattr(pl, "process_subject", failing)
    with pytest.raises(pl.GraphRunIncomplete, match="D2-17"):     # a failed subject blocks assembly
        pl.run(paths, CFG, tmp / "out", tmp / "sd")
    run_dir = Path(pl.graph_run_dir(tmp / "out", CFG))
    assert pl.status(paths, CFG, tmp / "out")["subjects"].set_index("key").loc["D2-17", "state"] == "failed"
    assert json.loads((run_dir / "subjects" / "D2-17" / "failed.json").read_text())["status"].startswith("error")
    assert not (run_dir / "assembly.done.json").exists() and not (run_dir / "graphs.npz").exists()
    monkeypatch.setattr(pl, "process_subject", real)
    counts["process"].clear()
    p = run_dir / "subjects" / "D1-1" / "arrays.npz"
    p.write_bytes(p.read_bytes()[:-10])                   # truncated -> checksum fails
    calls = []
    monkeypatch.setattr(pl, "process_subject", lambda r, c, o: (calls.append(r.key), real(r, c, o))[1])
    pl.run(paths, CFG, tmp / "out", tmp / "sd")
    assert sorted(calls) == ["D1-1", "D2-17"]             # corrupted + previously failed; others skipped
    assert (run_dir / "_superseded").exists()            # old (corrupted) unit kept, not overwritten
    assert pd.read_csv(run_dir / "subjects.csv")["status"].eq("ok").all()


def test_config_changes_and_force(cohort):
    pl, paths, counts, tmp = cohort
    out = tmp / "out"
    a = pl.run(paths, CFG, out, tmp / "sd")
    assert counts["build"] == 1
    cfg_psd = set_dotted(CFG, "psd.welch.segment_s", 1.0)            # graph config changed, source model not
    b = pl.run(paths, cfg_psd, out, tmp / "sd")
    assert b != a and counts["build"] == 1                           # new run dir, operator reused
    cfg_src = set_dotted(CFG, "source.loose", 0.3)                   # source model changed
    pl.run(paths, cfg_src, out, tmp / "sd")
    assert counts["build"] == 2
    n = len(counts["process"])
    c = pl.run(paths, CFG, out, tmp / "sd", force=True)
    assert c.name.startswith(a.name + "_force") and counts["build"] == 3 and len(counts["process"]) == n + 4
    assert (a / "graphs.npz").exists()                               # original untouched
    with pytest.raises(FileExistsError):
        pl.run(paths, CFG, out, tmp / "sd", resume=False)


# ======================================================================= experiments
@pytest.fixture(scope="module")
def small_store():
    return make_synthetic_store({"D1": (5, 5), "D2": (5, 6)}, windows_per_subject=(3, 4), psd_effect=0.5, seed=1)


def lr_variant(n_repeats=2):
    v = [x for x in expand_variants(load_experiment(EXP / "exp1_source_psd.yaml")) if x["model"]["name"] == "logreg"][0]
    for k, val in {"evaluation.n_repeats": n_repeats, "evaluation.k": 3, "evaluation.inner_k": 2,
                   "evaluation.n_bootstrap": 50}.items():
        v = set_dotted(v, k, val)
    v["hparams"] = {"model.params.C": [0.1, 1.0]}
    return v


@pytest.fixture
def fit_counter(monkeypatch):
    """Counts model fits; optionally raises SimulatedCrash at the n-th fit."""
    state = {"fits": 0, "crash_at": None}
    real = runner.make_model

    def make(v, seed):
        m = real(v, seed)
        fit = m.fit

        def counted(view):
            state["fits"] += 1
            if state["crash_at"] is not None and state["fits"] == state["crash_at"]:
                raise SimulatedCrash("disconnect during training")
            return fit(view)

        m.fit = counted
        return m

    monkeypatch.setattr(runner, "make_model", make)
    return state


def test_experiment_resumes_from_unfinished_folds(small_store, tmp_path, fit_counter):
    v = lr_variant()
    full = runner.run_variant(small_store, v, "within_D1", tmp_path / "ref")    # uninterrupted reference
    total = fit_counter["fits"]               # 2 repeats x 3 folds x (2 C x 2 inner + 1 outer) = 30
    assert total == 30
    fit_counter.update(fits=0, crash_at=13)
    with pytest.raises(SimulatedCrash):
        runner.run_variant(small_store, v, "within_D1", tmp_path / "run")
    st = runner.experiment_status(small_store, v, "within_D1", tmp_path / "run")
    # classical inner trials are not cached (a fit is cheaper than a Drive write); outer folds are
    assert st["state"] == "partial" and st["folds_completed"] == 2 and st["inner_trials_cached"] == 0
    fit_counter.update(fits=0, crash_at=None)
    resumed = runner.run_variant(small_store, v, "within_D1", tmp_path / "run")
    assert fit_counter["fits"] == total - 10                       # the 2 completed folds (5 fits each) reused
    a = pd.read_csv(tmp_path / "ref/exp1_source_psd/logreg/within_D1/predictions_windows.csv")
    b = pd.read_csv(tmp_path / "run/exp1_source_psd/logreg/within_D1/predictions_windows.csv")
    assert np.allclose(a["score"], b["score"])
    assert json.dumps(resumed["subject_level"], sort_keys=True) == json.dumps(full["subject_level"], sort_keys=True)
    fit_counter["fits"] = 0
    runner.run_variant(small_store, v, "within_D1", tmp_path / "run")
    assert fit_counter["fits"] == 0                                  # complete: nothing retrained
    assert runner.experiment_status(small_store, v, "within_D1", tmp_path / "run")["state"] == "completed"


def test_more_repeats_reuse_existing_folds(small_store, tmp_path, fit_counter):
    runner.run_variant(small_store, lr_variant(2), "within_D2", tmp_path)
    fit_counter["fits"] = 0
    s3 = runner.run_variant(small_store, lr_variant(3), "within_D2", tmp_path)
    assert fit_counter["fits"] == 15                                 # only the new repeat (3 folds x 5 fits)
    assert s3["n_repeats"] == 3 and (tmp_path / "_superseded").exists()


def test_incompatible_results_are_never_reused(small_store, tmp_path, fit_counter):
    v = lr_variant()
    runner.run_variant(small_store, v, "within_D1", tmp_path)
    changed = set_dotted(v, "hparams", {"model.params.C": [0.01, 10.0]})
    with pytest.raises(IncompatibleCheckpoint, match="different configuration"):
        runner.run_variant(small_store, changed, "within_D1", tmp_path)
    other_graphs = make_synthetic_store({"D1": (5, 5), "D2": (5, 6)}, windows_per_subject=(3, 4), seed=7)
    with pytest.raises(IncompatibleCheckpoint, match="graph_run_id"):
        runner.run_variant(other_graphs, v, "within_D1", tmp_path)
    assert runner.experiment_status(other_graphs, v, "within_D1", tmp_path)["state"] == "incompatible"
    fit_counter["fits"] = 0
    runner.run_variant(small_store, changed, "within_D1", tmp_path, force=True)
    assert fit_counter["fits"] == 30 and list((tmp_path / "_superseded").rglob("final.done.json"))
    from adgraph.visualization.results import index_results

    assert len(index_results(tmp_path)) == 1                         # superseded results never indexed


def test_corrupted_fold_unit_is_recomputed_alone(small_store, tmp_path, fit_counter):
    v = lr_variant()
    runner.run_variant(small_store, v, "within_D1", tmp_path)
    out = tmp_path / "exp1_source_psd/logreg/within_D1"
    p = out / "units/fold_r001_f02.pred.csv"
    p.write_text(p.read_text().replace("0.", "9.", 1))
    (out / "final.done.json").unlink()                               # e.g. crash before the final marker
    fit_counter["fits"] = 0
    runner.run_variant(small_store, v, "within_D1", tmp_path)
    assert fit_counter["fits"] == 5                                  # that fold only (classical: inner search redone)


def test_permutations_resume(small_store, tmp_path, fit_counter):
    v = set_dotted(lr_variant(1), "evaluation.permutation", {"n": 3, "n_repeats": 1})
    v["hparams"] = {}
    fit_counter.update(fits=0, crash_at=3 + 3 + 2)                   # 3 outer fits, perm 0, crash in perm 1
    with pytest.raises(SimulatedCrash):
        runner.run_variant(small_store, v, "pooled", tmp_path)
    fit_counter.update(fits=0, crash_at=None)
    s = runner.run_variant(small_store, v, "pooled", tmp_path)
    assert fit_counter["fits"] == 6 and s["permutation"]["n"] == 3   # perms 1 and 2 only


def test_gnn_inner_trials_and_folds_resume(small_store, tmp_path, fit_counter):
    pytest.importorskip("torch")
    v = expand_variants(load_experiment(EXP / "exp6_multiband_psd_strength.yaml"))[0]
    for k, val in {"evaluation.n_repeats": 1, "evaluation.k": 3, "evaluation.inner_k": 2, "training.epochs": 1,
                   "model.hidden": 8, "evaluation.save_embeddings": True}.items():
        v = set_dotted(v, k, val)
    v["hparams"] = {"model.hidden": [8, 16]}
    fit_counter.update(fits=0, crash_at=8)
    with pytest.raises(SimulatedCrash):
        runner.run_variant(small_store, v, "within_D2", tmp_path)
    fit_counter.update(fits=0, crash_at=None)
    runner.run_variant(small_store, v, "within_D2", tmp_path)
    assert fit_counter["fits"] == 15 - 7
    z = np.load(tmp_path / v["experiment"] / v["variant"] / "within_D2" / "embeddings.npz")
    assert {f"r000_f{f:02d}__emb" for f in range(3)} <= set(z.files)   # reused + new folds, one model each


# ======================================================================= figures never train
def test_figures_never_trigger_training(small_store, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from adgraph.visualization.figures import make_figures

    v = lr_variant(1)
    for s in ("within_D1", "within_D2"):
        runner.run_variant(small_store, v, s, tmp_path / "res")
    for name in ("make_model", "cross_validate", "run_variant", "select_hparams"):
        monkeypatch.setattr(runner, name, lambda *a, **k: pytest.fail("figures must not train"))
    cfg = {"figures": [{"id": "perf", "family": "performance", "settings": ["within_D1", "within_D2"],
                        "metrics": ["roc_auc"], "entries": [{"label": "LR", "experiment": "exp1_source_psd",
                                                             "where": {"model.name": "logreg"}}]}]}
    w = make_figures(cfg, tmp_path / "figs", tmp_path / "res", synthetic=True, on_missing="error")
    assert [f["id"] for f in w.figures] == ["perf"]


# ======================================================================= CLI
def test_cli_resume_and_status(tmp_path):
    base = [sys.executable, "scripts/run_experiment.py", "--config", str(EXP / "exp0_sensor_psd.yaml"),
            "--synthetic", "--out", str(tmp_path), "--settings", "within_D1", "--variants", "logreg",
            "--override", "evaluation.n_repeats=1", "--override", "evaluation.k=3", "--override", "hparams={}"]
    st = subprocess.run(base + ["--status"], capture_output=True, text=True, check=True).stdout
    assert "not started" in st
    subprocess.run(base, capture_output=True, text=True, check=True)
    st = subprocess.run(base + ["--status"], capture_output=True, text=True, check=True).stdout
    assert "completed" in st
    again = subprocess.run(base, capture_output=True, text=True, check=True)
    assert "nothing recomputed" in again.stderr
    bad = subprocess.run(base[:-1] + ["hparams={model.params.C: [5.0]}"], capture_output=True, text=True)
    assert bad.returncode != 0 and "IncompatibleCheckpoint" in bad.stderr


def test_graph_run_discovery(cohort):
    pl, paths, counts, tmp = cohort
    out = tmp / "out"
    with pytest.raises(FileNotFoundError):
        pl.latest_complete_graph_run(out)
    pl.run(paths, CFG, out, tmp / "sd", limit=1)                      # smoke test is never selected
    with pytest.raises(FileNotFoundError):
        pl.latest_complete_graph_run(out)
    full = pl.run(paths, CFG, out, tmp / "sd")
    runs = pl.list_graph_runs(out)
    assert runs["assembled"].all() and runs["limit_smoke_test"].sum() == 1
    assert pl.latest_complete_graph_run(out) == full
