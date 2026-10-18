"""A graph run without a complete, non-empty cohort fails explicitly and is never used (log entry 38)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adgraph.config import load_yaml
from adgraph.experiments import runner
from adgraph.experiments.config import expand_variants, load_experiment
from adgraph.graphs.store import IncompleteGraphRun, cohort_problems, require_complete_cohort
from adgraph.legacy.synthetic import make_synthetic_store
from test_checkpoint import cohort  # noqa: F401 (fixture)

CFG = load_yaml("configs/pipeline.yaml")


def _amplitude_failure(pl, monkeypatch, keys=None):
    def failing(rec, cfg, op):
        if keys is None or rec.key in keys:
            raise ValueError("median channel std 0.000221 V outside plausible range [1e-07, 0.0002]")
        return real(rec, cfg, op)

    real = pl.process_subject
    monkeypatch.setattr(pl, "process_subject", failing)


def test_all_subjects_failing_is_an_explicit_failure(cohort, monkeypatch):  # noqa: F811
    pl, paths, _, tmp = cohort
    _amplitude_failure(pl, monkeypatch)
    with pytest.raises(pl.GraphRunIncomplete, match="4 of 4 subject"):
        pl.run(paths, CFG, tmp / "out", tmp / "sd", limit=2)
    run_dir = Path(pl.graph_run_dir(tmp / "out", CFG, 2))
    assert not (run_dir / "assembly.done.json").exists()            # no completion marker
    assert not (run_dir / "graphs.npz").exists() and not (run_dir / "manifest.json").exists()
    last = json.loads((run_dir / "last_attempt.json").read_text())   # diagnostics kept
    assert last["result"] == "incomplete" and sorted(last["failed"]) == ["D1-1", "D1-2", "D2-17", "D2-2"]
    assert "plausible range" in last["failed"]["D1-1"]
    for key in last["failed"]:
        assert json.loads((run_dir / "subjects" / key / "failed.json").read_text())["status"].startswith("error")
    assert set(pd.read_csv(run_dir / "status.csv")["state"]) == {"failed"}
    assert (pd.read_csv(run_dir / "memory_log.csv")["event"] == "failed").sum() == 4
    assert pl.list_graph_runs(tmp / "out")["assembled"].eq(False).all()


def test_no_subject_with_graphs_is_an_explicit_failure(cohort):  # noqa: F811
    pl, paths, _, tmp = cohort
    cfg = json.loads(json.dumps(CFG))
    cfg["windowing"]["min_windows"] = 1000                        # every subject below_min_windows
    with pytest.raises(pl.GraphRunIncomplete, match="no subject produced graphs"):
        pl.run(paths, cfg, tmp / "out", tmp / "sd")
    run_dir = Path(pl.graph_run_dir(tmp / "out", cfg))
    assert not (run_dir / "assembly.done.json").exists()
    assert set(json.loads((run_dir / "last_attempt.json").read_text())["statuses"].values()) == {"below_min_windows"}


def test_failed_rerun_retires_an_earlier_assembly(cohort, monkeypatch):  # noqa: F811
    pl, paths, _, tmp = cohort
    run_dir = pl.run(paths, CFG, tmp / "out", tmp / "sd")
    assert (run_dir / "assembly.done.json").exists()
    p = run_dir / "subjects" / "D1-1" / "arrays.npz"
    p.write_bytes(p.read_bytes()[:-10])                            # forces D1-1 to be recomputed
    _amplitude_failure(pl, monkeypatch, {"D1-1"})
    with pytest.raises(pl.GraphRunIncomplete, match="D1-1"):
        pl.run(paths, CFG, tmp / "out", tmp / "sd")
    assert not (run_dir / "assembly.done.json").exists()            # moved, not deleted
    assert list((run_dir / "_superseded").rglob("assembly.done.json"))
    with pytest.raises(FileNotFoundError):
        pl.latest_complete_graph_run(tmp / "out")


def test_assemble_refuses_failed_or_empty_cohorts(tmp_path):
    import adgraph.legacy.pipeline as pl

    with pytest.raises(pl.GraphRunIncomplete, match="failed"):
        pl.assemble(tmp_path, pd.DataFrame({"key": ["D1-1"]}), None, CFG, {}, [], [], None, {"D1-1": "error: x"})


def test_cohort_problems():
    store = make_synthetic_store()
    assert cohort_problems(store) == [] and require_complete_cohort(store) is store
    empty = store.subset(np.array([], dtype=int))
    assert any("no graphs" in p for p in cohort_problems(empty))
    failed = store.subset(np.arange(store.n_windows))
    failed.subjects = store.subjects.assign(status=["error: x"] + ["ok"] * (len(store.subjects) - 1))
    assert any("failed" in p for p in cohort_problems(failed))
    one_ds = store.subset(np.flatnonzero(store.windows["dataset"] == "D1"))
    assert any("no graphs for dataset(s) ['D2']" in p for p in cohort_problems(one_ds))
    smoke = store.subset(np.arange(store.n_windows))
    smoke.manifest = {**store.manifest, "registry": {"subset_limit": 1}}
    assert any("smoke test" in p for p in cohort_problems(smoke)) and cohort_problems(smoke, allow_subset=True) == []
    units = store.subset(np.arange(store.n_windows))
    units.manifest = {**store.manifest, "subject_units": {"D1-1": {"failed": "error: x"}}}
    assert any("failed subject units" in p for p in cohort_problems(units))




def test_latest_run_refuses_a_legacy_empty_assembly(tmp_path):
    """An empty cohort assembled (with a marker) by earlier code is never selected for experiments."""
    import adgraph.legacy.pipeline as pl

    run_dir = tmp_path / "graphs_v3_abc"
    store = make_synthetic_store().subset(np.array([], dtype=int))
    store.manifest = {**store.manifest, "synthetic": False, "assembled_at": "2026-10-13T00:00:00"}
    store.save(run_dir)
    (run_dir / "assembly.done.json").write_text("{}")
    with pytest.raises(IncompleteGraphRun, match="no graphs"):
        pl.latest_complete_graph_run(tmp_path)
