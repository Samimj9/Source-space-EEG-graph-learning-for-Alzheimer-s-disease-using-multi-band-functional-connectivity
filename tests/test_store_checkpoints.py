"""Diagnostics that read per-participant checkpoints (source.resolution: operator ranks, reconstruction check, store
resolution) on both store layouts: a store with its own ``subjects/`` (wMNE, a new build) and a store without
(the corrected LCMV ``_corr1`` store, a mode A store), whose checkpoints are those of the original mode B build and
are used only after their provenance is verified. Synthetic files only; nothing is written into any store."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from adgraph.checkpoint import write_marker
from adgraph.config import config_hash
from adgraph.source.resolution import EmptyDiagnostic, operator_ranks, reconstruction_check, store_resolution
from adgraph.sourcegraphs.build import run_dirs
from adgraph.sourcegraphs.checkpoints import CheckpointProvenanceError, locate
from adgraph.sourcegraphs.config import resolve

KEYS = ["D3-1", "D3-2", "D3-3"]
PIECE = 6000                                                                   # saved samples per piece


def _avg_ref(n):
    return np.eye(n) - 1.0 / n


def _layout(tmp_path, seed=0):
    """<processed>/D3_LCMV_XU: the original mode B run (with checkpoints), its corrected copy ``_corr1`` (no
    ``subjects/``; one window of D3-2 removed) and the original mode A run (no ``subjects/``)."""
    rng = np.random.default_rng(seed)
    cfg = resolve("lcmv")
    b, a = run_dirs(tmp_path, cfg)
    b.mkdir(parents=True)
    starts = 1000 + np.arange(0, PIECE - 2500 + 1, 250)
    inputs, subj, win = [], [], []
    for i, k in enumerate(KEYS):
        unit = b / "subjects" / k
        unit.mkdir(parents=True)
        c = _avg_ref(19)
        W = rng.standard_normal((68, 19)) @ c
        roi = (W @ (c @ rng.standard_normal((19, PIECE)))).astype(np.float32)
        np.savez(unit / "operator.npz", W=W)
        np.savez(unit / "roi_timeseries.npz", piece0=roi, piece0_start_sample=np.array(1000))
        w = pd.DataFrame({"window_id": [f"{k}-w{j:04d}" for j in range(len(starts))], "key": k, "piece": 0,
                          "start_sample": starts})
        w.to_csv(unit / "windows.csv", index=False)
        win.append(w)
        sha, geom = f"sha-{k}", "geomA"
        write_marker(unit, "subject", {"kind": "lx_subject", "key": k, "config_hash": config_hash(cfg),
                                       "input": {"sha256": sha, "size_bytes": 1}, "geometry": geom},
                     ["operator.npz", "roi_timeseries.npz", "windows.csv"])
        inputs.append({"key": k, "path": f"{k}.set", "size_bytes": 1, "sha256": sha})
        subj.append({"key": k, "status": "ok", "y": i % 2, "montage_hash": geom})
    man = {"config": cfg, "graph_run_id": "orig-B", "inputs": inputs, "source_operator": {"method": "lcmv"}}
    (b / "manifest.json").write_text(json.dumps(man))
    pd.DataFrame(subj).to_csv(b / "subjects.csv", index=False)
    pd.concat(win).to_csv(b / "windows.csv", index=False)
    corr = b.with_name(b.name + "_corr1")
    corr.mkdir()
    cman = {**man, "graph_run_id": "corr-B",
            "correction": {"name": "corr1", "participants_changed": ["D3-2"],
                           "source": {"mode_B": {"run_dir": f"/content/drive/MyDrive/x/{b.name}",
                                                 "graph_run_id": "orig-B"}}}}
    (corr / "manifest.json").write_text(json.dumps(cman))
    pd.DataFrame(subj).to_csv(corr / "subjects.csv", index=False)
    allw = pd.concat(win)
    allw[allw["window_id"] != "D3-2-w0005"].to_csv(corr / "windows.csv", index=False)   # the correction's subset
    a.mkdir()
    (a / "manifest.json").write_text(json.dumps({**man, "graph_run_id": "orig-A"}))
    pd.DataFrame(subj).to_csv(a / "subjects.csv", index=False)
    allw.groupby("key").head(10).to_csv(a / "windows.csv", index=False)
    return b, corr, a


def _snapshot(d):
    return sorted((str(p.relative_to(d)), p.stat().st_size) for p in d.rglob("*"))


def test_own_checkpoints_are_used_directly(tmp_path):
    b, _, _ = _layout(tmp_path)
    prov = locate(b)
    assert prov["relation"] == "own checkpoints" and prov["participants"] == KEYS and not prov["covariance_caveat"]
    t, _ = reconstruction_check(b, margin=825, window=2500)
    assert (t["rank"] == 18).all() and (t["rel_residual"] < 1e-6).all()
    starts = 1000 + np.arange(0, PIECE - 2500 + 1, 250) - 1000
    inside = int(((starts >= 825) & (starts + 2500 + 825 <= PIECE)).sum())
    assert (t["windows_reconstructable"] == inside).all() and (t["n_windows"] > inside).all()   # edge windows


def test_corrected_store_reads_the_verified_original_checkpoints_and_its_own_windows(tmp_path):
    b, corr, _ = _layout(tmp_path)
    before = _snapshot(corr)
    prov = locate(corr)
    assert prov["checkpoint_dir"] == str(b) and "corrected" in prov["relation"] and prov["n_verified"] == 3
    assert set(prov["covariance_caveat"]) == {"D3-2"}                          # LCMV covariance caveat (entry 53)
    t, _ = reconstruction_check(corr)
    n = t.set_index("key")["n_windows"]
    assert n["D3-2"] == n["D3-1"] - 1 == pd.read_csv(corr / "windows.csv").query("key == 'D3-2'").shape[0]
    assert t.set_index("key").loc["D3-2", "covariance_caveat"] and not t.set_index("key").loc["D3-1", "covariance_caveat"]
    r, _ = operator_ranks(corr)
    assert (r["rank"] == 18).all() and len(r) == 3
    assert _snapshot(corr) == before and not (corr / "subjects").exists()      # nothing copied into _corr1


def test_mode_a_store_reads_the_mode_b_checkpoints(tmp_path):
    b, _, a = _layout(tmp_path)
    prov = locate(a)
    assert prov["checkpoint_dir"] == str(b) and prov["relation"] == "mode B run of the same build"
    t, _ = reconstruction_check(a)
    assert (t["n_windows"] == 10).all()


@pytest.mark.parametrize("damage", ["graph_run_id", "input", "geometry", "operator", "windows", "config"])
def test_unverifiable_checkpoints_are_refused(tmp_path, damage):
    b, corr, _ = _layout(tmp_path)
    if damage == "graph_run_id":
        m = json.loads((b / "manifest.json").read_text())
        m["graph_run_id"] = "rebuilt"
        (b / "manifest.json").write_text(json.dumps(m))
    elif damage == "input":
        m = json.loads((corr / "manifest.json").read_text())
        m["inputs"][1]["sha256"] = "another-recording"
        (corr / "manifest.json").write_text(json.dumps(m))
    elif damage == "geometry":
        s = pd.read_csv(corr / "subjects.csv")
        s.loc[0, "montage_hash"] = "geomB"
        s.to_csv(corr / "subjects.csv", index=False)
    elif damage == "operator":
        np.savez(b / "subjects" / "D3-3" / "operator.npz", W=np.ones((68, 19)))
    elif damage == "windows":
        w = pd.read_csv(corr / "windows.csv")
        w.loc[0, "window_id"] = "D3-1-w9999"
        w.to_csv(corr / "windows.csv", index=False)
    else:
        m = json.loads((corr / "manifest.json").read_text())
        m["config"]["windowing"] = {**m["config"]["windowing"], "step_s": 2.0}
        (corr / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(CheckpointProvenanceError):
        locate(corr)


def test_missing_checkpoints_are_reported_and_an_empty_result_is_an_error(tmp_path):
    b, corr, _ = _layout(tmp_path)
    shutil.rmtree(b / "subjects" / "D3-1")
    prov = locate(corr)
    assert prov["missing"] == ["D3-1"] and prov["participants"] == ["D3-2", "D3-3"]
    t, prov = reconstruction_check(corr)
    assert len(t) == 2 and prov["missing"] == ["D3-1"]
    with pytest.raises(EmptyDiagnostic, match="0 participants"):
        operator_ranks(corr, keys=["D3-1"])
    with pytest.raises(EmptyDiagnostic):
        store_resolution(corr, tmp_path / "no-fsaverage", "dk68", keys=["D3-1"])
    shutil.rmtree(b / "subjects")
    with pytest.raises(CheckpointProvenanceError, match="no participant checkpoints"):
        locate(corr)
