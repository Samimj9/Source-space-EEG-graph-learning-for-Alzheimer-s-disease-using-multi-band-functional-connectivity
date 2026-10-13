"""Memory behaviour of graph construction (synthetic recordings only; no real EEG).

* bounded-memory code paths give results identical to the unbounded ones;
* per-recording peak is bounded relative to the recording size;
* nothing accumulates from one subject to the next;
* cohort assembly holds the cohort arrays once (no list + concatenate duplicate);
* an OOM-style kill (SIGKILL) while a subject is written leaves completed subjects intact,
  and the rerun recomputes only the interrupted subject;
* memory QC is recorded (subjects.csv, memory_log.csv).
"""

import json
import subprocess
import sys
import textwrap
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adgraph import memory
from adgraph.checkpoint import file_sha256
from adgraph.config import common_channels
from adgraph.connectivity.plv import band_analytic
from test_checkpoint import CFG, cohort  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]


def _recording(tmp_path, seconds=None):
    from adgraph.io.eeg import load_raw
    from adgraph.preprocessing.clean import preprocess
    from test_end_to_end import _write_edf

    path = tmp_path / "rec.edf"
    _write_edf(path, CFG["channels"]["recorded"] + ["ECG"], "uV")
    raw, _ = load_raw(path, "D2", CFG)
    raw, _ = preprocess(raw, CFG)
    return path, raw


# ------------------------------------------------------------------ identical results
def test_chunked_band_analytic_is_identical():
    x = np.random.default_rng(0).standard_normal((68, 20000))
    full = band_analytic(x, 250.0, 8.0, 10.0)
    for chunk in (1, 8, 30):
        assert np.allclose(band_analytic(x, 250.0, 8.0, 10.0, chunk_rows=chunk), full, rtol=0, atol=1e-12)


def test_bounded_process_recording_is_identical(tmp_path, monkeypatch):
    import adgraph.pipeline as pl

    _, raw = _recording(tmp_path)
    W = np.random.default_rng(1).standard_normal((68, len(common_channels(CFG))))
    bounded = pl.process_recording(raw, W, CFG)
    monkeypatch.setattr(pl, "ANALYTIC_CHUNK_ROWS", None)                  # previous, unchunked path
    full = pl.process_recording(raw, W, CFG)
    for k in ("plv", "ciplv", "ciplv_signed", "psd_abs", "psd_log", "psd_rel", "sensor_psd_rel"):
        assert np.allclose(bounded[k], full[k], rtol=0, atol=1e-6), k
    assert bounded["roi_timeseries"] is None                               # not saved -> not copied
    cfg = {**CFG, "output": {**CFG["output"], "save_roi_timeseries": True}}
    assert pl.process_recording(raw, W, cfg)["roi_timeseries"].dtype == np.float32


def test_lazy_edf_loading_reads_identical_samples(tmp_path):
    import mne

    path, _ = _recording(tmp_path)
    from adgraph.io.eeg import load_raw

    raw, _ = load_raw(path, "D2", CFG)
    ref = mne.io.read_raw_edf(path, preload=True, verbose="error")
    names = [n for n in ref.ch_names if n.upper() in {c.upper() for c in raw.ch_names}]
    ref_data = ref.get_data(picks=names)
    order = [names.index(next(n for n in names if n.upper() == c.upper())) for c in raw.ch_names]
    assert np.array_equal(raw.get_data(), ref_data[order] * float(CFG["units"]["scale"]["D2"]))


# ------------------------------------------------------------------ bounded peak, no accumulation
def test_recording_peak_memory_is_bounded(tmp_path):
    import adgraph.pipeline as pl

    _, raw = _recording(tmp_path)
    W = np.random.default_rng(1).standard_normal((68, len(common_channels(CFG))))
    roi_bytes = 68 * raw.n_times * 8                                       # continuous ROI signals, float64
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        pl.process_recording(raw, W, CFG)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # ROI signals + one band's analytic signal (2x) + chunk temporaries; the unbounded path
    # held two analytic signals plus whole-array filter/FFT temporaries (about 9x).
    assert peak < 4.5 * roi_bytes, f"peak {peak / 1e6:.0f} MB vs ROI signals {roi_bytes / 1e6:.0f} MB"


def test_no_memory_accumulates_across_subjects(cohort, monkeypatch):
    pl, paths, counts, tmp = cohort
    retained = []
    real_save = pl.save_subject

    def snapshot(*a, **k):
        real_save(*a, **k)
        import gc
        gc.collect()
        retained.append(tracemalloc.get_traced_memory()[0])

    monkeypatch.setattr(pl, "save_subject", snapshot)
    tracemalloc.start()
    try:
        run_dir = pl.run(paths, CFG, tmp / "out", tmp / "sd")
    finally:
        tracemalloc.stop()
    assert len(retained) == 4
    growth = retained[-1] - retained[0]
    assert growth < 8e6, f"memory retained grew by {growth / 1e6:.1f} MB over 3 subjects: {retained}"
    subj = pd.read_csv(run_dir / "subjects.csv")
    assert {"mem_peak_mb", "mem_rss_before_mb", "mem_available_before_mb", "mem_total_mb"} <= set(subj.columns)
    log = pd.read_csv(run_dir / "memory_log.csv")
    assert log["unit"].tolist() == ["operator", "D1-1", "D1-2", "D2-2", "D2-17", "assembly"]
    assert (log.loc[log["unit"].str.startswith("D"), "mem_peak_mb"] > 0).all()


def test_memory_tracker_scope():
    with memory.MemoryTracker() as m:
        block = np.ones(int(2e7))                                          # 160 MB
        block[0] = 2
    rec = m.record()
    del block
    if rec["mem_peak_scope"] == "unit":
        assert rec["mem_peak_mb"] >= rec["mem_rss_before_mb"] + 140
    assert rec["mem_total_mb"] is None or rec["mem_total_mb"] > 0


# ------------------------------------------------------------------ assembly holds the cohort once
def test_assembly_holds_cohort_arrays_once(tmp_path):
    import adgraph.pipeline as pl
    from adgraph.synthetic import make_synthetic_store

    store = make_synthetic_store({"D1": (3, 3), "D2": (3, 3)}, windows_per_subject=(60, 60))
    keys = [k for k in pl.DATA_ARRAY_KEYS if k in store.arrays]
    total = sum(store.arrays[k].nbytes for k in keys)
    run_dir = tmp_path / "run"
    reg = store.subjects[["key", "dataset", "subject_id", "label", "y"]].copy()
    reg["path"], reg["size_bytes"] = "synthetic", 0
    for key, idx in store.windows.groupby("key", sort=False).indices.items():
        row = {**store.subjects.set_index("key").loc[key].to_dict(), "key": key, "status": "ok"}
        pl.save_subject(run_dir / "subjects" / key, {"key": key}, {k: store.arrays[k][idx] for k in keys},
                        store.windows.iloc[idx], json.loads(json.dumps(row, default=str)), None)
    op = {"operator_id": "x", "roi_names": store.roi_names, "ch_names": store.channels, "W": np.zeros((68, 30)),
          "n_sources": 1, "verification_rel_error": 0.0, "roi_coords": None}
    order = list(store.windows.groupby("key", sort=False).indices)
    reg = reg.set_index("key").loc[order].reset_index()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        pl.assemble(run_dir, reg, reg, CFG, op, [], [], None, {})
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1.5 * total + 30e6, f"assembly peak {peak / 1e6:.0f} MB for {total / 1e6:.0f} MB of arrays"
    from adgraph.graphs.store import GraphStore

    out = GraphStore.load(run_dir)
    for k in keys:
        assert np.array_equal(out.arrays[k], store.arrays[k][np.concatenate(
            [store.windows.groupby("key", sort=False).indices[x] for x in order])]), k


# ------------------------------------------------------------------ OOM-style kill
KILL_SCRIPT = textwrap.dedent('''
    import json, os, signal, sys
    from pathlib import Path
    import numpy as np, yaml
    import adgraph.pipeline as pl
    paths, out, sd, cache = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    cfg = yaml.safe_load(open("configs/pipeline.yaml"))
    W = np.random.default_rng(2).standard_normal((68, 30))
    def build(info, cfg, d):
        return {"W": W, "roi_names": [f"roi{i:02d}" for i in range(68)], "ch_names": list(info["ch_names"]),
                "n_sources": 20484, "fwd_ch_names": list(info["ch_names"]), "roi_coords": np.zeros((68, 3))}
    pl.fetch_fsaverage = lambda d: None
    pl.build_roi_operator = build
    pl.verify_operator = lambda *a, **k: 0.0
    real = pl.atomic_savez
    def killing(path, **arrays):
        if "D2-2" in str(path):                   # OOM kill while this subject's arrays are written
            tmp = Path(path).with_name(f".{Path(path).name}.tmp-killed")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(b"partial")
            os.kill(os.getpid(), signal.SIGKILL)
        return real(path, **arrays)
    pl.atomic_savez = killing
    pl.run(paths, cfg, out, sd, operator_cache=cache)
''')


def test_sigkill_during_subject_write_keeps_completed_subjects(cohort):
    pl, paths, counts, tmp = cohort
    out, sd, cache = tmp / "out", tmp / "sd", tmp / "ops"
    r = subprocess.run([sys.executable, "-c", KILL_SCRIPT, json.dumps(paths), str(out), str(sd), str(cache)],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == -9, r.stderr[-2000:]                           # killed, like an OOM kill
    run_dir = pl.graph_run_dir(out, CFG)
    done = {k: {p.name: file_sha256(p) for p in (run_dir / "subjects" / k).iterdir() if p.is_file()}
            for k in ("D1-1", "D1-2")}
    assert (run_dir / "subjects" / "D2-2" / ".arrays.npz.tmp-killed").exists()
    assert not (run_dir / "subjects" / "D2-2" / "subject.done.json").exists()
    st = pl.status(paths, CFG, out, operator_cache=cache)["subjects"].set_index("key")["state"].to_dict()
    assert st == {"D1-1": "completed", "D1-2": "completed", "D2-2": "interrupted", "D2-17": "pending"}

    counts["process"].clear()
    pl.run(paths, CFG, out, sd, operator_cache=cache)                     # restart in a new session
    assert counts["process"] == ["D2-2", "D2-17"]
    for k, files in done.items():                                        # completed units untouched
        assert {p.name: file_sha256(p) for p in (run_dir / "subjects" / k).iterdir() if p.is_file()} == files
    assert not list(run_dir.rglob(".*.tmp-*"))                            # leftovers of the killed write removed
    from adgraph.graphs.store import GraphStore

    assert set(GraphStore.load(run_dir).windows["key"]) == {"D1-1", "D1-2", "D2-2", "D2-17"}
