"""Correspondence check between a matched wMNE store and the corrected LCMV store of the same mode (entry 55).

Read-only. The matched wMNE stores (``graphsWX*``) are built by the same ``lcmvxu`` code as the LCMV stores, from
the same recordings and configuration except the inverse solution. ``match`` requires, against the reference:

* configuration: identical except ``output_subdir``, ``store_names`` and ``source``;
* participants: the same keys, labels, status, five-minute selection (duration, pieces, exact segment table and
  boundary cuts) and window counts;
* windows: the same rows in the same order (window id, participant, start / stop sample, segment, piece, tenfold
  flag) and the same subject weights;
* frozen split (``xu_split.json``), bands, band edges, ROI names (order) and channels: identical;
* sensor-space features (``sensor_psd_*``): equal to float32 precision. They do not involve the inverse, so this is
  the consistency control of Exp 0 (identical inputs give identical Exp 0 results).

Source-space arrays are expected to differ (that is the comparison); their summary differences are reported only.
The result is written to ``match.json`` in the target store; experiments of the WX / WX10 families refuse a store
without a passed ``match.json`` for its current graph_run_id and the current reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED_CONFIG_DIFF = {"output_subdir", "store_names", "source"}
WINDOW_COLS = ["window_id", "key", "y", "window_index", "start_sample", "stop_sample", "segment_index", "piece",
               "xu_tenfold"]
SUBJECT_COLS = ["key", "y", "status", "selection_mode", "n_pieces", "selected_duration_s", "segments_samples",
                "segments_in", "boundary_cuts_in", "n_windows_candidate", "n_windows_accepted", "n_windows_kept",
                "n_tenfold"]
META_ARRAYS = ("bands", "band_edges", "roi_names", "channels")
SENSOR = ("sensor_psd_abs", "sensor_psd_log", "sensor_psd_rel", "sensor_psd_de")
SOURCE = ("psd_log", "psd_rel", "psd_de", "plv", "ciplv")


def match(target: str | Path, reference: str | Path, sensor_rtol: float = 1e-5) -> dict:
    target, reference = Path(target), Path(reference)
    checks = {}
    mt, mr = (json.loads((d / "manifest.json").read_text()) for d in (target, reference))
    diff = sorted(k for k in set(mt["config"]) | set(mr["config"]) if mt["config"].get(k) != mr["config"].get(k))
    checks["configuration identical except output_subdir / store_names / source"] = (
        set(diff) <= ALLOWED_CONFIG_DIFF, diff)
    checks["same mode"] = (mt.get("mode") == mr.get("mode"), [mt.get("mode"), mr.get("mode")])
    checks["inverse methods wMNE (target) / LCMV (reference)"] = (
        (mt["source_operator"]["method"], mr["source_operator"]["method"]) == ("wmne", "lcmv"),
        [mt["source_operator"]["method"], mr["source_operator"]["method"]])
    st, sr = (pd.read_csv(d / "subjects.csv") for d in (target, reference))
    cols = [c for c in SUBJECT_COLS if c in sr]
    missing = [c for c in SUBJECT_COLS if c not in st or c not in sr]
    a = st[cols].sort_values("key").reset_index(drop=True).astype(str)
    b = sr[cols].sort_values("key").reset_index(drop=True).astype(str)
    bad = sorted(set(a["key"]) ^ set(b["key"]))
    if not bad and len(a) == len(b):
        bad = [f"{a.loc[i, 'key']}: {c}" for c in cols for i in np.flatnonzero((a[c] != b[c]).to_numpy())]
    checks["participants, labels, selection and counts identical"] = (not bad and not missing, (missing + bad)[:10])
    wt, wr = (pd.read_csv(d / "windows.csv") for d in (target, reference))
    same_len = len(wt) == len(wr)
    wbad = [] if same_len else [f"{len(wt)} vs {len(wr)} windows"]
    if same_len:
        for c in WINDOW_COLS:
            n = int((wt[c].astype(str).to_numpy() != wr[c].astype(str).to_numpy()).sum())
            if n:
                wbad.append(f"{c}: {n} rows differ")
        if not np.allclose(wt["sample_weight"], wr["sample_weight"], rtol=0, atol=1e-12):
            wbad.append("sample_weight differs")
    checks["windows identical (ids, participants, start/stop, segment, piece, tenfold, weights)"] = (not wbad, wbad)
    xt, xr = (json.loads((d / "xu_split.json").read_text()) for d in (target, reference))
    checks["frozen split identical"] = (xt == xr, "")
    report, mbad, sbad = {}, [], []
    with np.load(target / "graphs.npz") as zt, np.load(reference / "graphs.npz") as zr:
        for k in META_ARRAYS:
            if not np.array_equal(zt[k], zr[k]):
                mbad.append(k)
        if same_len and not wbad:
            for k in SENSOR:
                x, y = zt[k], zr[k]
                rel = float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-30))
                report[k] = {"max_rel_diff": rel}
                if x.shape != y.shape or rel > sensor_rtol:
                    sbad.append(f"{k}: max rel diff {rel:.2e}")
            for k in SOURCE:
                x, y = zt[k], zr[k]
                step = 512                                     # chunked: no full-size temporary
                dsum = sum(float(np.abs(x[i:i + step] - y[i:i + step]).sum(dtype=np.float64))
                           for i in range(0, len(x), step))
                report[k] = {"shape": list(x.shape), "mean_target": float(x.mean(dtype=np.float64)),
                             "mean_reference": float(y.mean(dtype=np.float64)),
                             "mean_abs_diff": dsum / max(x.size, 1)}
                del x, y
    checks["bands, band edges, ROI order and channels identical"] = (not mbad, mbad)
    checks["sensor-space features equal (Exp 0 consistency control)"] = (not sbad and bool(report), sbad)
    out = {"target": str(target), "reference": str(reference), "graph_run_id": mt.get("graph_run_id"),
           "reference_graph_run_id": mr.get("graph_run_id"), "n_windows": int(len(wt)),
           "checks": {k: {"passed": bool(v[0]), "detail": v[1]} for k, v in checks.items()}, "arrays": report}
    out["passed"] = all(c["passed"] for c in out["checks"].values())
    return out


def require_matched(target: str | Path, reference: str | Path) -> dict:
    """Raise unless ``match.json`` of ``target`` passed for its current graph_run_id and the current reference."""
    target = Path(target)
    path = target / "match.json"
    if not path.exists():
        raise SystemExit(f"{target.name} has not been matched against the LCMV store; run "
                         "`python scripts/wx_build.py match` first")
    res = json.loads(path.read_text())
    gid = json.loads((target / "manifest.json").read_text()).get("graph_run_id")
    rid = json.loads((Path(reference) / "manifest.json").read_text()).get("graph_run_id")
    if (res.get("graph_run_id"), res.get("reference_graph_run_id")) != (gid, rid):
        raise SystemExit(f"{path} refers to another store or reference; run `python scripts/wx_build.py match`")
    if not res.get("passed"):
        failed = [k for k, c in res["checks"].items() if not c["passed"]]
        raise SystemExit(f"{target.name} does not match the LCMV store ({failed}); experiments refuse it")
    return res
