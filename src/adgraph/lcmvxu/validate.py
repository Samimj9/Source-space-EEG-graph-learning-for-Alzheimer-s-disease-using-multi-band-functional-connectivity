"""Checks of an assembled LCMV / Xu run (read-only): boundary safety, participant split, shapes, finiteness."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd


def boundary_violations(windows: pd.DataFrame, subjects: pd.DataFrame, margin: int, sfreq: float) -> list[str]:
    """Windows not inside one continuous segment with ``margin`` samples to both segment ends, or with an EEGLAB
    boundary closer than ``margin``.

    Stores with exact positions (``segments_in`` / ``boundary_cuts_in``, methodology log entry 53) are checked in
    exact rational arithmetic at the INPUT rate (``boundary_violations_exact``). Older stores only hold the cuts
    as seconds rounded to 1 ms and the segment starts rounded to the 250 Hz grid; for them the legacy check below
    is kept unchanged (with the same strict conditions it can flag a window lying exactly at the margin)."""
    if "segments_in" in subjects and "boundary_cuts_in" in subjects:
        return boundary_violations_exact(windows, subjects, margin, sfreq)
    bad = []
    subj = subjects.set_index("key")
    for key, g in windows.groupby("key"):
        segs = json.loads(subj.loc[key, "segments_samples"])
        cuts = [c * sfreq for c in json.loads(subj.loc[key, "boundary_positions_s"])]
        for s, e in zip(g["start_sample"].to_numpy(), g["stop_sample"].to_numpy()):
            inside = [a + margin <= s and e <= b - margin for a, b in segs]
            near = [c for c in cuts if s - margin < c < e + margin]
            if not any(inside) or near:
                bad.append(f"{key}: window [{s}, {e})")
    return bad


def boundary_violations_exact(windows: pd.DataFrame, subjects: pd.DataFrame, margin: int, sfreq: float) -> list[str]:
    """Exact check at the input rate (no floating point). A window of segment ``j`` (input start ``a``, trimmed
    length ``n_in``, 250 Hz origin ``o``) with ``start_sample`` ``s`` covers the input positions
    [p, q) = [a + (s - o) r, a + (s - o + W) r), r = sf_in / sf. It passes only if it lies in ITS OWN segment
    with the margin M r to both ends (a + M r <= p and q <= a + n_in - M r) and no boundary cut c satisfies
    p - M r < c < q + M r. The conditions are those of the legacy check, evaluated exactly."""
    bad = []
    subj = subjects.set_index("key")
    for key, g in windows.groupby("key"):
        r = Fraction(float(subj.loc[key, "orig_sfreq"])) / Fraction(sfreq)
        segs = {int(j): (int(a), int(n), int(o)) for j, a, n, o in json.loads(subj.loc[key, "segments_in"])}
        cuts = [int(c) for c in json.loads(subj.loc[key, "boundary_cuts_in"])]
        m = margin * r
        for j, s, e in zip(g["segment_index"].to_numpy(), g["start_sample"].to_numpy(), g["stop_sample"].to_numpy()):
            if int(j) not in segs:
                bad.append(f"{key}: window [{s}, {e}) in unknown segment {j}")
                continue
            a, n_in, o = segs[int(j)]
            p, q = a + (int(s) - o) * r, a + (int(e) - o) * r
            inside = a + m <= p and q <= a + n_in - m
            near = [c for c in cuts if p - m < c < q + m]
            if not inside or near:
                bad.append(f"{key}: window [{s}, {e})")
    return bad


def selection_excess(windows: pd.DataFrame, subjects: pd.DataFrame, sfreq: float) -> list[str]:
    """Participants whose selected EEG exceeds the five-minute target: from the reported selection AND from the
    windows themselves (union of the window spans, which never exceeds the selection)."""
    out = []
    for row in subjects.itertuples():
        if row.status != "ok":
            continue
        target = float(row.selection_target_s)
        if float(row.selected_duration_s) > target + 1e-9:
            out.append(f"{row.key}: selected {row.selected_duration_s} s > {target} s")
        g = windows[windows["key"] == row.key]
        spans = sorted(zip(g["start_sample"], g["stop_sample"]))
        covered, end = 0, None
        for s, e in spans:
            if end is None or s >= end:
                covered += e - s
            elif e > end:
                covered += e - end
            end = e if end is None else max(end, e)
        if covered > round(target * sfreq):
            out.append(f"{row.key}: windows cover {covered / sfreq} s > {target} s")
    return out


def require_validated(run_dir: str | Path) -> dict:
    """Raise unless ``validation.json`` of this store passed AND refers to the store's current graph_run_id
    (experiments of the LX / LX10 families use only validated stores)."""
    run_dir = Path(run_dir)
    vpath = run_dir / "validation.json"
    if not vpath.exists():
        raise SystemExit(f"{run_dir.name} has not been validated; run `python scripts/lx_build.py validate` first")
    res = json.loads(vpath.read_text())
    gid = json.loads((run_dir / "manifest.json").read_text()).get("graph_run_id")
    if res.get("graph_run_id") != gid:
        raise SystemExit(f"{vpath} does not refer to graph run {gid}; run `python scripts/lx_build.py validate`")
    if not res.get("passed"):
        failed = [k for k, c in res["checks"].items() if not c["passed"]]
        raise SystemExit(f"{run_dir.name} FAILED validation ({failed}); experiments refuse it")
    return res


def check_split(windows: pd.DataFrame, subjects: pd.DataFrame, split: dict) -> list[str]:
    from ..evaluation.splits import xu_partition

    problems = []
    ok = subjects.loc[subjects["status"] == "ok", ["key", "dataset", "y"]]
    dev, test = xu_partition(ok)
    if (dev, test) != (split["development"], split["test"]):
        problems.append("saved split differs from xu_partition")
    if set(dev) & set(test):
        problems.append(f"participants in both partitions: {sorted(set(dev) & set(test))}")
    if set(dev) | set(test) != set(ok["key"]):
        problems.append("a participant is in no partition")
    for name, keys in (("development", dev), ("test", test)):
        if ok[ok.key.isin(keys)]["y"].nunique() < 2:
            problems.append(f"{name} partition lacks a class")
    # every window of a participant lies in that participant's partition (by construction: split by key)
    side = {**{k: "dev" for k in dev}, **{k: "test" for k in test}}
    if windows["key"].map(side).isna().any():
        problems.append("windows of participants without a partition")
    return problems


def validate_run(run_dir: str | Path, check_arrays: bool = True) -> dict:
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    windows = pd.read_csv(run_dir / "windows.csv")
    subjects = pd.read_csv(run_dir / "subjects.csv")
    split = json.loads((run_dir / "xu_split.json").read_text())
    seg = man["segmentation"]
    sf = float(man["config"]["preprocessing"]["sfreq"])
    checks = {}
    viol = boundary_violations(windows, subjects, seg["margin_samples"], sf)
    exact = "segments_in" in subjects and "boundary_cuts_in" in subjects
    checks["no window crosses a boundary or its filter margin"
           + (" (exact input-rate positions)" if exact else " (legacy rounded positions)")] = (not viol, viol[:5])
    excess = selection_excess(windows, subjects, sf)
    checks["selected EEG at most the five-minute target"] = (not excess, excess[:5])
    if (man.get("registry") or {}).get("subset_limit"):
        checks["participant split (not applicable: --limit smoke test)"] = (True, "")
    else:
        probs = check_split(windows, subjects, split)
        checks["participant split disjoint, complete, frozen"] = (not probs, probs)
    checks["window length 10 s, step 1 s"] = (bool(((windows.stop_sample - windows.start_sample)
                                                     == seg["window_samples"]).all()), "")
    per = windows.groupby("key").size()
    checks["at most 291 windows per participant"] = (bool((per <= 291).all()), per.describe().round(1).to_dict())
    if check_arrays:
        n, r = len(windows), len(man["source_operator"]["roi_names"])
        k = len(man["config"]["bands"])
        bad = []
        with np.load(run_dir / "graphs.npz") as z:
            for a in z.files:
                if a in ("bands", "band_edges", "roi_names", "channels", "roi_coords"):
                    continue
                v = z[a]
                want = (n, k, r, r) if a in ("plv", "ciplv") else (n, k, len(man["source_operator"]["channels"])) \
                    if a.startswith("sensor") else (n, k, r)
                if v.shape != want or not np.isfinite(v).all():
                    bad.append(f"{a}: shape {v.shape} (want {want}), finite {np.isfinite(v).all()}")
                del v
        checks["arrays: 68 ROIs x 5 bands (19 sensors), finite"] = (not bad, bad)
    w = windows.groupby("key")["sample_weight"].sum() if "sample_weight" in windows else pd.Series(dtype=float)
    checks["sample weights sum to 1 per participant"] = (bool(np.allclose(w.to_numpy(), 1.0)), "")
    ok = subjects.loc[subjects.status == "ok"].set_index("key")
    n_col = "n_tenfold" if man.get("mode") == "A_xu_tenfold" else "n_windows_kept"
    mism = [k for k, n in windows.groupby("key").size().items() if k not in ok.index or int(ok.loc[k, n_col]) != n]
    mism += [k for k in ok.index if k not in set(windows["key"])]
    checks[f"windows per participant equal subjects.csv {n_col}"] = (not mism, mism[:5])
    out = {"run_dir": str(run_dir), "graph_run_id": man.get("graph_run_id"), "mode": man.get("mode"),
           "correction": (man.get("correction") or {}).get("name"), "n_windows": int(len(windows)),
           "n_participants_ok": int((subjects.status == "ok").sum()),
           "n_participants_not_ok": int((subjects.status != "ok").sum()),
           "checks": {k: {"passed": bool(v[0]), "detail": v[1]} for k, v in checks.items()}}
    out["passed"] = all(c["passed"] for c in out["checks"].values())
    return out
