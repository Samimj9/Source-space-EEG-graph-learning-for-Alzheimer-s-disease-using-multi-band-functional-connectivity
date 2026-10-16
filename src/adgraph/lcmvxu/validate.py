"""Checks of an assembled LCMV / Xu run (read-only): boundary safety, participant split, shapes, finiteness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def boundary_violations(windows: pd.DataFrame, subjects: pd.DataFrame, margin: int, sfreq: float) -> list[str]:
    """Windows not inside one continuous segment with ``margin`` samples to both segment ends, or with an EEGLAB
    boundary closer than ``margin`` (all positions at the 250 Hz rate, relative to the recording start)."""
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
    checks["no window crosses a boundary or its filter margin"] = (not viol, viol[:5])
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
    out = {"run_dir": str(run_dir), "mode": man.get("mode"), "n_windows": int(len(windows)),
           "n_participants_ok": int((subjects.status == "ok").sum()),
           "n_participants_not_ok": int((subjects.status != "ok").sum()),
           "checks": {k: {"passed": bool(v[0]), "detail": v[1]} for k, v in checks.items()}}
    out["passed"] = all(c["passed"] for c in out["checks"].values())
    return out
