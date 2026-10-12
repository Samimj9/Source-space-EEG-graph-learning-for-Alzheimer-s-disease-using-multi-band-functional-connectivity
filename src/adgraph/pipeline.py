"""End-to-end graph construction: raw EDF -> multi-band source-space connectivity graphs.

Outputs (written to a new, timestamped directory; nothing is overwritten):
    connectivity.npz  plv / ciplv: (n_windows, n_bands, n_rois, n_rois) float32,
                      power: (n_windows, n_bands, n_rois), plus bands, roi_names, channels
    windows.csv       one row per window, row i <-> array index i
    subjects.csv      one row per subject: metadata + QC
    manifest.json     config, config hash, git commit, versions, inputs, QC summary
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import common_channels, config_hash
from .connectivity.phase import band_analytic, log_power, phase_sync
from .io.eeg import list_recordings, load_raw, make_montage
from .io.participants import load_participants
from .preprocessing.clean import preprocess
from .preprocessing.windows import reject_windows, window_starts
from .source.inverse import build_roi_operator, fetch_fsaverage, load_labels

log = logging.getLogger(__name__)

DATASETS = ("D1", "D2")


def canonical_info(cfg: dict):
    """Sensor ``Info`` shared by both datasets: common channels, montage, average-reference projection."""
    import mne

    info = mne.create_info(common_channels(cfg), cfg["preprocessing"]["sfreq"], "eeg")
    info.set_montage(make_montage(cfg["channels"]["montage"]))
    raw = mne.io.RawArray(np.zeros((len(info["ch_names"]), 1)), info, verbose="error")
    raw.set_eeg_reference("average", projection=True, verbose="error")
    raw.apply_proj(verbose="error")
    return raw.info


def verify_operator(op: dict, info, cfg: dict, subjects_dir: str | Path, seed: int = 0) -> float:
    """Max relative error between ``W @ x`` and the full MNE inverse + label extraction on random data."""
    import mne
    from mne.minimum_norm import apply_inverse_raw, make_inverse_operator

    scfg = cfg["source"]
    fs_dir = Path(subjects_dir) / "fsaverage"
    x = np.random.default_rng(seed).standard_normal((len(info["ch_names"]), 200)) * 1e-5
    raw = mne.io.RawArray(x, info, verbose="error")
    raw.apply_proj(verbose="error")
    src = mne.read_source_spaces(fs_dir / "bem" / f"fsaverage-{scfg['spacing'].replace('ico', 'ico-')}-src.fif",
                                 verbose="error")
    fwd = mne.make_forward_solution(raw.info, "fsaverage", src, str(fs_dir / "bem" / scfg["bem"]), eeg=True,
                                    meg=False, mindist=scfg["mindist_mm"], verbose="error")
    inv = make_inverse_operator(raw.info, fwd, mne.make_ad_hoc_cov(raw.info, verbose="error"),
                                loose=scfg["loose"], depth=scfg["depth"], verbose="error")
    stc = apply_inverse_raw(raw, inv, scfg["lambda2"], scfg["method"], pick_ori=scfg["pick_ori"], verbose="error")
    ref = mne.extract_label_time_course(stc, load_labels(subjects_dir, scfg["parcellation"]), inv["src"],
                                        mode=scfg["roi_mode"], verbose="error")
    return float(np.abs(op["W"] @ raw.get_data() - ref).max() / np.abs(ref).max())


def process_recording(raw, W: np.ndarray, cfg: dict) -> dict:
    """Windows, rejection and per-window multi-band connectivity for one cleaned recording."""
    sfreq = raw.info["sfreq"]
    wcfg = cfg["windowing"]
    data = raw.get_data()
    starts = window_starts(data.shape[1], sfreq, wcfg["window_s"], wcfg["step_s"], wcfg["edge_trim_s"])
    win_len = int(round(wcfg["window_s"] * sfreq))
    keep, thr, _ = reject_windows(data, starts, win_len, wcfg["rejection"]["mad_k"],
                                  wcfg["rejection"]["max_bad_channel_fraction"])
    kept = starts[keep]

    roi = W @ data  # (n_rois, n_times): continuous source-space ROI signals
    bands = cfg["connectivity"]["bands"]
    n_w, n_b, n_r = len(kept), len(bands), roi.shape[0]
    plv = np.zeros((n_w, n_b, n_r, n_r), np.float32)
    ciplv = np.zeros_like(plv)
    power = np.zeros((n_w, n_b, n_r), np.float32)
    # Filter the continuous signal once per band, then cut windows (no per-window filter edges).
    for b, (lo, hi) in enumerate(bands.values()):
        analytic = band_analytic(roi, sfreq, lo, hi)
        for w, s in enumerate(kept):
            seg = analytic[:, s:s + win_len]
            plv[w, b], ciplv[w, b] = phase_sync(seg)
            power[w, b] = log_power(seg)
    return {
        "plv": plv, "ciplv": ciplv, "power": power,
        "tmin": kept / sfreq, "tmax": (kept + win_len) / sfreq,
        "n_windows_total": len(starts), "n_windows_kept": n_w, "rejection_threshold_uv": thr * 1e6,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _versions() -> dict:
    import mne
    import scipy

    return {"python": platform.python_version(), "mne": mne.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__}


def run(paths: dict, cfg: dict, out_root: str | Path, subjects_dir: str | Path,
        limit: int | None = None) -> Path:
    """Build graphs for all D1 and D2 resting-state recordings."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    chash = config_hash(cfg)
    out_dir = Path(out_root) / f"graphs_v{cfg['version']}_{stamp}_{chash}"
    out_dir.mkdir(parents=True, exist_ok=False)

    # Subject registry: recordings joined to metadata. All subjects are used as provided.
    recs, meta = [], []
    for ds in DATASETS:
        recs.append(list_recordings(paths[f"dataset{ds[1]}"]["eeg_dir"], ds))
        meta.append(load_participants(paths[f"dataset{ds[1]}"]["participants"], ds))
    recs, meta = pd.concat(recs, ignore_index=True), pd.concat(meta, ignore_index=True)
    registry = recs.merge(meta.drop(columns=["dataset", "subject_id"]), on="key", how="outer", indicator=True)
    no_meta = registry.loc[registry["_merge"] == "left_only", "key"].tolist()
    no_rec = registry.loc[registry["_merge"] == "right_only", "key"].tolist()
    registry = (registry[registry["_merge"] != "right_only"].drop(columns="_merge")
                .sort_values(["dataset", "subject_id"]).reset_index(drop=True))
    if limit:
        registry = registry.groupby("dataset", group_keys=False).head(limit).reset_index(drop=True)

    # One sensor->ROI operator for both datasets.
    fetch_fsaverage(subjects_dir)
    info = canonical_info(cfg)
    op = build_roi_operator(info, cfg, subjects_dir)
    op_err = verify_operator(op, info, cfg, subjects_dir)
    if op_err > 1e-6:
        raise RuntimeError(f"ROI operator does not match the full inverse (rel. err {op_err:.2e})")
    if op["fwd_ch_names"] != common_channels(cfg):
        raise RuntimeError(f"forward model channels {op['fwd_ch_names']} != common set")
    log.info("ROI operator %s, %d sources, verified (rel. err %.1e)", op["W"].shape, op["n_sources"], op_err)

    arrays = {"plv": [], "ciplv": [], "power": []}
    win_rows, subj_rows = [], []
    for rec in registry.itertuples():
        row = {k: getattr(rec, k) for k in ("key", "dataset", "subject_id", "label", "y", "age", "sex",
                                             "mmse", "acer", "comment", "path", "size_bytes")}
        try:
            raw, load_qc = load_raw(rec.path, rec.dataset, cfg)
            raw, pp_qc = preprocess(raw, cfg)
            if raw.ch_names != op["ch_names"]:
                raise RuntimeError(f"channel order {raw.ch_names} differs from operator")
            res = process_recording(raw, op["W"], cfg)
        except Exception as exc:  # recorded in subjects.csv and the manifest, never silent
            log.exception("%s failed", rec.key)
            subj_rows.append({**row, "status": f"error: {exc}"})
            continue

        n = res["n_windows_kept"]
        status = "ok" if n >= cfg["windowing"]["min_windows"] else "below_min_windows"
        subj_rows.append({
            **row, **{k: load_qc[k] for k in ("orig_sfreq", "duration_s", "unit_scale")},
            "median_std_uv": pp_qc["median_std_uv"], "n_bad_channels": pp_qc["n_bad_channels"],
            "bad_channels": json.dumps(pp_qc["bad_channels"]), "bad_fraction_warning": pp_qc["bad_fraction_warning"],
            "unrecognised_channels": ",".join(load_qc["unrecognised_channels"]),
            "n_windows_total": res["n_windows_total"], "n_windows_kept": n,
            "rejection_threshold_uv": res["rejection_threshold_uv"], "status": status,
        })
        if status != "ok":
            continue
        for k in arrays:
            arrays[k].append(res[k])
        for t0, t1 in zip(res["tmin"], res["tmax"]):
            win_rows.append({"key": rec.key, "dataset": rec.dataset, "subject_id": rec.subject_id,
                             "label": rec.label, "y": rec.y, "tmin": t0, "tmax": t1, "sample_weight": 1.0 / n})
        log.info("%s: %d/%d windows, %d bad channels", rec.key, n, res["n_windows_total"], pp_qc["n_bad_channels"])

    bands = list(cfg["connectivity"]["bands"])
    stacked = {k: np.concatenate(v) if v else np.empty(0, np.float32) for k, v in arrays.items()}
    np.savez_compressed(out_dir / "connectivity.npz", **{m: stacked[m] for m in cfg["connectivity"]["metrics"]},
                        power=stacked["power"], bands=np.array(bands), roi_names=np.array(op["roi_names"]),
                        channels=np.array(op["ch_names"]), W=op["W"])
    windows = pd.DataFrame(win_rows)
    subjects = pd.DataFrame(subj_rows)
    windows.to_csv(out_dir / "windows.csv", index=False)
    subjects.to_csv(out_dir / "subjects.csv", index=False)

    manifest = {
        "run_timestamp": stamp, "config_hash": chash, "git_commit": _git_commit(), "versions": _versions(),
        "config": cfg,
        "inputs": recs[["key", "path", "size_bytes"]].to_dict("records"),
        "registry": {"recordings_without_metadata": no_meta, "metadata_without_recording": no_rec,
                     "subset_limit": limit},
        "source_operator": {"shape": list(op["W"].shape), "n_sources": op["n_sources"], "channels": op["ch_names"],
                            "shared_across_datasets": True, "verification_rel_error": op_err,
                            "roi_names": op["roi_names"]},
        "summary": {
            "n_graphs": int(len(windows)),
            "subjects_ok": subjects.loc[subjects["status"] == "ok", "key"].tolist(),
            "subjects_not_ok": subjects.loc[subjects["status"] != "ok", ["key", "status"]].to_dict("records"),
            "per_dataset": {ds: {"subjects": int((subjects["dataset"].eq(ds) & subjects["status"].eq("ok")).sum()),
                                 "graphs": int(windows["dataset"].eq(ds).sum()) if len(windows) else 0}
                            for ds in DATASETS},
        },
        "outputs": ["connectivity.npz", "windows.csv", "subjects.csv", "manifest.json"],
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return out_dir
