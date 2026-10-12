"""End-to-end graph construction: raw EDF -> multi-band source-space graphs + PSD features.

Order per recording: load -> continuous preprocessing (resample, band-pass,
bad channels, average reference) -> continuous ROI signals (shared operator W)
-> continuous band filtering + Hilbert -> segmentation into 20 s / 50 % windows
-> artifact rejection -> per-window PLV, ciPLV, source PSD, sensor PSD.

Outputs follow the schema in ``adgraph.graphs.store`` and are written to a new,
timestamped directory; nothing is overwritten.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import common_channels, config_hash
from .connectivity.plv import band_analytic, phase_sync
from .features.psd import check_relative_range, window_band_powers
from .graphs.store import PSD_KINDS, WINDOW_COLUMNS, GraphStore, window_id
from .io.eeg import list_recordings, load_raw, make_montage
from .io.participants import load_participants
from .preprocessing.clean import preprocess
from .preprocessing.windows import reject_windows, window_samples, window_starts
from .provenance import git_commit, versions
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
    """Windows, rejection and per-window features for one cleaned, continuous recording.

    Filtering (sensor band-pass upstream; per-band ROI filtering here) is applied to
    the continuous signal before segmentation.
    """
    sfreq = raw.info["sfreq"]
    wcfg = cfg["windowing"]
    data = raw.get_data()
    win_len, step = window_samples(sfreq, wcfg["window_s"], wcfg["step_s"])
    starts = window_starts(data.shape[1], sfreq, wcfg["window_s"], wcfg["step_s"], wcfg["edge_trim_s"])
    keep, thr, ptp = reject_windows(data, starts, win_len, wcfg["rejection"]["mad_k"],
                                    wcfg["rejection"]["max_bad_channel_fraction"])
    kept_idx = np.flatnonzero(keep)
    kept = starts[kept_idx]

    roi = W @ data  # (n_rois, n_times): continuous source-space ROI signals
    bands = cfg["connectivity"]["bands"]
    n_w, n_b, n_r, n_c = len(kept), len(bands), roi.shape[0], data.shape[0]
    plv = np.zeros((n_w, n_b, n_r, n_r), np.float32)
    ciplv = np.zeros_like(plv)
    ciplv_signed = np.zeros_like(plv)
    # Band-pass + Hilbert once per band on the continuous ROI signals, then cut windows.
    for b, (lo, hi) in enumerate(bands.values()):
        analytic = band_analytic(roi, sfreq, lo, hi)
        for w, s in enumerate(kept):
            plv[w, b], ciplv[w, b], ciplv_signed[w, b] = phase_sync(analytic[:, s:s + win_len])

    psd_cfg = cfg["psd"]
    out = {"plv": plv, "ciplv": ciplv}
    if cfg["output"].get("save_signed_ciplv", True):
        out["ciplv_signed"] = ciplv_signed
    for prefix, sig, n in (("psd", roi, n_r), ("sensor_psd", data, n_c)):
        if prefix == "sensor_psd" and not psd_cfg.get("sensor", True):
            continue
        feats = {k: np.zeros((n_w, n_b, n), np.float32) for k in PSD_KINDS}
        for w, s in enumerate(kept):
            bp = window_band_powers(sig[:, s:s + win_len], sfreq, bands, psd_cfg)
            for k in PSD_KINDS:
                feats[k][w] = bp[k]
        out.update({f"{prefix}_{k}": v for k, v in feats.items()})

    thr_exceed = (ptp > thr).mean(axis=1) if len(starts) else np.empty(0)
    out.update({
        "window_index": kept_idx, "start_sample": kept, "stop_sample": kept + win_len,
        "tmin": kept / sfreq, "tmax": (kept + win_len) / sfreq,
        "qc_frac_channels_over_threshold": thr_exceed[kept_idx] if len(starts) else np.empty(0),
        "qc_max_ptp_uv": ptp[kept_idx].max(axis=1) * 1e6 if n_w else np.empty(0),
        "n_windows_total": len(starts), "n_windows_kept": n_w, "rejection_threshold_uv": thr * 1e6,
        "roi_timeseries": roi.astype(np.float32),
    })
    return out


def run(paths: dict, cfg: dict, out_root: str | Path, subjects_dir: str | Path,
        limit: int | None = None) -> Path:
    """Build graphs for all D1 and D2 resting-state recordings."""
    pp = cfg["preprocessing"]
    check_relative_range(cfg["psd"]["relative_range"], pp["l_freq"], pp["h_freq"])
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

    array_keys = ["plv", "ciplv"] + [f"psd_{k}" for k in PSD_KINDS]
    if cfg["output"].get("save_signed_ciplv", True):
        array_keys.append("ciplv_signed")
    if cfg["psd"].get("sensor", True):
        array_keys += [f"sensor_psd_{k}" for k in PSD_KINDS]
    arrays = {k: [] for k in array_keys}
    save_ts = cfg["output"].get("save_roi_timeseries", False)
    if save_ts:
        (out_dir / "roi_timeseries").mkdir()
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
        subj_qc = {
            "median_std_uv": pp_qc["median_std_uv"], "n_bad_channels": pp_qc["n_bad_channels"],
            "rejection_threshold_uv": res["rejection_threshold_uv"],
            "n_windows_total": res["n_windows_total"], "n_windows_kept": n,
        }
        subj_rows.append({
            **row, **{k: load_qc[k] for k in ("orig_sfreq", "duration_s", "unit_scale")}, **subj_qc,
            "bad_channels": json.dumps(pp_qc["bad_channels"]), "bad_fraction_warning": pp_qc["bad_fraction_warning"],
            "unrecognised_channels": ",".join(load_qc["unrecognised_channels"]), "status": status,
        })
        if status != "ok":
            continue
        for k in arrays:
            arrays[k].append(res[k])
        if save_ts:
            np.save(out_dir / "roi_timeseries" / f"{rec.key}.npy", res["roi_timeseries"])
        for w in range(n):
            win_rows.append({
                "window_id": window_id(rec.key, int(res["window_index"][w])), "key": rec.key,
                "dataset": rec.dataset, "subject_id": rec.subject_id, "label": rec.label, "y": rec.y,
                "window_index": int(res["window_index"][w]), "start_sample": int(res["start_sample"][w]),
                "stop_sample": int(res["stop_sample"][w]), "tmin": res["tmin"][w], "tmax": res["tmax"][w],
                "sample_weight": 1.0 / n,
                "qc_frac_channels_over_threshold": res["qc_frac_channels_over_threshold"][w],
                "qc_max_ptp_uv": res["qc_max_ptp_uv"][w],
                **{f"subject_{k}": v for k, v in subj_qc.items()},
            })
        log.info("%s: %d/%d windows, %d bad channels", rec.key, n, res["n_windows_total"], pp_qc["n_bad_channels"])

    bands = cfg["connectivity"]["bands"]
    n_b, n_r, n_c = len(bands), len(op["roi_names"]), len(op["ch_names"])
    empty_shape = {k: (0, n_b, n_r, n_r) for k in ("plv", "ciplv", "ciplv_signed")}
    stacked = {k: np.concatenate(v) if v else np.empty(empty_shape.get(k, (0, n_b, n_c if "sensor" in k else n_r)),
                                                         np.float32) for k, v in arrays.items()}
    windows = pd.DataFrame(win_rows)
    subjects = pd.DataFrame(subj_rows)

    manifest = {
        "run_timestamp": stamp, "config_hash": chash, "git_commit": git_commit(), "versions": versions(),
        "config": cfg,
        "inputs": recs[["key", "path", "size_bytes"]].to_dict("records"),
        "registry": {"recordings_without_metadata": no_meta, "metadata_without_recording": no_rec,
                     "subset_limit": limit},
        "source_operator": {"shape": list(op["W"].shape), "n_sources": op["n_sources"], "channels": op["ch_names"],
                            "shared_across_datasets": True, "verification_rel_error": op_err,
                            "roi_names": op["roi_names"], "W": op["W"].tolist()},
        "segmentation": {"window_samples": window_samples(cfg["preprocessing"]["sfreq"], cfg["windowing"]["window_s"],
                                                          cfg["windowing"]["step_s"])[0],
                         "step_samples": window_samples(cfg["preprocessing"]["sfreq"], cfg["windowing"]["window_s"],
                                                        cfg["windowing"]["step_s"])[1]},
        "summary": {
            "n_graphs": int(len(windows)),
            "subjects_ok": subjects.loc[subjects["status"] == "ok", "key"].tolist(),
            "subjects_not_ok": subjects.loc[subjects["status"] != "ok", ["key", "status"]].to_dict("records"),
            "per_dataset": {ds: {"subjects": int((subjects["dataset"].eq(ds) & subjects["status"].eq("ok")).sum()),
                                 "graphs": int(windows["dataset"].eq(ds).sum()) if len(windows) else 0}
                            for ds in DATASETS},
        },
        "outputs": ["graphs.npz", "windows.csv", "subjects.csv", "manifest.json"]
                   + (["roi_timeseries/"] if save_ts else []),
    }
    if windows.empty:
        windows = pd.DataFrame(columns=list(WINDOW_COLUMNS))
    store = GraphStore(stacked, windows, list(bands), np.array(list(bands.values()), dtype=float),
                       op["roi_names"], op["ch_names"], subjects, manifest, roi_coords=op.get("roi_coords"))
    store.save(out_dir)
    return out_dir
