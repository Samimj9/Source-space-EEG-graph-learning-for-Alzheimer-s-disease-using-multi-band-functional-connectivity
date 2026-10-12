"""End-to-end graph construction: raw EDF -> multi-band source-space graphs + PSD features.

Order per recording: load -> continuous preprocessing (resample, band-pass,
bad channels, average reference) -> continuous ROI signals (shared operator W)
-> continuous band filtering + Hilbert -> segmentation into 20 s / 50 % windows
-> artifact rejection -> per-window PLV, ciPLV, source PSD, sensor PSD.

Outputs follow the schema in ``adgraph.graphs.store`` and are written, resumably, to a
directory keyed by the config hash; nothing is overwritten. Checkpoint identities carry
the input files' SHA-256 and an automatic code fingerprint (see ``adgraph.checkpoint``).
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
from .io.fingerprint import InputHashes, auxiliary_inputs
from .io.participants import load_participants
from .preprocessing.clean import preprocess
from .preprocessing.windows import reject_windows, window_samples, window_starts
from .checkpoint import (CODE_VERSIONS, IncompatibleCheckpoint, StatusTable, atomic_save_npy, atomic_savez,
                         atomic_write_csv, atomic_write_json, check_unit, code_identity, identity_diff, stable_hash,
                         supersede, write_marker)
from .provenance import git_commit, git_dirty, versions
from .storage import check_persistent
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


# ====================================================================== resumable run
DATA_ARRAY_KEYS = ("plv", "ciplv", "ciplv_signed", "psd_abs", "psd_log", "psd_rel",
                   "sensor_psd_abs", "sensor_psd_log", "sensor_psd_rel")


def graph_run_dir(out_root: str | Path, cfg: dict, limit: int | None = None) -> Path:
    """Deterministic run directory for a configuration (so an interrupted run can be resumed)."""
    return Path(out_root) / (f"graphs_v{cfg['version']}_{config_hash(cfg)}" + (f"_limit{limit}" if limit else ""))


def build_registry(paths: dict, limit: int | None = None, hashes: InputHashes | None = None):
    """Recordings joined to participant metadata (all subjects used as provided).

    With ``hashes``, a ``sha256`` column holds each used recording's content hash."""
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
    if hashes is not None:
        registry["sha256"] = [hashes.sha256(p) for p in registry["path"]]
    return registry, recs, no_meta, no_rec


def hash_auxiliary_inputs(paths: dict, hashes: InputHashes) -> dict:
    """Content hashes of the participant sheets and electrode layout (those that exist)."""
    return {name: {"path": str(p), "sha256": hashes.sha256(p)}
            for name, p in auxiliary_inputs(paths).items() if Path(p).exists()}


def default_hash_cache(run_dir: Path) -> Path:
    return run_dir.parent / "_input_hashes.json"


# ---------------------------------------------------------------------- shared operator
def operator_identity(cfg: dict) -> dict:
    """Everything the sensor->ROI operator depends on (the source-model configuration)."""
    import mne

    return {"kind": "operator", "code_version": CODE_VERSIONS["operator"], "source": cfg["source"],
            "channels": common_channels(cfg), "montage": cfg["channels"]["montage"],
            "reference": cfg["preprocessing"]["reference"], "mne": mne.__version__,
            "code": code_identity("operator")}


def get_operator(cfg: dict, subjects_dir: str | Path, cache_dir: str | Path, force: bool = False) -> dict:
    """Load the verified operator from the cache if its source-model hash matches; else build it once.

    fsaverage is fetched and the forward/inverse model computed only when no valid cached
    operator exists (or ``force``)."""
    ident = operator_identity(cfg)
    op_id = stable_hash(ident)
    cache_dir = Path(cache_dir)
    name = f"operator_{op_id}"
    state, marker = check_unit(cache_dir, name, ident)
    if state == "complete" and not force:
        z = np.load(cache_dir / f"{name}.npz")
        log.info("reusing cached ROI operator %s", op_id)
        return {"W": z["W"], "roi_names": z["roi_names"].tolist(), "ch_names": z["ch_names"].tolist(),
                "fwd_ch_names": z["fwd_ch_names"].tolist(), "n_sources": int(z["n_sources"]),
                "roi_coords": z["roi_coords"] if "roi_coords" in z.files else None,
                "verification_rel_error": float(z["verification_rel_error"]), "operator_id": op_id,
                "from_cache": True}
    if state in ("corrupt", "incompatible") or (state == "complete" and force):
        supersede(cache_dir / f"{name}.npz", cache_dir)
    fetch_fsaverage(subjects_dir)
    info = canonical_info(cfg)
    op = build_roi_operator(info, cfg, subjects_dir)
    op_err = verify_operator(op, info, cfg, subjects_dir)
    if op_err > 1e-6:
        raise RuntimeError(f"ROI operator does not match the full inverse (rel. err {op_err:.2e})")
    if op["fwd_ch_names"] != common_channels(cfg):
        raise RuntimeError(f"forward model channels {op['fwd_ch_names']} != common set")
    atomic_savez(cache_dir / f"{name}.npz", W=op["W"], roi_names=np.array(op["roi_names"]),
                 ch_names=np.array(op["ch_names"]), fwd_ch_names=np.array(op["fwd_ch_names"]),
                 n_sources=op["n_sources"], verification_rel_error=op_err,
                 **({"roi_coords": np.asarray(op["roi_coords"])} if op.get("roi_coords") is not None else {}))
    write_marker(cache_dir, name, ident, [f"{name}.npz"])
    log.info("ROI operator %s built and verified (rel. err %.1e)", op_id, op_err)
    return {**op, "verification_rel_error": op_err, "operator_id": op_id, "from_cache": False}


# ---------------------------------------------------------------------- per-subject units
METADATA_FIELDS = ("label", "y", "age", "sex", "mmse", "acer", "comment")


def subject_identity(cfg: dict, rec, operator_id: str) -> dict:
    """A subject's output is reusable only for the same config, code, operator, EEG content and metadata.

    The EEG file enters by its SHA-256 (not path/size/mtime), so moving the Drive mount does
    not invalidate it but any change of content does; the subject's metadata row enters by
    hash, so a corrected label or age recomputes that subject."""
    sha = getattr(rec, "sha256", None)
    if not sha:
        raise ValueError(f"{rec.key}: no input content hash; build the registry with InputHashes")
    return {"kind": "graph_subject", "code_version": CODE_VERSIONS["graph_subject"], "config_hash": config_hash(cfg),
            "operator_id": operator_id, "key": rec.key,
            "input": {"sha256": sha, "size_bytes": int(rec.size_bytes)},
            "metadata": stable_hash({k: getattr(rec, k, None) for k in METADATA_FIELDS}),
            "code": code_identity("graph")}


def process_subject(rec, cfg: dict, op: dict) -> tuple[dict, pd.DataFrame, dict, np.ndarray]:
    """Full per-subject computation: load, preprocess, ROI signals, windows, PSD, connectivity."""
    row = {k: getattr(rec, k) for k in ("key", "dataset", "subject_id", "label", "y", "age", "sex",
                                         "mmse", "acer", "comment", "path", "size_bytes")}
    raw, load_qc = load_raw(rec.path, rec.dataset, cfg)
    raw, pp_qc = preprocess(raw, cfg)
    if raw.ch_names != list(op["ch_names"]):
        raise RuntimeError(f"channel order {raw.ch_names} differs from operator")
    res = process_recording(raw, op["W"], cfg)
    n = res["n_windows_kept"]
    status = "ok" if n >= cfg["windowing"]["min_windows"] else "below_min_windows"
    subj_qc = {"median_std_uv": pp_qc["median_std_uv"], "n_bad_channels": pp_qc["n_bad_channels"],
               "rejection_threshold_uv": res["rejection_threshold_uv"],
               "n_windows_total": res["n_windows_total"], "n_windows_kept": n}
    subject_row = {**row, **{k: load_qc[k] for k in ("orig_sfreq", "duration_s", "unit_scale")}, **subj_qc,
                   "bad_channels": json.dumps(pp_qc["bad_channels"]),
                   "bad_fraction_warning": pp_qc["bad_fraction_warning"],
                   "unrecognised_channels": ",".join(load_qc["unrecognised_channels"]), "status": status}
    arrays = {k: res[k] for k in DATA_ARRAY_KEYS if k in res}
    win_rows = [{
        "window_id": window_id(rec.key, int(res["window_index"][w])), "key": rec.key,
        "dataset": rec.dataset, "subject_id": rec.subject_id, "label": rec.label, "y": rec.y,
        "window_index": int(res["window_index"][w]), "start_sample": int(res["start_sample"][w]),
        "stop_sample": int(res["stop_sample"][w]), "tmin": res["tmin"][w], "tmax": res["tmax"][w],
        "sample_weight": 1.0 / n if n else np.nan,
        "qc_frac_channels_over_threshold": res["qc_frac_channels_over_threshold"][w],
        "qc_max_ptp_uv": res["qc_max_ptp_uv"][w], **{f"subject_{k}": v for k, v in subj_qc.items()},
    } for w in range(n)]
    windows = pd.DataFrame(win_rows, columns=list(WINDOW_COLUMNS) if not win_rows else None)
    log.info("%s: %d/%d windows, %d bad channels", rec.key, n, res["n_windows_total"], pp_qc["n_bad_channels"])
    return arrays, windows, subject_row, res["roi_timeseries"]


def save_subject(unit_dir: Path, identity: dict, arrays: dict, windows: pd.DataFrame, subject_row: dict,
                 roi_ts: np.ndarray | None) -> None:
    """Payload files first (atomic), completion marker last."""
    files = [atomic_savez(unit_dir / "arrays.npz", **arrays), atomic_write_csv(unit_dir / "windows.csv", windows),
             atomic_write_json(unit_dir / "subject.json", subject_row)]
    if roi_ts is not None:
        files.append(atomic_save_npy(unit_dir / "roi_timeseries.npy", roi_ts.astype(np.float32)))
    (unit_dir / "failed.json").unlink(missing_ok=True)
    write_marker(unit_dir, "subject", identity, files)


def subject_states(run_dir: Path, registry: pd.DataFrame, cfg: dict, operator_id: str) -> pd.DataFrame:
    """Checkpoint state of every subject, read from disk only (no computation)."""
    status = StatusTable(run_dir / "status.csv")
    rows = []
    for rec in registry.itertuples():
        unit_dir = run_dir / "subjects" / rec.key
        expected = subject_identity(cfg, rec, operator_id)
        state, marker = check_unit(unit_dir, "subject", expected)
        reason = ",".join(identity_diff(marker.get("identity"), expected)) if state == "incompatible" else ""
        table = status.get(rec.key)
        if state == "complete":
            shown = "completed"
        elif (unit_dir / "failed.json").exists() and table == "failed":
            shown = "failed"
        elif table == "running":
            shown = "interrupted"            # was running when the process stopped; will be recomputed
        elif state in ("incompatible", "corrupt"):
            shown = state
        else:
            shown = "pending"
        rows.append({"key": rec.key, "dataset": rec.dataset, "state": shown, "checkpoint": state,
                     "changed": reason, "message": (json.loads((unit_dir / "failed.json").read_text()).get("status", "")
                                 if (unit_dir / "failed.json").exists() else "")})
    return pd.DataFrame(rows)


def status(paths: dict, cfg: dict, out_root: str | Path, limit: int | None = None,
           run_dir: str | Path | None = None, operator_cache: str | Path | None = None,
           input_hash_cache: str | Path | None = None) -> dict:
    """What is completed / pending / failed for this configuration, without computing anything.

    (Input files not yet in the hash cache are hashed: identities depend on their content.)"""
    run_dir = Path(run_dir) if run_dir else graph_run_dir(out_root, cfg, limit)
    registry, *_ = build_registry(paths, limit, InputHashes(input_hash_cache or default_hash_cache(run_dir)))
    ident = operator_identity(cfg)
    op_id = stable_hash(ident)
    cache = Path(operator_cache) if operator_cache else run_dir.parent / "_operator_cache"
    op_state, _ = check_unit(cache, f"operator_{op_id}", ident)
    subj = subject_states(run_dir, registry, cfg, op_id) if run_dir.exists() else \
        pd.DataFrame({"key": registry["key"], "dataset": registry["dataset"], "state": "pending",
                      "checkpoint": "missing", "changed": "", "message": ""})
    asm = "missing"
    if (run_dir / "assembly.done.json").exists():
        asm = "complete" if (run_dir / "graphs.npz").exists() else "corrupt"
    return {"run_dir": str(run_dir), "config_hash": config_hash(cfg), "operator_id": op_id,
            "git_commit": git_commit(), "git_dirty": git_dirty(),
            "operator_cache": op_state, "subjects": subj, "assembly": asm,
            "counts": subj["state"].value_counts().to_dict()}


# ---------------------------------------------------------------------- assembly
def assemble(run_dir: Path, registry: pd.DataFrame, recs: pd.DataFrame, cfg: dict, op: dict,
             no_meta: list, no_rec: list, limit: int | None, keys_failed: dict, aux_inputs: dict | None = None) -> Path:
    """Concatenate completed subject units into the cohort store (skipped if already assembled)."""
    units = {}
    for rec in registry.itertuples():
        mp = run_dir / "subjects" / rec.key / "subject.done.json"
        units[rec.key] = (json.loads(mp.read_text())["files"] if rec.key not in keys_failed and mp.exists()
                          else {"failed": keys_failed.get(rec.key, "missing")})
    ident = {"kind": "graph_assembly", "code_version": CODE_VERSIONS["graph_assembly"],
             "config_hash": config_hash(cfg), "operator_id": op["operator_id"], "units": units,
             "subset_limit": limit, "auxiliary_inputs": {k: v["sha256"] for k, v in (aux_inputs or {}).items()},
             "code": code_identity("graph")}
    state, _ = check_unit(run_dir, "assembly", ident)
    if state == "complete":
        log.info("cohort already assembled for these subject units; nothing to do")
        return run_dir
    if state != "missing":                     # units changed (e.g. a failed subject now succeeded)
        for f in ("graphs.npz", "windows.csv", "subjects.csv", "manifest.json", "assembly.done.json"):
            supersede(run_dir / f, run_dir)

    bands = cfg["connectivity"]["bands"]
    n_b, n_r, n_c = len(bands), len(op["roi_names"]), len(op["ch_names"])
    keys = [k for k in DATA_ARRAY_KEYS if (k != "ciplv_signed" or cfg["output"].get("save_signed_ciplv", True))
            and (not k.startswith("sensor") or cfg["psd"].get("sensor", True))]
    arrays = {k: [] for k in keys}
    win, subj_rows = [], []
    for rec in registry.itertuples():
        unit_dir = run_dir / "subjects" / rec.key
        if rec.key in keys_failed:
            subj_rows.append(json.loads((unit_dir / "failed.json").read_text()))
            continue
        row = json.loads((unit_dir / "subject.json").read_text())
        subj_rows.append(row)
        if row["status"] != "ok":
            continue
        z = np.load(unit_dir / "arrays.npz")
        for k in keys:
            arrays[k].append(z[k])
        win.append(pd.read_csv(unit_dir / "windows.csv"))
    shape = {k: (0, n_b, n_r, n_r) for k in ("plv", "ciplv", "ciplv_signed")}
    stacked = {k: np.concatenate(v) if v else np.empty(shape.get(k, (0, n_b, n_c if "sensor" in k else n_r)),
                                                        np.float32) for k, v in arrays.items()}
    windows = pd.concat(win, ignore_index=True) if win else pd.DataFrame(columns=list(WINDOW_COLUMNS))
    subjects = pd.DataFrame(subj_rows)
    win_s, step_s = window_samples(cfg["preprocessing"]["sfreq"], cfg["windowing"]["window_s"],
                                   cfg["windowing"]["step_s"])
    manifest = {
        "graph_run_id": stable_hash(ident), "assembled_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(cfg), "code_versions": CODE_VERSIONS, "git_commit": git_commit(),
        "git_dirty": git_dirty(), "code": ident["code"], "versions": versions(), "config": cfg,
        "inputs": registry[["key", "path", "size_bytes"] + (["sha256"] if "sha256" in registry else [])]
        .to_dict("records"),
        "auxiliary_inputs": aux_inputs or {},
        "registry": {"recordings_without_metadata": no_meta, "metadata_without_recording": no_rec,
                     "subset_limit": limit},
        "source_operator": {"operator_id": op["operator_id"], "from_cache": op.get("from_cache", False),
                            "shape": list(np.asarray(op["W"]).shape), "n_sources": op["n_sources"],
                            "channels": list(op["ch_names"]), "shared_across_datasets": True,
                            "verification_rel_error": op["verification_rel_error"],
                            "roi_names": list(op["roi_names"]), "W": np.asarray(op["W"]).tolist()},
        "segmentation": {"window_samples": win_s, "step_samples": step_s},
        "subject_units": {k: v for k, v in units.items()},
        "summary": {
            "n_graphs": int(len(windows)),
            "subjects_ok": subjects.loc[subjects["status"] == "ok", "key"].tolist(),
            "subjects_not_ok": subjects.loc[subjects["status"] != "ok", ["key", "status"]].to_dict("records"),
            "per_dataset": {ds: {"subjects": int((subjects["dataset"].eq(ds) & subjects["status"].eq("ok")).sum()),
                                 "graphs": int(windows["dataset"].eq(ds).sum()) if len(windows) else 0}
                            for ds in DATASETS},
        },
        "outputs": ["graphs.npz", "windows.csv", "subjects.csv", "manifest.json", "subjects/<key>/"],
    }
    store = GraphStore(stacked, windows, list(bands), np.array(list(bands.values()), dtype=float),
                       list(op["roi_names"]), list(op["ch_names"]), subjects, manifest,
                       roi_coords=op.get("roi_coords"))
    store.save(run_dir)
    write_marker(run_dir, "assembly", ident, ["graphs.npz", "windows.csv", "subjects.csv", "manifest.json"])
    return run_dir


# ---------------------------------------------------------------------- discovery
def list_graph_runs(processed_dir: str | Path) -> pd.DataFrame:
    """All graph run directories with their state (assembled or not, smoke test or full)."""
    rows = []
    for d in sorted(Path(processed_dir).glob("graphs_v*")):
        if not d.is_dir():
            continue
        man = d / "manifest.json"
        assembled = (d / "assembly.done.json").exists() and man.exists()
        m = json.loads(man.read_text()) if assembled else {}
        rows.append({"run_dir": str(d), "assembled": assembled, "limit_smoke_test": "_limit" in d.name,
                     "forced": "_force" in d.name, "assembled_at": m.get("assembled_at", ""),
                     "graph_run_id": m.get("graph_run_id", ""), "config_hash": m.get("config_hash", "")})
    return pd.DataFrame(rows, columns=["run_dir", "assembled", "limit_smoke_test", "forced", "assembled_at",
                                       "graph_run_id", "config_hash"])


def latest_complete_graph_run(processed_dir: str | Path) -> Path:
    """Most recently assembled full (non-smoke-test) graph run; raises if there is none."""
    runs = list_graph_runs(processed_dir)
    runs = runs[runs["assembled"] & ~runs["limit_smoke_test"]].sort_values("assembled_at")
    if runs.empty:
        raise FileNotFoundError(f"no assembled graph run in {processed_dir}")
    return Path(runs["run_dir"].iloc[-1])


# ---------------------------------------------------------------------- driver
def run(paths: dict, cfg: dict, out_root: str | Path, subjects_dir: str | Path, limit: int | None = None,
        resume: bool = True, force: bool = False, run_dir: str | Path | None = None,
        operator_cache: str | Path | None = None, input_hash_cache: str | Path | None = None,
        rehash_inputs: bool = False, allow_ephemeral: bool = False) -> Path:
    """Build graphs for all D1 and D2 resting-state recordings, resumably.

    * Each subject is a checkpointed unit (``subjects/<key>/``), saved as soon as it finishes.
    * On restart (``resume``, default) subjects whose unit validates against the current
      config hash, code version, operator and input file are skipped; missing, interrupted,
      corrupted or incompatible units are recomputed; failed subjects are retried.
    * ``force`` starts a fresh run directory (``..._force<stamp>``) and rebuilds the operator;
      existing outputs are never overwritten.
    * Input recordings are identified by SHA-256 (cached in ``input_hash_cache``; ``rehash_inputs``
      re-reads every file), code by ``code_identity`` (git commit + source hash + libraries).
    * The run directory, operator cache, fsaverage directory and hash cache must be persistent
      storage (mounted Drive in Colab), else ``EphemeralStorageError`` unless ``allow_ephemeral``.
    """
    pp = cfg["preprocessing"]
    check_relative_range(cfg["psd"]["relative_range"], pp["l_freq"], pp["h_freq"])
    run_dir = Path(run_dir) if run_dir else graph_run_dir(out_root, cfg, limit)
    operator_cache = Path(operator_cache) if operator_cache else run_dir.parent / "_operator_cache"
    input_hash_cache = Path(input_hash_cache) if input_hash_cache else default_hash_cache(run_dir)
    check_persistent({"graph run directory": run_dir, "operator cache": operator_cache,
                      "fsaverage subjects_dir": subjects_dir, "input hash cache": input_hash_cache},
                     allow_ephemeral=allow_ephemeral)
    if force and run_dir.exists():
        run_dir = run_dir.with_name(run_dir.name + "_force" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    if run_dir.exists() and not resume:
        raise FileExistsError(f"{run_dir} exists; use resume (default) to continue it or force to start fresh")
    run_dir.mkdir(parents=True, exist_ok=True)
    ident_path = run_dir / "run_identity.json"
    ident = {"config_hash": config_hash(cfg), "subset_limit": limit}
    if ident_path.exists():
        if json.loads(ident_path.read_text())["identity"] != ident:
            raise IncompatibleCheckpoint(f"{run_dir} belongs to a different configuration")
    else:
        atomic_write_json(ident_path, {"identity": ident, "created_at": datetime.now(timezone.utc).isoformat(),
                                       "git_commit": git_commit(), "config": cfg})   # commit: to resume, pin it

    hashes = InputHashes(input_hash_cache, rehash=rehash_inputs)
    registry, recs, no_meta, no_rec = build_registry(paths, limit, hashes)
    aux = hash_auxiliary_inputs(paths, hashes)
    log.info("input content hashes: %d files read, %d from cache", hashes.n_hashed,
             len(registry) + len(aux) - hashes.n_hashed)
    op = get_operator(cfg, subjects_dir, operator_cache, force)
    table = StatusTable(run_dir / "status.csv")
    table.ensure(registry["key"].tolist())
    save_ts = cfg["output"].get("save_roi_timeseries", False)
    failed = {}
    for rec in registry.itertuples():
        unit_dir = run_dir / "subjects" / rec.key
        sident = subject_identity(cfg, rec, op["operator_id"])
        state, marker = check_unit(unit_dir, "subject", sident)
        if state == "complete":
            if table.get(rec.key) != "completed":
                table.set(rec.key, "completed", "validated existing checkpoint")
            log.info("%s: completed checkpoint found, skipped", rec.key)
            continue
        if state in ("incompatible", "corrupt"):
            why = (f" (changed: {', '.join(identity_diff(marker.get('identity'), sident))})"
                   if state == "incompatible" else "")
            log.warning("%s: %s checkpoint%s, recomputing", rec.key, state, why)
            supersede(unit_dir, run_dir)
        table.set(rec.key, "running")
        try:
            arrays, windows, subject_row, roi_ts = process_subject(rec, cfg, op)
        except Exception as exc:  # recorded, never silent; KeyboardInterrupt etc. propagate
            log.exception("%s failed", rec.key)
            row = {k: getattr(rec, k) for k in ("key", "dataset", "subject_id", "label", "y", "age", "sex",
                                                 "mmse", "acer", "comment", "path", "size_bytes")}
            atomic_write_json(unit_dir / "failed.json", {**row, "status": f"error: {exc}"})
            table.set(rec.key, "failed", str(exc)[:500])
            failed[rec.key] = f"error: {exc}"
            continue
        save_subject(unit_dir, sident, arrays, windows, subject_row, roi_ts if save_ts else None)
        table.set(rec.key, "completed")
    return assemble(run_dir, registry, recs, cfg, op, no_meta, no_rec, limit, failed, aux)
