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
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import common_channels, config_hash
from ..connectivity.plv import band_analytic, band_analytic_fir, check_filters_fit_edge_trim, phase_sync
from ..features.psd import check_relative_range, window_band_powers
from ..graphs.store import PSD_KINDS, WINDOW_COLUMNS, GraphStore, require_complete_cohort, window_id
from ..io.eeg import list_recordings, load_raw, make_montage
from ..io.layout import check_layout, spatial_consistency
from ..io import chanlocs
from ..io.fingerprint import InputHashes, auxiliary_inputs
from ..io.participants import load_participants
from ..preprocessing import discontinuities as disc
from ..preprocessing.boundary import boundary_aware, crop_edges, valid_window_starts
from ..preprocessing.boundary import margins as boundary_margins
from ..preprocessing.boundary import report as boundary_report
from ..preprocessing.clean import FlatChannelError, bandpass_half_length, preprocess
from ..preprocessing.windows import reject_windows, window_samples, window_starts
from ..checkpoint import (CODE_VERSIONS, IncompatibleCheckpoint, StatusTable, atomic_save_npy, atomic_savez,
                         atomic_write_csv, atomic_write_json, check_unit, cleanup_temp_files, code_identity,
                         identity_diff, marker_matches, stable_hash, supersede, write_marker)
from .. import progress
from ..memory import MemoryTracker
from ..provenance import git_commit, git_dirty, versions
from ..storage import check_persistent
from ..source.inverse import build_roi_operator, fetch_fsaverage, load_labels

log = logging.getLogger(__name__)

DATASETS = ("D1", "D2")


def datasets_of(cfg: dict | None) -> tuple[str, ...]:
    """Datasets a pipeline config builds (``datasets``; default D1 and D2, the frozen v4 run)."""
    return tuple((cfg or {}).get("datasets", DATASETS))


def run_prefix(cfg: dict | None) -> str:
    """Directory prefix of a config's graph runs: ``graphs`` (D1/D2) or e.g. ``graphsD3`` (Dataset 3), so a
    run of another dataset family is never selected by ``latest_complete_graph_run`` of this one."""
    return str((cfg or {}).get("run_prefix", "graphs"))
ASSEMBLY_FILES = ("graphs.npz", "windows.csv", "subjects.csv", "manifest.json", "assembly.done.json")

# ROIs band-filtered + Hilbert-transformed together (bounds memory; results identical, see band_analytic)
ANALYTIC_CHUNK_ROWS = 8
BAND_POWER_KINDS = ("abs", "log", "rel")      # from the Welch PSD (features.psd); "de" from the band signal


def differential_entropy(seg: np.ndarray) -> np.ndarray:
    """Differential entropy of each row of a window of band analytic signals (n, T) -> (n,): the Gaussian DE
    0.5 ln(2 pi e var(x_band)) of the band-passed signal x_band = Re(analytic), the definition of the LCMV / Xu
    stores (``sourcegraphs.features``, methodology log entry 50) and of ``graphs.store.PSD_KINDS``."""
    return 0.5 * np.log(2 * np.pi * np.e * np.var(seg.real, axis=1))


class GraphRunIncomplete(RuntimeError):
    """Graph construction finished without a usable cohort (failed subjects or no graphs);
    nothing is assembled and no completion marker is written (methodology log entry 38)."""


def canonical_info(cfg: dict, geometry: dict | None = None):
    """Sensor ``Info`` shared by both datasets: common channels, montage, average-reference projection.

    ``geometry`` (Dataset 3, entry 46): electrode coordinates read from the recordings
    (``io.chanlocs``) instead of the configured template montage."""
    import mne

    info = mne.create_info(common_channels(cfg), cfg["preprocessing"]["sfreq"], "eeg")
    info.set_montage(chanlocs.montage(geometry) if geometry is not None else make_montage(cfg["channels"]["montage"]))
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


def process_recording(raw, W: np.ndarray, cfg: dict, offset: int = 0, segments: list | None = None) -> dict:
    """Windows, rejection and per-window features for one cleaned, continuous recording.

    Filtering (sensor band-pass upstream; per-band ROI filtering here) is applied to
    the continuous signal before segmentation.

    Pipeline v4 (``boundary`` in the config; methodology log entry 41): ``raw`` is the recording
    cropped to the analysed interval; windows tile [margin, n - margin) (``boundary.valid_window_starts``)
    and the analytic signal is the finite complex FIR ``band_analytic_fir``. ``offset`` (samples at
    ``sfreq``) is where the cropped signal starts in the original recording; stored start samples and
    times are relative to the original recording.

    ``segments`` (recordings with discontinuities, Dataset 3, entry 45): ``raw`` is the concatenation of
    independently resampled continuous segments (``discontinuities.process_segments``); windows are placed
    inside each segment with the margin from both of its ends, and stored positions map back to the
    original recording.
    """
    v4 = boundary_aware(cfg)
    sfreq = raw.info["sfreq"]
    wcfg = cfg["windowing"]
    progress.set(stage="windowing + artifact rejection")
    data = raw.get_data()
    win_len, step = window_samples(sfreq, wcfg["window_s"], wcfg["step_s"])
    if v4:
        if offset != int(round(float(wcfg["edge_trim_s"]) * sfreq)):
            raise ValueError("pipeline v4 expects the recording cropped by boundary.crop_edges before preprocessing "
                             f"(offset {offset} samples, expected {int(round(float(wcfg['edge_trim_s']) * sfreq))})")
        margin = boundary_margins(cfg)["margin_samples"]
        if segments is not None:
            starts, seg_idx = disc.window_starts(segments, cfg)
        else:
            starts = valid_window_starts(data.shape[1], cfg, margin)
    else:
        starts = window_starts(data.shape[1], sfreq, wcfg["window_s"], wcfg["step_s"], wcfg["edge_trim_s"])
    keep, thr, ptp = reject_windows(data, starts, win_len, wcfg["rejection"]["mad_k"],
                                    wcfg["rejection"]["max_bad_channel_fraction"])
    kept_idx = np.flatnonzero(keep)
    kept = starts[kept_idx]
    # stored positions (samples at sfreq relative to the original recording)
    stored = disc.to_original(starts, seg_idx, segments) if segments is not None else starts + offset

    progress.set(stage="source reconstruction (ROI signals)")
    roi = W @ data  # (n_rois, n_times): continuous source-space ROI signals
    bands = cfg["connectivity"]["bands"]
    n_w, n_b, n_r, n_c = len(kept), len(bands), roi.shape[0], data.shape[0]
    psd_cfg = cfg["psd"]
    sensor_psd = psd_cfg.get("sensor", True)
    plv = np.zeros((n_w, n_b, n_r, n_r), np.float32)
    ciplv = np.zeros_like(plv)
    ciplv_signed = np.zeros_like(plv)
    psd_de = np.zeros((n_w, n_b, n_r), np.float32)
    sensor_psd_de = np.zeros((n_w, n_b, n_c), np.float32)
    analytic_fn = band_analytic_fir if v4 else band_analytic
    tbw = cfg["connectivity"].get("transition_bandwidth_hz")
    # Band-pass + Hilbert once per band on the continuous ROI signals, then cut windows.
    # One band's analytic signal is in memory at a time (released before the next band).
    for b, (name, (lo, hi)) in enumerate(bands.items()):
        progress.set(stage=f"connectivity {name} ({b + 1}/{n_b} bands, {n_w} windows)")
        analytic = analytic_fn(roi, sfreq, lo, hi, chunk_rows=ANALYTIC_CHUNK_ROWS, transition_bandwidth_hz=tbw)
        for w, s in enumerate(kept):
            seg = analytic[:, s:s + win_len]
            plv[w, b], ciplv[w, b], ciplv_signed[w, b] = phase_sync(seg)
            psd_de[w, b] = differential_entropy(seg)
            del seg                                   # a view: it would keep this band's analytic signal alive
        del analytic
        if sensor_psd:                                # the same band filter on the sensor signals
            analytic = analytic_fn(data, sfreq, lo, hi, chunk_rows=ANALYTIC_CHUNK_ROWS, transition_bandwidth_hz=tbw)
            for w, s in enumerate(kept):
                sensor_psd_de[w, b] = differential_entropy(analytic[:, s:s + win_len])
            del analytic

    out = {"plv": plv, "ciplv": ciplv}
    if cfg["output"].get("save_signed_ciplv", True):
        out["ciplv_signed"] = ciplv_signed
    for prefix, sig, n, de in (("psd", roi, n_r, psd_de), ("sensor_psd", data, n_c, sensor_psd_de)):
        if prefix == "sensor_psd" and not sensor_psd:
            continue
        progress.set(stage=f"PSD {'source ROIs' if prefix == 'psd' else 'sensors'}")
        feats = {k: np.zeros((n_w, n_b, n), np.float32) for k in BAND_POWER_KINDS}
        for w, s in enumerate(kept):
            bp = window_band_powers(sig[:, s:s + win_len], sfreq, bands, psd_cfg)
            for k in BAND_POWER_KINDS:
                feats[k][w] = bp[k]
        feats["de"] = de
        out.update({f"{prefix}_{k}": feats[k] for k in PSD_KINDS})

    thr_exceed = (ptp > thr).mean(axis=1) if len(starts) else np.empty(0)
    # every candidate window, kept or rejected (QC table saved with the subject). Windows within
    # the 0.1-45 Hz filter's half length of the discarded edges can carry slow (< l_freq) leakage
    # from edge transients; they are marked so their amplitude can be compared with the others.
    if segments is not None:                      # by construction no window reaches a segment's margin
        s0 = np.array([segments[j]["proc_start"] for j in seg_idx], dtype=int)
        s1 = np.array([segments[j]["proc_stop"] for j in seg_idx], dtype=int)
        near = (starts < s0 + margin) | (starts + win_len > s1 - margin)
    elif v4:                                      # by construction no window reaches the padded boundary
        near = (starts < margin) | (starts + win_len > data.shape[1] - margin)
    else:
        reach = bandpass_half_length(cfg)
        trim = int(round(wcfg["edge_trim_s"] * sfreq))
        near = (starts < trim + reach) | (starts + win_len > data.shape[1] - trim - reach)
    max_frac = wcfg["rejection"]["max_bad_channel_fraction"]
    qc_windows = pd.DataFrame({
        "window_index": np.arange(len(starts)), "start_sample": stored, "tmin": stored / sfreq,
        "tmax": (stored + win_len) / sfreq, "kept": keep,
        "n_channels_over_threshold": (ptp > thr).sum(axis=1) if len(starts) else np.empty(0, int),
        "frac_channels_over_threshold": thr_exceed,
        "max_ptp_uv": ptp.max(axis=1) * 1e6 if len(starts) else np.empty(0),
        "median_ptp_uv": np.median(ptp, axis=1) * 1e6 if len(starts) else np.empty(0),
        "threshold_uv": thr * 1e6, "within_edge_filter_reach": near,
        "decision": np.where(keep, "kept", f"rejected: > {max_frac:.0%} of channels above threshold"),
    })
    if segments is not None:
        so = np.array([segments[j]["orig_start"] for j in seg_idx], dtype=int)
        se = np.array([segments[j]["orig_stop"] for j in seg_idx], dtype=int)
        qc_windows["segment_index"], qc_windows["segment_start_sample"], qc_windows["segment_stop_sample"] = (
            seg_idx, so, se)
        out.update({"segment_index": seg_idx[kept_idx], "segment_start_sample": so[kept_idx],
                    "segment_stop_sample": se[kept_idx]})
    kept_stored = stored[kept_idx]
    out.update({
        "window_index": kept_idx, "start_sample": kept_stored, "stop_sample": kept_stored + win_len,
        "tmin": kept_stored / sfreq, "tmax": (kept_stored + win_len) / sfreq,
        "qc_frac_channels_over_threshold": thr_exceed[kept_idx] if len(starts) else np.empty(0),
        "qc_max_ptp_uv": ptp[kept_idx].max(axis=1) * 1e6 if n_w else np.empty(0),
        "n_windows_total": len(starts), "n_windows_kept": n_w, "rejection_threshold_uv": thr * 1e6,
        "qc_windows": qc_windows,
        # continuous ROI signals only when they will be saved (otherwise a needless full copy)
        "roi_timeseries": roi.astype(np.float32) if cfg["output"].get("save_roi_timeseries", False) else None,
    })
    return out


# ====================================================================== resumable run
DATA_ARRAY_KEYS = ("plv", "ciplv", "ciplv_signed", "psd_abs", "psd_log", "psd_rel", "psd_de",
                   "sensor_psd_abs", "sensor_psd_log", "sensor_psd_rel", "sensor_psd_de")


def graph_run_dir(out_root: str | Path, cfg: dict, limit: int | None = None) -> Path:
    """Deterministic run directory for a configuration (so an interrupted run can be resumed)."""
    return Path(out_root) / (f"{run_prefix(cfg)}_v{cfg['version']}_{config_hash(cfg)}"
                             + (f"_limit{limit}" if limit else ""))


def design_exclusions(paths: dict, cfg: dict | None = None) -> dict:
    """Recordings excluded by a pre-specified design rule (not by results): Dataset 3 FTD participants."""
    if "D3" not in datasets_of(cfg):
        return {}
    from ..io import bids

    _, excluded = bids.load_participants_tsv(bids.participants_path(paths["dataset3"]["root"]))
    return {"D3": excluded.to_dict("records")}


def build_registry(paths: dict, limit: int | None = None, hashes: InputHashes | None = None, cfg: dict | None = None):
    """Recordings joined to participant metadata (all subjects of the config's datasets used as provided;
    Dataset 3: AD and HC only, FTD excluded by design, ``design_exclusions``).

    With ``hashes``, a ``sha256`` column holds each used recording's content hash."""
    recs, meta = [], []
    for ds in datasets_of(cfg):
        if ds == "D3":
            from ..io import bids

            root = paths["dataset3"]["root"]
            kept, excluded = bids.load_participants_tsv(bids.participants_path(root))
            bids.check_cohort(kept)
            r = bids.list_recordings(root)
            recs.append(r[~r["key"].isin(excluded["key"])])          # FTD recordings are never used
            meta.append(kept)
            continue
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
        shas = []
        with progress.reporter(len(registry), "input SHA-256", unit="files", eta=False) as bar:
            for p in registry["path"]:
                bar.set(_file=Path(p).name)
                shas.append(hashes.sha256(p))
                bar.advance(timed=False)
        registry["sha256"] = shas
    return registry, recs, no_meta, no_rec


def hash_auxiliary_inputs(paths: dict, hashes: InputHashes, cfg: dict | None = None) -> dict:
    """Content hashes of the participant sheets and electrode layout (those that exist)."""
    return {name: {"path": str(p), "sha256": hashes.sha256(p)}
            for name, p in auxiliary_inputs(paths, datasets_of(cfg),
                                            (cfg or {}).get("channels", {}).get("verify_layout", True)).items()
            if Path(p).exists()}


def default_hash_cache(run_dir: Path) -> Path:
    return run_dir.parent / "_input_hashes.json"


# ---------------------------------------------------------------------- shared operator
def operator_identity(cfg: dict, geometry: dict | None = None) -> dict:
    """Everything the sensor->ROI operator depends on (the source-model configuration and, for file
    coordinates, the electrode geometry: one operator per distinct geometry, entry 46)."""
    import mne

    ident = {"kind": "operator", "code_version": CODE_VERSIONS["operator"], "source": cfg["source"],
             "channels": common_channels(cfg), "montage": cfg["channels"]["montage"],
             "reference": cfg["preprocessing"]["reference"], "mne": mne.__version__,
             "code": code_identity("operator")}
    if geometry is not None:
        ident["geometry"] = {k: geometry[k] for k in ("source", "frame", "lookup", "hash", "pos_m")}
    return ident


def subject_geometries(registry: pd.DataFrame, cfg: dict) -> dict:
    """{key: electrode geometry} read from every recording when the configuration uses file coordinates
    (``channels.coordinates.source: eeglab_chanlocs``); {} otherwise. Raises ``CoordinateError`` for a recording
    whose coordinates are invalid or whose frame cannot be established (no template fallback)."""
    if not chanlocs.enabled(cfg):
        return {}
    return {rec.key: chanlocs.geometry_of(rec.path, common_channels(cfg), cfg) for rec in registry.itertuples()}


def get_operator(cfg: dict, subjects_dir: str | Path, cache_dir: str | Path, force: bool = False,
                 geometry: dict | None = None) -> dict:
    """Load the verified operator from the cache if its source-model hash matches; else build it once.

    fsaverage is fetched and the forward/inverse model computed only when no valid cached
    operator exists (or ``force``). ``geometry``: electrode coordinates from the recordings (entry 46)."""
    ident = operator_identity(cfg, geometry)
    geo = ({"geometry": {**chanlocs.provenance(geometry), "channels": geometry["channels"], "pos_m": geometry["pos_m"]}}
           if geometry is not None else {})
    op_id = stable_hash(ident)
    cache_dir = Path(cache_dir)
    name = f"operator_{op_id}"
    state, marker = check_unit(cache_dir, name, ident)
    if state == "complete" and not force:
        with progress.reporter(1, "source operator", unit="step", eta=False) as bar:
            bar.set(_step="loaded from cache (no rebuild)")
            bar.advance(timed=False)
        z = np.load(cache_dir / f"{name}.npz")
        log.info("reusing cached ROI operator %s", op_id)
        return {"W": z["W"], "roi_names": z["roi_names"].tolist(), "ch_names": z["ch_names"].tolist(),
                "fwd_ch_names": z["fwd_ch_names"].tolist(), "n_sources": int(z["n_sources"]),
                "roi_coords": z["roi_coords"] if "roi_coords" in z.files else None,
                "verification_rel_error": float(z["verification_rel_error"]), "operator_id": op_id,
                "from_cache": True, **geo}
    if state in ("corrupt", "incompatible") or (state == "complete" and force):
        supersede(cache_dir / f"{name}.npz", cache_dir)
    with progress.reporter(3, "source operator", unit="steps", eta=False) as bar:
        bar.set(_step="fetching fsaverage (first time only)")
        fetch_fsaverage(subjects_dir)
        bar.advance(timed=False)
        bar.set(_step="forward model + inverse operator -> 68x30 ROI operator")
        info = canonical_info(cfg, geometry)
        op = build_roi_operator(info, cfg, subjects_dir)
        bar.advance(timed=False)
        bar.set(_step="verifying against the full MNE inverse")
        op_err = verify_operator(op, info, cfg, subjects_dir)
        bar.advance(timed=False)
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
    previous = compare_with_cached_operators(cache_dir, ident, op["W"], exclude=name)
    for c in previous:
        log.info("operator %s vs cached %s (same source model, other code version): max |dW| / max |W| = %.1e",
                 op_id, c["operator_id"], c["max_rel_diff"])
    return {**op, "verification_rel_error": op_err, "operator_id": op_id, "from_cache": False,
            "compared_with_cached": previous, **geo}


def compare_with_cached_operators(cache_dir: Path, ident: dict, W: np.ndarray, exclude: str = "") -> list[dict]:
    """Cached operators whose identity equals ``ident`` except for the code fingerprint (i.e. the same
    source model, channels, montage, reference and MNE version), with their max relative difference
    to ``W``. Documents that a rebuilt operator reproduces the earlier one; nothing is reused or changed."""
    out = []
    strip = lambda d: {k: v for k, v in (d or {}).items() if k != "code"}         # noqa: E731
    for mp in sorted(Path(cache_dir).glob("operator_*.done.json")):
        name = mp.name.replace(".done.json", "")
        if name == exclude:
            continue
        try:
            other = json.loads(mp.read_text())
            if strip(other.get("identity")) != strip(json.loads(json.dumps(ident, default=str))):
                continue
            with np.load(Path(cache_dir) / f"{name}.npz") as z:
                Wo = z["W"]
        except (OSError, ValueError, KeyError):
            continue
        if Wo.shape == W.shape:
            out.append({"operator_id": name.replace("operator_", ""),
                        "max_rel_diff": float(np.abs(Wo - W).max() / np.abs(W).max())})
    return out


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


def edge_window_ptp_ratio(qc_windows: pd.DataFrame) -> float:
    """Largest median peak-to-peak among KEPT windows within the band-pass reach of the discarded
    edges, over the median of the other kept windows (NaN if either group is empty). Near 1: no
    edge leakage in the retained data; clearly above 1: inspect those windows."""
    k = qc_windows[qc_windows["kept"]]
    near, far = k.loc[k["within_edge_filter_reach"], "median_ptp_uv"], k.loc[~k["within_edge_filter_reach"],
                                                                              "median_ptp_uv"]
    return float(near.max() / far.median()) if len(near) and len(far) else float("nan")


def process_subject(rec, cfg: dict, op: dict) -> tuple[dict, pd.DataFrame, dict, np.ndarray, dict]:
    """Full per-subject computation: load, preprocess, ROI signals, windows, PSD, connectivity.

    Returns ``(arrays, windows, subject_row, roi_timeseries, qc_tables)``; ``qc_tables`` holds
    ``qc_channels`` (per-channel amplitudes and bad-channel decisions) and ``qc_windows`` (every
    candidate window, kept or rejected, and why), saved with the subject checkpoint."""
    row = {k: getattr(rec, k) for k in ("key", "dataset", "subject_id", "label", "y", "age", "sex",
                                         "mmse", "acer", "comment", "path", "size_bytes")}
    progress.set(stage="loading EDF")
    raw, load_qc = load_raw(rec.path, rec.dataset, cfg)
    geo_rows = {}
    if "geometry" in load_qc:                     # Dataset 3: never an operator built for another geometry
        g = load_qc["geometry"]
        if (op.get("geometry") or {}).get("montage_hash") != g["hash"]:
            raise chanlocs.CoordinateError(f"{rec.key}: electrode geometry {g['hash']} differs from the operator's "
                                           f"{(op.get('geometry') or {}).get('montage_hash')}")
        geo_rows = {**chanlocs.provenance(g), "operator_id": op["operator_id"]}
    split = disc.enabled(cfg)
    if split:                                     # Dataset 3: EEGLAB boundary events (entry 45)
        cuts, n_events = disc.boundary_cuts(raw)
        sf_in = raw.info["sfreq"]
    crop, offset = {}, 0
    if boundary_aware(cfg):                       # v4: discard the edges before any filtering (entry 41)
        crop = crop_edges(raw, cfg)
        offset = int(round(crop["crop_start_s"] * cfg["preprocessing"]["sfreq"]))
    seg_rows = {}
    if split:
        p = disc.plan(raw, int(round(crop["crop_start_s"] * sf_in)), cuts, cfg)
        sf_new = float(cfg["preprocessing"]["sfreq"])
        seg_rows = {"n_boundary_events": n_events, "n_discontinuities_analysed": int(len(p["cuts_cropped"])),
                    "n_segments": len(p["segments"]), "n_segments_usable": len(p["usable"]),
                    "usable_duration_s": sum(b - a for a, b in p["usable"]) / sf_in,
                    "n_windows_continuous_grid": p["continuous_candidates"],
                    "n_windows_continuous_grid_crossing_discontinuity": p["continuous_windows_spanning_cut"],
                    "discontinuity_samples": json.dumps([round(c, 3) for c in disc.cuts_original(cuts, sf_in, sf_new)])}
        if not p["usable"]:                       # no continuous stretch can hold one window
            return _no_window_subject(rec, cfg, op, load_qc, crop, offset, {**seg_rows, **geo_rows})
    progress.set(stage="preprocessing (resample, filter, bad channels, reference)")
    try:
        raw, pp_qc = preprocess(raw, cfg, segments=p["usable"] if split else None, offset_new=offset)
    except FlatChannelError as exc:               # Dataset 3: never silently analysed (entry 44)
        raise FlatChannelError(f"subject {rec.key}: {exc}") from exc
    # QC of the channel-to-electrode mapping only (changes nothing): nearby electrodes must
    # correlate more than distant ones; near 0 would indicate a wrong index mapping
    pos = np.array([raw.info["chs"][i]["loc"][:3] for i in range(len(raw.ch_names))])
    rho = spatial_consistency(raw.get_data(), pos)
    if not rho < -0.1:
        log.warning("%s: distance vs |correlation| rho = %.2f (expected clearly negative); "
                    "check the channel mapping (%s labels)", rec.key, rho, load_qc["channel_naming"])
    if raw.ch_names != list(op["ch_names"]):
        raise RuntimeError(f"channel order {raw.ch_names} differs from operator")
    res = process_recording(raw, op["W"], cfg, offset=offset, segments=pp_qc.get("segments"))
    n = res["n_windows_kept"]
    if split:
        seg_rows["n_windows_excluded_discontinuity"] = max(0, seg_rows["n_windows_continuous_grid"]
                                                           - res["n_windows_total"])
    status = "ok" if n >= cfg["windowing"]["min_windows"] else "below_min_windows"
    subj_qc = {"median_std_uv": pp_qc["median_std_uv"], "n_bad_channels": pp_qc["n_bad_channels"],
               "rejection_threshold_uv": res["rejection_threshold_uv"],
               "n_windows_total": res["n_windows_total"], "n_windows_kept": n}
    subject_row = {**row, **{k: load_qc[k] for k in ("orig_sfreq", "duration_s", "unit_scale")}, **subj_qc,
                   "bad_channels": json.dumps(pp_qc["bad_channels"]),
                   "bad_fraction_warning": pp_qc["bad_fraction_warning"],
                   "unrecognised_channels": ",".join(load_qc["unrecognised_channels"]),
                   "channel_naming": load_qc["channel_naming"], "qc_spatial_rho": rho,
                   "median_robust_std_uv": pp_qc["median_robust_std_uv"],
                   "gross_robust_std_uv": pp_qc["gross_robust_std_uv"],
                   "whole_recording_std_uv": pp_qc["whole_recording_std_uv"],
                   "bad_channels_whole_recording": json.dumps(pp_qc["bad_channels_whole_recording"]),
                   "n_windows_rejected": res["n_windows_total"] - n,
                   "qc_edge_window_ptp_ratio": edge_window_ptp_ratio(res["qc_windows"]),
                   "sample_offset": offset, "processed_n_samples": int(raw.n_times), **crop, **seg_rows,
                   **geo_rows, "status": status}
    if split:                                     # time line of the cropped recording, not of the concatenation
        subject_row["processed_n_samples"] = disc.resampled_length(int(round((crop["crop_stop_s"] - crop["crop_start_s"])
                                                                             * sf_in)), sf_in, sf_new)
        subject_row["processed_concat_n_samples"] = int(raw.n_times)
    if subject_row["qc_edge_window_ptp_ratio"] > 2:
        log.warning("%s: kept windows near the recording edges have %.1fx the peak-to-peak of the others; "
                    "inspect qc_windows.csv", rec.key, subject_row["qc_edge_window_ptp_ratio"])
    arrays = {k: res[k] for k in DATA_ARRAY_KEYS if k in res}
    win_rows = [{
        "window_id": window_id(rec.key, int(res["window_index"][w])), "key": rec.key,
        "dataset": rec.dataset, "subject_id": rec.subject_id, "label": rec.label, "y": rec.y,
        "window_index": int(res["window_index"][w]), "start_sample": int(res["start_sample"][w]),
        "stop_sample": int(res["stop_sample"][w]), "tmin": res["tmin"][w], "tmax": res["tmax"][w],
        "sample_weight": 1.0 / n if n else np.nan,
        "qc_frac_channels_over_threshold": res["qc_frac_channels_over_threshold"][w],
        "qc_max_ptp_uv": res["qc_max_ptp_uv"][w], **{f"subject_{k}": v for k, v in subj_qc.items()},
        **({k: int(res[k][w]) for k in ("segment_index", "segment_start_sample", "segment_stop_sample")}
           if "segment_index" in res else {}),
    } for w in range(n)]
    windows = pd.DataFrame(win_rows, columns=list(WINDOW_COLUMNS) if not win_rows else None)
    log.info("%s: %d/%d windows, %d bad channels, %s channel labels, spatial rho %.2f", rec.key, n,
             res["n_windows_total"], pp_qc["n_bad_channels"], load_qc["channel_naming"], rho)
    qc_tables = {"qc_channels": pp_qc["channels"], "qc_windows": res["qc_windows"]}
    return arrays, windows, subject_row, res["roi_timeseries"], qc_tables


def _no_window_subject(rec, cfg: dict, op: dict, load_qc: dict, crop: dict, offset: int, seg_rows: dict):
    """A recording none of whose continuous segments can hold one window (entry 45): reported with status
    ``below_min_windows`` (the pre-specified rule), nothing analysed; never a silent drop or a failure."""
    n_b, n_r, n_c = len(cfg["connectivity"]["bands"]), len(op["roi_names"]), len(op["ch_names"])
    arrays = {k: np.zeros((0, n_b, n_r, n_r) if k in ("plv", "ciplv", "ciplv_signed")
                          else (0, n_b, n_c if k.startswith("sensor") else n_r), np.float32)
              for k in DATA_ARRAY_KEYS if (k != "ciplv_signed" or cfg["output"].get("save_signed_ciplv", True))}
    row = {k: getattr(rec, k) for k in ("key", "dataset", "subject_id", "label", "y", "age", "sex", "mmse", "acer",
                                         "comment", "path", "size_bytes")}
    subject_row = {**row, **{k: load_qc[k] for k in ("orig_sfreq", "duration_s", "unit_scale")}, **crop, **seg_rows,
                   "n_windows_total": 0, "n_windows_kept": 0, "n_windows_rejected": 0,
                   "n_windows_excluded_discontinuity": seg_rows.get("n_windows_continuous_grid", 0),
                   "sample_offset": offset, "status": "below_min_windows",
                   "note": "no continuous segment long enough for one window plus margins"}
    log.warning("%s: %s", rec.key, subject_row["note"])
    return arrays, pd.DataFrame(columns=list(WINDOW_COLUMNS)), subject_row, None, {}


def save_subject(unit_dir: Path, identity: dict, arrays: dict, windows: pd.DataFrame, subject_row: dict,
                 roi_ts: np.ndarray | None, qc_tables: dict | None = None) -> None:
    """Payload files first (atomic), completion marker last."""
    files = [atomic_savez(unit_dir / "arrays.npz", **arrays), atomic_write_csv(unit_dir / "windows.csv", windows),
             atomic_write_json(unit_dir / "subject.json", subject_row)]
    files += [atomic_write_csv(unit_dir / f"{name}.csv", table) for name, table in (qc_tables or {}).items()]
    if roi_ts is not None:
        files.append(atomic_save_npy(unit_dir / "roi_timeseries.npy", roi_ts.astype(np.float32)))
    (unit_dir / "failed.json").unlink(missing_ok=True)
    write_marker(unit_dir, "subject", identity, files)


def subject_states(run_dir: Path, registry: pd.DataFrame, cfg: dict, operator_id: str | dict) -> pd.DataFrame:
    """Checkpoint state of every subject, read from disk only (no computation). ``operator_id``: one id, or
    {key: id} when operators differ by electrode geometry (Dataset 3)."""
    status = StatusTable(run_dir / "status.csv")
    rows = []
    for rec in registry.itertuples():
        unit_dir = run_dir / "subjects" / rec.key
        expected = subject_identity(cfg, rec, operator_id[rec.key] if isinstance(operator_id, dict) else operator_id)
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
    registry, *_ = build_registry(paths, limit, InputHashes(input_hash_cache or default_hash_cache(run_dir)), cfg)
    cache = Path(operator_cache) if operator_cache else run_dir.parent / "_operator_cache"
    geoms = subject_geometries(registry, cfg)     # Dataset 3: one operator per electrode geometry (entry 46)
    idents = {g["hash"]: operator_identity(cfg, g) for g in geoms.values()} or {None: operator_identity(cfg)}
    ids = {h: stable_hash(i) for h, i in idents.items()}
    states = {h: check_unit(cache, f"operator_{ids[h]}", i)[0] for h, i in idents.items()}
    op_id = ids[None] if None in ids else ",".join(sorted(ids.values()))
    op_state = ",".join(sorted(set(states.values())))
    by_key = {k: ids[g["hash"]] for k, g in geoms.items()} if geoms else op_id
    subj = subject_states(run_dir, registry, cfg, by_key) if run_dir.exists() else \
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
             no_meta: list, no_rec: list, limit: int | None, keys_failed: dict, aux_inputs: dict | None = None,
             exclusions: dict | None = None, all_ops: list | None = None) -> Path:
    """Concatenate completed subject units into the cohort store (skipped if already assembled).

    Refuses (``GraphRunIncomplete``) a cohort with failed subjects or without any graph: an
    assembled run with a completion marker is always complete and non-empty."""
    if keys_failed:
        raise GraphRunIncomplete(f"not assembling: {len(keys_failed)} subject(s) failed: {sorted(keys_failed)}")
    units = {}
    for rec in registry.itertuples():
        mp = run_dir / "subjects" / rec.key / "subject.done.json"
        units[rec.key] = (json.loads(mp.read_text())["files"] if rec.key not in keys_failed and mp.exists()
                          else {"failed": keys_failed.get(rec.key, "missing")})
    ident = {"kind": "graph_assembly", "code_version": CODE_VERSIONS["graph_assembly"],
             "config_hash": config_hash(cfg),
             "operator_id": ",".join(sorted(o["operator_id"] for o in all_ops)) if all_ops else op["operator_id"],
             "units": units,
             "subset_limit": limit, "auxiliary_inputs": {k: v["sha256"] for k, v in (aux_inputs or {}).items()},
             "code": code_identity("graph")}
    state, _ = check_unit(run_dir, "assembly", ident)
    if state == "complete":
        log.info("cohort already assembled for these subject units; nothing to do")
        return run_dir
    if state != "missing":                     # units changed (e.g. a failed subject now succeeded)
        for f in ASSEMBLY_FILES:
            supersede(run_dir / f, run_dir)

    bands = cfg["connectivity"]["bands"]
    n_b, n_r, n_c = len(bands), len(op["roi_names"]), len(op["ch_names"])
    keys = [k for k in DATA_ARRAY_KEYS if (k != "ciplv_signed" or cfg["output"].get("save_signed_ciplv", True))
            and (not k.startswith("sensor") or cfg["psd"].get("sensor", True))]
    # Pass 1 (small files only): subject rows and window tables -> exact cohort size.
    win, subj_rows, ok_units = [], [], []
    for rec in registry.itertuples():
        unit_dir = run_dir / "subjects" / rec.key
        if rec.key in keys_failed:
            subj_rows.append(json.loads((unit_dir / "failed.json").read_text()))
            continue
        row = json.loads((unit_dir / "subject.json").read_text())
        subj_rows.append(row)
        if row["status"] != "ok":
            continue
        win.append(pd.read_csv(unit_dir / "windows.csv"))
        ok_units.append((unit_dir, len(win[-1])))
    # Pass 2: preallocate each cohort array once and copy one subject at a time into its slice
    # (no per-subject lists + concatenate, which held every array twice).
    n_total = sum(n for _, n in ok_units)
    if not n_total:
        raise GraphRunIncomplete(f"not assembling an empty cohort: no subject has graphs "
                                 f"(statuses: {[(r.get('key'), r.get('status')) for r in subj_rows]})")
    tail = {k: (n_b, n_r, n_r) if k in ("plv", "ciplv", "ciplv_signed") else (n_b, n_c if "sensor" in k else n_r)
            for k in keys}
    stacked = {k: np.empty((n_total, *tail[k]), np.float32) for k in keys}
    pos = 0
    with progress.reporter(len(ok_units), "assembly", unit="subjects", eta=False) as bar:
        for unit_dir, n in ok_units:
            bar.set(subject=unit_dir.name, windows=n)
            with np.load(unit_dir / "arrays.npz") as z:
                for k in keys:
                    block = z[k]
                    if block.shape != (n, *tail[k]):
                        raise ValueError(f"{unit_dir}/arrays.npz[{k}] has shape {block.shape}, "
                                         f"expected {(n, *tail[k])}")
                    stacked[k][pos:pos + n] = block
                    del block
            pos += n
            bar.advance(timed=False)
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
                     "subset_limit": limit, **({"excluded_by_design": exclusions} if exclusions else {})},
        "source_operator": {"operator_id": op["operator_id"], "from_cache": op.get("from_cache", False),
                            "compared_with_cached": op.get("compared_with_cached", []),
                            "shape": list(np.asarray(op["W"]).shape), "n_sources": op["n_sources"],
                            "channels": list(op["ch_names"]), "shared_across_datasets": True,
                            "verification_rel_error": op["verification_rel_error"],
                            "roi_names": list(op["roi_names"]), "W": np.asarray(op["W"]).tolist(),
                            **({"geometry": op["geometry"]} if op.get("geometry") else {})},
        **({"source_operators": [{"operator_id": o["operator_id"], "geometry": o.get("geometry"),
                                  "verification_rel_error": o["verification_rel_error"],
                                  "W": np.asarray(o["W"]).tolist()} for o in all_ops]} if all_ops else {}),
        "segmentation": {"window_samples": win_s, "step_samples": step_s},
        "boundary_processing": (boundary_report(cfg) if boundary_aware(cfg) else
                                {"version": "v3", "description": "continuous resampling, filtering and FFT Hilbert "
                                 "transform of the whole recording; windows inside the 10 s edge trim"}),
        "subject_units": {k: v for k, v in units.items()},
        "summary": {
            "n_graphs": int(len(windows)),
            "subjects_ok": subjects.loc[subjects["status"] == "ok", "key"].tolist(),
            "subjects_not_ok": subjects.loc[subjects["status"] != "ok", ["key", "status"]].to_dict("records"),
            "per_dataset": {ds: {"subjects": int((subjects["dataset"].eq(ds) & subjects["status"].eq("ok")).sum()),
                                 "graphs": int(windows["dataset"].eq(ds).sum()) if len(windows) else 0}
                            for ds in datasets_of(cfg)},
        },
        "outputs": ["graphs.npz", "windows.csv", "subjects.csv", "manifest.json", "subjects/<key>/"],
    }
    with progress.reporter(3, "saving cohort", unit="steps", eta=False) as bar:
        bar.set(_step=f"validating {n_total} graphs")
        store = GraphStore(stacked, windows, list(bands), np.array(list(bands.values()), dtype=float),
                           list(op["roi_names"]), list(op["ch_names"]), subjects, manifest,
                           roi_coords=op.get("roi_coords"))
        bar.advance(timed=False)
        bar.set(_step="writing graphs.npz, windows.csv, subjects.csv, manifest.json to Drive")
        store.save(run_dir)
        bar.advance(timed=False)
        bar.set(_step="writing completion marker")
        write_marker(run_dir, "assembly", ident, ["graphs.npz", "windows.csv", "subjects.csv", "manifest.json"])
        bar.advance(timed=False)
    return run_dir


# ---------------------------------------------------------------------- discovery
def list_graph_runs(processed_dir: str | Path, prefix: str = "graphs") -> pd.DataFrame:
    """All graph run directories of one family (``prefix``: ``graphs`` = D1/D2, ``graphsD3`` = Dataset 3)
    with their state (assembled or not, smoke test or full)."""
    rows = []
    for d in sorted(Path(processed_dir).glob(f"{prefix}_v*")):
        if not d.is_dir():
            continue
        man = d / "manifest.json"
        assembled = (d / "assembly.done.json").exists() and man.exists()
        m = json.loads(man.read_text()) if assembled else {}
        rows.append({"run_dir": str(d), "assembled": assembled, "limit_smoke_test": "_limit" in d.name,
                     "forced": "_force" in d.name, "assembled_at": m.get("assembled_at", ""),
                     "graph_run_id": m.get("graph_run_id", ""), "config_hash": m.get("config_hash", ""),
                     "version": int((m.get("config") or {}).get("version", 0) or 0)})
    return pd.DataFrame(rows, columns=["run_dir", "assembled", "limit_smoke_test", "forced", "assembled_at",
                                       "graph_run_id", "config_hash", "version"])


def latest_complete_graph_run(processed_dir: str | Path, prefix: str = "graphs") -> Path:
    """Assembled full (non-smoke-test) graph run of the highest pipeline version, most recent first; raises
    if there is none. Experiments use this run unless --graphs names another. ``prefix`` selects the dataset
    family (``graphs`` = D1/D2, default; ``graphsD3`` = Dataset 3); families are never mixed."""
    runs = list_graph_runs(processed_dir, prefix)
    # the highest pipeline version first (v4 boundary-aware graphs are never mixed with or replaced by an
    # older version), then the most recently assembled
    runs = runs[runs["assembled"] & ~runs["limit_smoke_test"]].sort_values(["version", "assembled_at"])
    if runs.empty:
        raise FileNotFoundError(f"no assembled graph run in {processed_dir}")
    latest = Path(runs["run_dir"].iloc[-1])
    require_complete_cohort(GraphStore.load(latest, keys=()))    # never falls back to an older run
    return latest


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
    if boundary_aware(cfg):                       # v4: windows keep a computed margin from the cropped ends
        m = boundary_margins(cfg)
        log.info("boundary-aware processing (entry 41): edges cropped before filtering; margin %d samples "
                 "(%.2f s, policy %s)", m["margin_samples"], m["margin_s"], m["policy"])
    else:
        check_filters_fit_edge_trim(cfg)
    if paths.get("electrode_layout") and cfg["channels"].get("verify_layout", True):   # D1/D2 index-only labels
        layout = check_layout(paths, cfg["channels"]["recorded"])
        log.info("electrode layout %s (sha256 %s) matches channels.recorded", layout["path"], layout["sha256"][:12])
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
    n_tmp = cleanup_temp_files(run_dir)           # partial writes of a killed session (never part of a unit)
    if n_tmp:
        log.warning("removed %d temporary file(s) left by an interrupted session", n_tmp)
    ident_path = run_dir / "run_identity.json"
    ident = {"config_hash": config_hash(cfg), "subset_limit": limit}
    if ident_path.exists():
        if json.loads(ident_path.read_text())["identity"] != ident:
            raise IncompatibleCheckpoint(f"{run_dir} belongs to a different configuration")
    else:
        atomic_write_json(ident_path, {"identity": ident, "created_at": datetime.now(timezone.utc).isoformat(),
                                       "git_commit": git_commit(), "config": cfg})   # commit: to resume, pin it

    hashes = InputHashes(input_hash_cache, rehash=rehash_inputs)
    registry, recs, no_meta, no_rec = build_registry(paths, limit, hashes, cfg)
    unlabelled = registry.loc[registry["label"].isna(), "key"].tolist()
    if unlabelled:                                # never build graphs without a diagnosis label
        raise ValueError(f"recordings without a participant row (no AD/HC label): {unlabelled}")
    aux = hash_auxiliary_inputs(paths, hashes, cfg)
    log.info("input content hashes: %d files read, %d from cache", hashes.n_hashed,
             len(registry) + len(aux) - hashes.n_hashed)
    mem_log = MemoryLog(run_dir / "memory_log.csv")
    geoms = subject_geometries(registry, cfg)     # Dataset 3: coordinates from each recording (entry 46)
    with MemoryTracker() as m:
        if geoms:                                 # one verified operator per distinct electrode geometry
            uniq = {g["hash"]: g for g in geoms.values()}
            ops = {h: get_operator(cfg, subjects_dir, operator_cache, force, geometry=g) for h, g in uniq.items()}
            op_of = {k: ops[g["hash"]] for k, g in geoms.items()}
            log.info("electrode geometries: %s", {h: sum(g["hash"] == h for g in geoms.values()) for h in uniq})
            if len({(tuple(o["ch_names"]), tuple(o["roi_names"])) for o in ops.values()}) != 1:
                raise RuntimeError("operators of different electrode geometries disagree on channels or ROI order")
        else:
            op = get_operator(cfg, subjects_dir, operator_cache, force)
            ops, op_of = {None: op}, {k: op for k in registry["key"]}
    # the operator of most subjects describes the run; with several geometries all are in the manifest
    op = max(ops.values(), key=lambda o: sum(x is o for x in op_of.values()))
    mem_log.add("operator", ",".join("from cache" if o.get("from_cache") else "built + verified"
                                     for o in ops.values()), m.record())
    table = StatusTable(run_dir / "status.csv")
    table.ensure(registry["key"].tolist())
    save_ts = cfg["output"].get("save_roi_timeseries", False)
    failed = {}
    idents = {rec.key: subject_identity(cfg, rec, op_of[rec.key]["operator_id"]) for rec in registry.itertuples()}
    # checkpoint-aware progress: subjects whose marker already matches are shown as done from the
    # start (full validation still happens below; a corrupt one is then recomputed and counted)
    likely_done = {k for k, i in idents.items() if marker_matches(run_dir / "subjects" / k, "subject", i)}
    with progress.reporter(len(registry), "subjects", initial=len(likely_done), unit="recordings",
                           counts={"done": 0, "resumed": 0, "failed": 0, "pending": len(registry)}) as bar:
        _run_subjects(registry, run_dir, cfg, op_of, idents, likely_done, table, save_ts, failed, mem_log, bar)
    check_run_complete(run_dir, registry, failed)
    with MemoryTracker() as m:
        out = assemble(run_dir, registry, recs, cfg, op, no_meta, no_rec, limit, failed, aux,
                       design_exclusions(paths, cfg), all_ops=list(ops.values()) if len(ops) > 1 else None)
    mem_log.add("assembly", "done", m.record())
    return out


def check_run_complete(run_dir: Path, registry: pd.DataFrame, failed: dict) -> None:
    """Raise ``GraphRunIncomplete`` unless every subject succeeded and at least one has graphs.

    Before raising: ``last_attempt.json`` records the failed subjects and their errors (the
    per-subject ``failed.json``, ``status.csv`` and ``memory_log.csv`` are kept as well), and a
    previously assembled cohort in this directory, which no longer matches the subject units, is
    moved to ``_superseded/`` so the directory never advertises a completed cohort."""
    statuses = {}
    for key in registry["key"]:
        if key in failed:
            statuses[key] = failed[key]
            continue
        row = run_dir / "subjects" / key / "subject.json"
        statuses[key] = json.loads(row.read_text())["status"] if row.exists() else "missing"
    n_ok = sum(v == "ok" for v in statuses.values())
    complete = not failed and n_ok > 0
    reason = ("all subjects processed" if complete else f"{len(failed)} of {len(registry)} subject(s) failed"
              if failed else f"no subject produced graphs ({len(registry)} processed)")
    atomic_write_json(run_dir / "last_attempt.json", {         # replaced on every attempt (diagnostic only)
        "time": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "result": "subjects complete" if complete else "incomplete", "reason": reason,
        "n_subjects": len(registry), "n_ok": n_ok, "failed": failed, "statuses": statuses})
    if complete:
        return
    if (run_dir / "assembly.done.json").exists() or (run_dir / "graphs.npz").exists():
        for f in ASSEMBLY_FILES:
            supersede(run_dir / f, run_dir)
    lines = "\n  ".join(f"{k}: {v}" for k, v in failed.items()) or "(none failed)"
    raise GraphRunIncomplete(
        f"graph construction incomplete: {reason}; the cohort was NOT assembled.\n  {lines}\n"
        f"Details: {run_dir / 'last_attempt.json'}, {run_dir / 'status.csv'}, subjects/<key>/failed.json. "
        "Completed subjects are saved; after fixing the cause, rerun the same command (only failed ones rerun).")


def _run_subjects(registry, run_dir, cfg, op_of, idents, likely_done, table, save_ts, failed, mem_log, bar):
    """Subject loop of ``run``: one subject in memory at a time, each saved as soon as it finishes."""
    def counts(kind):
        bar.counts[kind] += 1
        bar.counts["pending"] -= 1
        bar.set()

    for rec in registry.itertuples():
        unit_dir = run_dir / "subjects" / rec.key
        sident = idents[rec.key]
        bar.set(subject=rec.key, dataset=rec.dataset, stage="checking checkpoint")
        state, marker = check_unit(unit_dir, "subject", sident)
        if state == "complete":
            if table.get(rec.key) != "completed":
                table.set(rec.key, "completed", "validated existing checkpoint")
            log.info("%s: completed checkpoint found, skipped", rec.key)
            if rec.key not in likely_done:
                bar.advance(timed=False)
            counts("resumed")
            continue
        if rec.key in likely_done:                # marker matched but payload invalid: recomputed now
            bar.advance(-1, timed=False)
        if state in ("incompatible", "corrupt"):
            why = (f" (changed: {', '.join(identity_diff(marker.get('identity'), sident))})"
                   if state == "incompatible" else "")
            log.warning("%s: %s checkpoint%s, recomputing", rec.key, state, why)
            supersede(unit_dir, run_dir)
        table.set(rec.key, "running")
        bar.start_item()
        t_start = time.perf_counter()
        try:
            with MemoryTracker() as m:            # one subject in memory at a time; released below
                arrays, windows, subject_row, roi_ts, qc_tables = process_subject(rec, cfg, op_of[rec.key])
        except Exception as exc:  # recorded, never silent; KeyboardInterrupt etc. propagate
            log.exception("%s failed", rec.key)
            row = {k: getattr(rec, k) for k in ("key", "dataset", "subject_id", "label", "y", "age", "sex",
                                                 "mmse", "acer", "comment", "path", "size_bytes")}
            atomic_write_json(unit_dir / "failed.json", {**row, "status": f"error: {exc}"})
            table.set(rec.key, "failed", str(exc)[:500])
            failed[rec.key] = f"error: {exc}"
            mem_log.add(rec.key, "failed", {**m.record(), "seconds": round(time.perf_counter() - t_start, 1)})
            bar.advance(timed=False)
            counts("failed")
            continue
        mem = {**m.record(), "seconds": round(time.perf_counter() - t_start, 1)}
        subject_row.update(mem)                   # runtime + memory QC in subjects.csv
        bar.set(stage="saving checkpoint to Drive")
        save_subject(unit_dir, sident, arrays, windows, subject_row, roi_ts if save_ts else None, qc_tables)
        table.set(rec.key, "completed")
        mem_log.add(rec.key, "completed", mem)
        log.info("%s: %s s, peak memory %s MB (available before: %s MB)", rec.key, mem["seconds"],
                 mem["mem_peak_mb"], mem["mem_available_before_mb"])
        del arrays, windows, subject_row, roi_ts, qc_tables   # release this subject before the next one
        bar.advance()
        counts("done")
    bar.set(subject=None, dataset=None, stage=None)


class MemoryLog:
    """``memory_log.csv`` in the run directory: one row per operator build, subject and assembly,
    appended across sessions (informational QC; never used to decide completeness)."""

    def __init__(self, path: Path):
        self.path = path

    def add(self, unit: str, event: str, mem: dict) -> None:
        old = pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()
        row = pd.DataFrame([{"time": datetime.now(timezone.utc).isoformat(), "unit": unit, "event": event, **mem}])
        atomic_write_csv(self.path, pd.concat([old, row], ignore_index=True))
