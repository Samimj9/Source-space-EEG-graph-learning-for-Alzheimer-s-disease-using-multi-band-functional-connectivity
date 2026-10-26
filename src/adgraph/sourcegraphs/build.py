"""Resumable construction of the ds004504 source-graph stores, LCMV or wMNE (methodology log entries 50, 55).

The inverse (``source.method``) and the atlas (``source.atlas``, default the 68 Desikan-Killiany ROIs) come from the
resolved store configuration (``sourcegraphs.config.resolve``). Layout under ``<processed>/<output_subdir>/``
(``D3_LCMV_XU`` or ``D3_WMNE_XU``; store names from ``store_names``, e.g. graphsWX / graphsWX10):

    graphsLX_v1_<hash>/            mode B: every accepted 10 s window (90 % overlap) of the 5-minute selection
        subjects/<key>/            per-participant checkpoint: arrays.npz windows.csv qc_windows.csv subject.json
                                   operator.npz (68 x 19 LCMV ROI operator) roi_timeseries.npz subject.done.json
        graphs.npz windows.csv subjects.csv manifest.json assembly.done.json
        qc_report.csv              one row per participant (durations, boundaries, segments, selection, windows,
                                   rejections, inclusion status and reason)
        xu_split.json              frozen 80/20 participant split + development folds (saved participant IDs)
    graphsLX10_v1_<hash>/          mode A (author reproduction): the first 10 accepted windows of each participant
                                   (10 consecutive 10 s windows, 1 s step: 65 participants -> 650 samples)
    cache/                         forward model per electrode geometry, input hashes

The graph stores follow ``graphs.store`` (band axis = the five Xu bands), so the existing experiment code reads them.
Graph construction and experiments stay separate: the run directory is the only interface.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import progress
from ..checkpoint import (StatusTable, atomic_savez, atomic_write_csv, atomic_write_json, check_unit, code_identity,
                          stable_hash, supersede, write_marker)
from ..config import common_channels, config_hash
from ..graphs.store import GraphStore, window_id
from ..io import chanlocs
from ..io.eeg import load_raw
from ..io.fingerprint import InputHashes
from ..preprocessing.clean import FlatChannelError
from ..provenance import git_commit, git_dirty, versions
from ..source import atlas
from . import recording, wmne
from .lcmv import data_covariance, forward, roi_operator, sensor_info, verify
from .features import window_features

log = logging.getLogger(__name__)
VERSION = 1
METADATA_FIELDS = ("label", "y", "age", "sex", "mmse", "acer", "comment")
ARRAYS = ("plv", "ciplv", "psd_abs", "psd_log", "psd_rel", "psd_de",
          "sensor_psd_abs", "sensor_psd_log", "sensor_psd_rel", "sensor_psd_de")


class GraphRunIncomplete(RuntimeError):
    pass


def root_dir(processed: str | Path, cfg: dict) -> Path:
    return Path(processed) / cfg.get("output_subdir", "D3_LCMV_XU")


def run_dirs(processed: str | Path, cfg: dict, limit: int | None = None) -> tuple[Path, Path]:
    """(mode B, mode A) store directories; names from ``store_names`` (LCMV: graphsLX / graphsLX10, the default;
    wMNE: graphsWX / graphsWX10)."""
    sfx = f"_v{cfg['version']}_{config_hash(cfg)}" + (f"_limit{limit}" if limit else "")
    root = root_dir(processed, cfg)
    b, a = cfg.get("store_names", ["graphsLX", "graphsLX10"])
    return root / f"{b}{sfx}", root / f"{a}{sfx}"


# ---------------------------------------------------------------------------------------------- one participant
def recording_qc(rec, cfg: dict, loader=None) -> tuple[dict, list, list, int, dict]:
    """Load + segments + selection (no source model): (row, segments, pieces, margin, geometry). ``loader``
    (default ``io.eeg.load_raw``) returns (raw, load record with ``geometry``); the cross-dataset D2 stores pass
    their own (``crossdataset.d2store.load_d2``)."""
    raw, load_qc = (loader or load_raw)(rec.path, rec.dataset, cfg)
    segs, seg_info = recording.segments(raw, cfg)
    margin = recording.margin_samples(cfg)
    pieces, sel = recording.select(segs, cfg, margin)
    recording.crossing_check(segs, pieces, cfg, margin)
    row = {k: getattr(rec, k) for k in ("key", "dataset", "subject_id", "label", "y", "age", "sex", "mmse", "path")}
    row.update({"orig_sfreq": float(raw.info["sfreq"]), **seg_info, **sel,
                "segments_samples": json.dumps([[s.orig_start, s.orig_start + s.n] for s in segs]),
                **chanlocs.provenance(load_qc["geometry"]),
                **{k: v for k, v in load_qc.items() if k.startswith("x_")}})
    for k in ("boundary_positions_s", "boundary_cuts_in", "segments_in", "segment_durations_s"):
        row[k] = json.dumps(row[k])
    return row, segs, pieces, margin, load_qc["geometry"]


def reject(segs, pieces, cfg: dict) -> tuple[list, np.ndarray, np.ndarray, float]:
    """The existing adaptive peak-to-peak rule (``preprocessing.windows.reject_windows``) over all candidate
    windows of the participant: threshold = median + mad_k * MAD of (window, channel) peak-to-peak values; a window
    is rejected if more than ``max_bad_channel_fraction`` of its channels exceed it."""
    win, _ = recording.window_len(cfg)
    rc = cfg["windowing"]["rejection"]
    cands = [(pi, s) for pi, p in enumerate(pieces) for s in p.windows]
    if not cands:
        return cands, np.zeros(0, bool), np.zeros((0, 0)), float("nan")
    ptp = np.stack([np.ptp(segs[pieces[pi].segment].data[:, s:s + win], axis=1) for pi, s in cands])
    med = np.median(ptp)
    thr = float(med + rc["mad_k"] * np.median(np.abs(ptp - med)))
    keep = (ptp > thr).mean(axis=1) <= rc["max_bad_channel_fraction"]
    return cands, keep, ptp, thr


def method(cfg: dict) -> str:
    """Inverse solution of this suite: ``lcmv`` (D3_LCMV_XU) or ``wmne`` (D3_WMNE_XU, entry 55)."""
    m = cfg["source"].get("method", "lcmv")
    if m not in ("lcmv", "wmne"):
        raise ValueError(f"source.method must be lcmv or wmne, got {m!r}")
    return m


def process_subject(rec, cfg: dict, ctx: dict, loader=None) -> tuple[dict, pd.DataFrame, dict, pd.DataFrame, dict]:
    """(arrays, windows, subject_row, qc_windows, extra files) for one participant (``loader``: ``recording_qc``)."""
    row, segs, pieces, margin, geom = recording_qc(rec, cfg, loader)
    if geom["hash"] != ctx["geometry"]["hash"]:
        raise chanlocs.CoordinateError(f"{rec.key}: electrode geometry {geom['hash']} differs from the forward model's")
    win, _ = recording.window_len(cfg)
    sf = float(cfg["preprocessing"]["sfreq"])
    if pieces:
        sel = np.concatenate([segs[p.segment].data[:, p.start:p.stop] for p in pieces], axis=1)
        std = sel.std(axis=1)
        flat = [c for c, s in zip(ctx["channels"], std) if s < cfg["units"]["flat_std_v"]]
        if flat:
            raise FlatChannelError(f"subject {rec.key}: flat channel(s) {flat} in the selected data")
        lo_hi = cfg["units"]["plausible_std_v"]
        if not lo_hi[0] <= float(np.median(std)) <= lo_hi[1]:
            raise ValueError(f"{rec.key}: median channel std {np.median(std):.3g} V outside {lo_hi}")
        row["median_std_uv"] = float(np.median(std) * 1e6)
        del sel
    cands, keep, ptp, thr = reject(segs, pieces, cfg)
    n_acc = int(keep.sum())
    row.update({"n_windows_candidate": len(cands), "n_windows_accepted": n_acc,
                "n_windows_rejected": int(len(cands) - n_acc), "rejection_threshold_uv": thr * 1e6,
                "rejection_reasons": json.dumps({"peak_to_peak": int(len(cands) - n_acc)} if len(cands) else {})})
    qc_w = pd.DataFrame([{"window_index": i, "piece": pi, "segment_index": segs[pieces[pi].segment].index,
                          "start_sample": segs[pieces[pi].segment].orig_start + s,
                          "accepted": bool(keep[i]), "max_ptp_uv": float(ptp[i].max() * 1e6),
                          "frac_channels_over_threshold": float((ptp[i] > thr).mean()),
                          "reason": "" if keep[i] else "peak-to-peak: > "
                          f"{cfg['windowing']['rejection']['max_bad_channel_fraction']:.0%} of channels above threshold"}
                         for i, (pi, s) in enumerate(cands)])
    k, r, c = len(cfg["bands"]), len(ctx["roi_names"]), len(ctx["channels"])
    empty = {a: np.zeros((0, k, r, r) if a in ("plv", "ciplv") else (0, k, c if a.startswith("sensor") else r),
                         np.float32) for a in ARRAYS}
    if n_acc < cfg["windowing"]["min_windows"]:
        why = ("no continuous segment long enough for one window plus the filter margins" if not pieces
               else f"{n_acc} accepted windows (< {cfg['windowing']['min_windows']})")
        row.update({"status": "below_min_windows", "reason": why, "n_windows_kept": 0, "n_tenfold": 0})
        return empty, pd.DataFrame(), row, qc_w, {}
    first = next(i for i, ok in enumerate(keep) if ok)
    pi0, s0 = cands[first]
    if method(cfg) == "wmne":
        # wMNE (D3_WMNE_XU, entry 55): the data-independent operator of this geometry, verified on this participant
        W = ctx["wmne"]["W"]
        err = wmne.verify(W, ctx["info"], ctx["wmne"]["inv"], ctx["labels"],
                          segs[pieces[pi0].segment].data[:, s0:s0 + win], cfg)
        if err > 1e-6:
            raise RuntimeError(f"{rec.key}: wMNE ROI operator differs from MNE apply_inverse_raw (rel. err {err:.2e})")
        op = {"rank": None}
    else:
        # LCMV from this participant's own accepted, broadband samples (labels and other participants never used)
        mask = {}
        for (pi, s), ok in zip(cands, keep):
            if ok:
                m = mask.setdefault(pieces[pi].segment, np.zeros(segs[pieces[pi].segment].n, bool))
                m[s:s + win] = True
        samples = np.concatenate([segs[j].data[:, m] for j, m in sorted(mask.items())], axis=1)
        op = roi_operator(ctx["info"], ctx["fwd"], data_covariance(ctx["info"], samples), ctx["labels"], cfg)
        W = op["W"]
        err = verify(W, ctx["info"], op["filters"], ctx["labels"], ctx["fwd"]["src"],
                     segs[pieces[pi0].segment].data[:, s0:s0 + win], cfg)
        if err > 1e-6:
            raise RuntimeError(f"{rec.key}: LCMV ROI operator differs from MNE apply_lcmv_raw (rel. err {err:.2e})")
        del samples
    feats, rows, ts = [], [], {}
    n_ten = int(cfg["windowing"]["tenfold_samples"])
    acc_rank = 0
    for pi, p in enumerate(pieces):
        seg = segs[p.segment]
        lo, hi = p.start - margin, p.stop + margin
        x = seg.data[:, lo:hi]
        roi = W @ x
        idx = [i for i, (q, _) in enumerate(cands) if q == pi and keep[i]]
        if not idx:
            continue
        starts = [cands[i][1] - lo for i in idx]
        feats.append(window_features(roi, x, starts, win, cfg))
        ts[f"piece{pi}"] = roi[:, margin:margin + p.length].astype(np.float32)
        ts[f"piece{pi}_start_sample"] = np.array(seg.orig_start + p.start)
        for i in idx:
            s = seg.orig_start + cands[i][1]
            rows.append({"window_id": window_id(rec.key, i), "key": rec.key, "dataset": rec.dataset,
                         "subject_id": rec.subject_id, "label": rec.label, "y": rec.y, "window_index": i,
                         "start_sample": s, "stop_sample": s + win, "tmin": s / sf, "tmax": (s + win) / sf,
                         "segment_index": seg.index, "piece": pi, "xu_tenfold": acc_rank < n_ten,
                         "qc_max_ptp_uv": float(ptp[i].max() * 1e6),
                         "qc_frac_channels_over_threshold": float((ptp[i] > thr).mean())})
            acc_rank += 1
    arrays = {a: np.concatenate([f[a] for f in feats]) for a in ARRAYS}
    for a, v in arrays.items():
        if not np.isfinite(v).all():
            raise RuntimeError(f"{rec.key}: non-finite {a}")
    windows = pd.DataFrame(rows)
    windows["sample_weight"] = 1.0 / len(windows)
    row.update({"status": "ok", "reason": "", "n_windows_kept": int(len(windows)),
                "n_tenfold": int(windows["xu_tenfold"].sum()),
                **({"lcmv_rank": op["rank"], "lcmv_verification_rel_error": err} if method(cfg) == "lcmv"
                   else {"wmne_verification_rel_error": err})})
    return arrays, windows, row, qc_w, {"W": W, "ts": ts}


def subject_identity(cfg: dict, rec, geometry_hash: str) -> dict:
    return {"kind": "lx_subject", "version": VERSION, "config_hash": config_hash(cfg), "key": rec.key,
            "input": {"sha256": rec.sha256, "size_bytes": int(rec.size_bytes)},
            "metadata": stable_hash({k: getattr(rec, k, None) for k in METADATA_FIELDS}),
            "geometry": geometry_hash, "code": code_identity("source_graph")}


def save_subject(unit: Path, ident: dict, arrays, windows, row, qc_w, extra) -> None:
    unit.mkdir(parents=True, exist_ok=True)
    files = ["arrays.npz", "windows.csv", "qc_windows.csv", "subject.json"]
    atomic_savez(unit / "arrays.npz", **arrays)
    atomic_write_csv(unit / "windows.csv", windows)
    atomic_write_csv(unit / "qc_windows.csv", qc_w)
    atomic_write_json(unit / "subject.json", row)
    if extra.get("W") is not None:
        atomic_savez(unit / "operator.npz", W=extra["W"])
        files.append("operator.npz")
    if extra.get("ts"):
        atomic_savez(unit / "roi_timeseries.npz", **extra["ts"])
        files.append("roi_timeseries.npz")
    write_marker(unit, "subject", ident, files)


# ---------------------------------------------------------------------------------------------- registry / split
def registry(paths: dict, cfg: dict, limit: int | None, hash_cache: Path) -> tuple[pd.DataFrame, dict]:
    from .registry import build_registry, design_exclusions

    reg, _, no_meta, _ = build_registry(paths, limit, InputHashes(hash_cache), cfg)
    if no_meta:
        raise ValueError(f"recordings without participant metadata: {no_meta}")
    return reg, design_exclusions(paths, cfg)


def xu_split_record(subjects: pd.DataFrame, cfg: dict) -> dict:
    from ..evaluation.splits import XU_SPLIT, xu_partition

    sc = cfg["split"]
    if (sc["test_fraction"], sc["seed"], sc["dev_folds"]) != (XU_SPLIT["test_fraction"], XU_SPLIT["seed"],
                                                              XU_SPLIT["dev_folds"]):
        raise ValueError("store configuration split differs from evaluation.splits.XU_SPLIT")
    subj = subjects.loc[subjects["status"] == "ok", ["key", "dataset", "y"]]
    dev, test = xu_partition(subj)
    return {"test_fraction": sc["test_fraction"], "seed": sc["seed"], "stratified_by": "diagnosis",
            "development": dev, "test": test,
            "n_development": {"AD": int(subj[subj.key.isin(dev)].y.sum()), "HC": int((subj[subj.key.isin(dev)].y == 0).sum())},
            "n_test": {"AD": int(subj[subj.key.isin(test)].y.sum()), "HC": int((subj[subj.key.isin(test)].y == 0).sum())}}


# ---------------------------------------------------------------------------------------------- driver
def inspect(paths: dict, cfg: dict, processed: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Recording-level QC of every participant (durations, boundaries, segments, selection, candidate windows)
    without the source model: written to <root>/inspection_<time>/qc_recordings.csv."""
    root = root_dir(processed, cfg)
    reg, _ = registry(paths, cfg, limit, root / "cache" / "input_sha256.json")
    rows = []
    with progress.reporter(len(reg), "recordings", unit="files") as bar:
        for rec in reg.itertuples():
            bar.set(subject=rec.key)
            try:
                row = recording_qc(rec, cfg)[0]
            except Exception as exc:              # reported, never silent
                row = {"key": rec.key, "label": rec.label, "status": f"error: {exc}"}
            rows.append(row)
            bar.advance()
    df = pd.DataFrame(rows)
    out = root / f"inspection_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    atomic_write_csv(out / "qc_recordings.csv", df)
    return df


def run(paths: dict, cfg: dict, processed: str | Path, subjects_dir: str | Path, limit: int | None = None,
        allow_ephemeral: bool = False) -> tuple[Path, Path]:
    from ..storage import check_persistent
    from ..source.inverse import fetch_fsaverage, load_labels

    run_dir, run10 = run_dirs(processed, cfg, limit)
    root = root_dir(processed, cfg)
    check_persistent({"LX run directory": run_dir, "fsaverage": subjects_dir}, allow_ephemeral=allow_ephemeral)
    run_dir.mkdir(parents=True, exist_ok=True)
    ident_file = run_dir / "run_identity.json"
    if not ident_file.exists():
        atomic_write_json(ident_file, {"config_hash": config_hash(cfg), "config": cfg, "git_commit": git_commit(),
                                       "created_at": datetime.now(timezone.utc).isoformat()})
    reg, excluded = registry(paths, cfg, limit, root / "cache" / "input_sha256.json")
    channels = common_channels(cfg)
    geoms = {rec.key: chanlocs.geometry_of(rec.path, channels, cfg) for rec in reg.itertuples()}
    uniq = {g["hash"]: g for g in geoms.values()}
    fetch_fsaverage(subjects_dir)
    labels = load_labels(subjects_dir, cfg["source"]["parcellation"])
    if len(labels) != 68:
        raise RuntimeError(f"expected 68 Desikan-Killiany ROIs, got {len(labels)}")
    # coarser atlas (docs/final/parcellation.md): merged Desikan-Killiany labels, each extracted with mean_flip over
    # its own sources (never an average of 68-ROI signals); dk68 keeps the labels unchanged
    labels = atlas.merged_labels(labels, cfg["source"].get("atlas", "dk68"))
    if len(labels) != atlas.N_REGIONS[cfg["source"].get("atlas", "dk68")]:
        raise RuntimeError(f"atlas {cfg['source'].get('atlas', 'dk68')}: unexpected number of regions {len(labels)}")
    ctxs = {}
    for h, g in uniq.items():
        info = sensor_info(cfg, g)
        ctxs[h] = {"geometry": g, "info": info, "labels": labels, "roi_names": [lab.name for lab in labels],
                   "channels": channels}
        if method(cfg) == "wmne":                  # one data-independent operator per geometry (entry 55)
            ctxs[h]["wmne"] = wmne.roi_operator(info, cfg, subjects_dir, root / "cache", h, labels)
        else:
            ctxs[h]["fwd"] = forward(info, cfg, subjects_dir, root / "cache", h)
    table = StatusTable(run_dir / "status.csv")
    table.ensure(reg["key"].tolist())
    failed = {}
    idents = {rec.key: subject_identity(cfg, rec, geoms[rec.key]["hash"]) for rec in reg.itertuples()}
    done0 = sum(check_unit(run_dir / "subjects" / k, "subject", i)[0] == "complete" for k, i in idents.items())
    with progress.reporter(len(reg), "participants", initial=done0, unit="participants") as bar:
        for rec in reg.itertuples():
            unit = run_dir / "subjects" / rec.key
            state, _ = check_unit(unit, "subject", idents[rec.key])
            if state == "complete":
                table.set(rec.key, "completed", "validated existing checkpoint")
                continue
            if state in ("incompatible", "corrupt"):
                supersede(unit, run_dir)
            bar.set(subject=rec.key)
            table.set(rec.key, "running")
            t0 = time.perf_counter()
            try:
                out = process_subject(rec, cfg, ctxs[geoms[rec.key]["hash"]])
            except Exception as exc:
                log.exception("%s failed", rec.key)
                failed[rec.key] = f"error: {exc}"
                atomic_write_json(unit / "failed.json", {"key": rec.key, "status": f"error: {exc}"})
                table.set(rec.key, "failed", str(exc)[:500])
                bar.advance()
                continue
            out[2]["seconds"] = round(time.perf_counter() - t0, 1)
            save_subject(unit, idents[rec.key], *out)
            table.set(rec.key, "completed")
            bar.advance()
    qc = pd.DataFrame([json.loads((run_dir / "subjects" / k / "subject.json").read_text())
                       if (run_dir / "subjects" / k / "subject.json").exists()
                       else {"key": k, "status": failed.get(k, "missing")} for k in reg["key"]])
    atomic_write_csv(run_dir / "qc_report.csv", qc)
    if failed:
        atomic_write_json(run_dir / "last_attempt.json", {"failed": failed, "time": datetime.now(timezone.utc).isoformat()})
        raise GraphRunIncomplete(f"{len(failed)} participant(s) failed (not assembled): {failed}. Completed "
                                 f"participants are saved; see {run_dir / 'qc_report.csv'}")
    return assemble(run_dir, run10, reg, cfg, idents, excluded, limit, ctxs)


def _load_units(run_dir: Path, reg: pd.DataFrame):
    rows, wins = [], []
    for k in reg["key"]:
        u = run_dir / "subjects" / k
        rows.append(json.loads((u / "subject.json").read_text()))
        if (u / "windows.csv").stat().st_size > 2:
            w = pd.read_csv(u / "windows.csv")
            if len(w):
                wins.append(w)
    return pd.DataFrame(rows), (pd.concat(wins, ignore_index=True) if wins else pd.DataFrame())


def assemble(run_dir: Path, run10: Path, reg: pd.DataFrame, cfg: dict, idents: dict, excluded: dict,
             limit: int | None, ctxs: dict, split: bool = True, extra_manifest: dict | None = None,
             kind: str = "lx_assembly") -> tuple[Path, Path]:
    """Assemble mode B and mode A. ``split`` False (the cross-dataset D2 stores): no ds004504 participant split is
    computed or saved; ``extra_manifest`` is added to the manifest."""
    units = {k: json.loads((run_dir / "subjects" / k / "subject.done.json").read_text())["files"] for k in reg["key"]}
    ident = {"kind": kind, "version": VERSION, "config_hash": config_hash(cfg), "units": units,
             "subset_limit": limit, "code": code_identity("source_graph")}
    subjects, windows = _load_units(run_dir, reg)
    if not len(windows):
        raise GraphRunIncomplete("no participant produced windows")
    ctx = next(iter(ctxs.values()))
    bands = list(cfg["bands"])
    split = xu_split_record(subjects, cfg) if split else None
    for target, subset in ((run_dir, None), (run10, "xu_tenfold")):
        a_ident = {**ident, "mode": subset or "all"}
        state, _ = check_unit(target, "assembly", a_ident)
        if state == "complete":
            continue
        sel = windows if subset is None else windows[windows["xu_tenfold"].astype(bool)]
        sel = sel.reset_index(drop=True)
        n = len(sel)
        pos = {k: 0 for k in ARRAYS}
        stacked = {}
        with progress.reporter(len(reg), f"assembly ({subset or 'all windows'})", unit="participants") as bar:
            for k in reg["key"]:
                u = run_dir / "subjects" / k
                with np.load(u / "arrays.npz") as z:
                    wk = windows[windows["key"] == k]
                    take = (np.ones(len(wk), bool) if subset is None else wk["xu_tenfold"].astype(bool).to_numpy())
                    for a in ARRAYS:
                        blk = z[a][take]
                        if a not in stacked:
                            stacked[a] = np.empty((n, *blk.shape[1:]), np.float32)
                        stacked[a][pos[a]:pos[a] + len(blk)] = blk
                        pos[a] += len(blk)
                bar.advance()
        sel["sample_weight"] = 1.0 / sel.groupby("key")["key"].transform("size")
        manifest = {
            "graph_run_id": stable_hash(a_ident), "assembled_at": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash(cfg), "config": cfg, "git_commit": git_commit(), "git_dirty": git_dirty(),
            "code": ident["code"], "versions": versions(), "suite": cfg.get("output_subdir", "D3_LCMV_XU"),
            "mode": "B_all_windows" if subset is None else "A_xu_tenfold",
            "inputs": reg[["key", "path", "size_bytes", "sha256"]].to_dict("records"),
            "registry": {"subset_limit": limit, "excluded_by_design": excluded},
            "subject_units": units,
            "source_operator": {"method": method(cfg), "per_participant": method(cfg) == "lcmv",
                                "roi_names": ctx["roi_names"],
                                "channels": ctx["channels"], "shape": [len(ctx["roi_names"]), len(ctx["channels"])],
                                "geometry": {k: v for k, v in chanlocs.provenance(ctx["geometry"]).items()},
                                method(cfg): cfg["source"][method(cfg)], "orientation": cfg["source"]["orientation"],
                                "roi_mode": cfg["source"]["roi_mode"],
                                "verification_rel_error_max": float(subjects[f"{method(cfg)}_verification_rel_error"]
                                                                    .max())
                                if f"{method(cfg)}_verification_rel_error" in subjects else None},
            "segmentation": {"window_samples": int(cfg["windowing"]["window_s"] * cfg["preprocessing"]["sfreq"]),
                             "step_samples": int(cfg["windowing"]["step_s"] * cfg["preprocessing"]["sfreq"]),
                             "margin_samples": recording.margin_samples(cfg)},
            "xu_split": split,
            **(extra_manifest or {}),
            "summary": {"n_graphs": int(n), "subjects_ok": subjects.loc[subjects.status == "ok", "key"].tolist(),
                        "subjects_not_ok": subjects.loc[subjects.status != "ok", ["key", "status", "reason"]]
                        .to_dict("records")},
        }
        store = GraphStore(stacked, sel, bands, np.array(list(cfg["bands"].values()), float), ctx["roi_names"],
                           ctx["channels"], subjects, manifest)
        store.save(target)
        if split is not None:
            atomic_write_json(target / "xu_split.json", split)
        atomic_write_csv(target / "qc_report.csv", subjects)
        write_marker(target, "assembly", a_ident, ["graphs.npz", "windows.csv", "subjects.csv", "manifest.json"])
        del stacked, store
    return run_dir, run10
