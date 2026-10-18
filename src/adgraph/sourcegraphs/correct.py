"""Targeted correction ``corr1`` of an assembled LCMV / Xu run (methodology log entry 53, docs/lcmv_xu.md §11).

The build at code b38685e had two defects, neither affecting the signal processing of any window:

1. **Over-target selection.** In the multi-segment five-minute selection, a final piece of one full window was
   added when less than one window was still needed (``max(need - got, win)``), so two participants got more than
   300 s (D3-23 305.62 s, D3-58 309.62 s; one extra 10 s piece with one window each). ``recording.select`` now
   stops instead. The windows outside the corrected selection are REMOVED; nothing else changes.
2. **Inexact validation metadata.** The boundary cuts were stored as seconds rounded to 1 ms and the segment
   starts on the 250 Hz grid; the validator converted them back in floating point, so a window lying EXACTLY at
   the filter margin (D3-23 [65725, 68225): 1650 input samples = 2 x 825 after its segment start, no cut nearer)
   was flagged. The exact integer positions (``boundary_cuts_in``, ``segments_in``) are added and validated
   exactly (``validate.boundary_violations_exact``; same strict conditions).

``correct`` builds NEW stores ``<run>_corr1`` (mode B) and ``<run10>_corr1`` (mode A) from the original ones; the
originals, their per-participant checkpoints and their code pin are never modified. No beamformer, connectivity or
feature is recomputed: the kept windows' arrays are copied unchanged (row subset, identical order, all aligned
arrays and metadata together). Every step is verified, and nothing is written when a check fails:

* from the EEGLAB header (boundary events, length, rate; no signal): the stored boundary positions, segment
  durations and segment table are reproduced exactly, and the original selection reproduces every stored
  candidate window (piece, segment, start) exactly;
* the corrected selection is a subset of the original candidates;
* for each CHANGED participant only, the signal is loaded and segmented again (no source model) to reproduce the
  stored peak-to-peak values and accept/reject decisions, and the rejection rule is re-applied to the corrected
  candidate set: it must not change the decision of any kept window (the adaptive threshold depends on the
  candidate set); otherwise the correction stops and asks for a decision;
* the frozen participant split, the mode A windows and their arrays are unchanged (bit-identical) unless a
  removed window was one of a participant's first ten.

Known, documented consequence (not recomputed, by decision): the LCMV data covariance of a changed participant was
estimated in the original build from its accepted windows INCLUDING the removed one (one extra clean 10 s window of
the same participant; no label is involved). ``lcmv_covariance_n_windows`` records this.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..checkpoint import atomic_write_csv, atomic_write_json, check_unit, code_identity, stable_hash, write_marker
from ..graphs.store import GraphStore
from ..io.eeg import load_raw, read_header
from ..preprocessing import discontinuities as disc
from ..provenance import git_commit, git_dirty, versions
from . import recording
from .build import ARRAYS, reject, xu_split_record

NAME = "corr1"
REASON = ("selection exceeded the 300 s target (extra final piece of one window when < 1 window was still "
          "needed); boundary metadata stored in rounded seconds made the exact-margin check imprecise")


class CorrectionError(RuntimeError):
    pass


def corrected_dirs(run_dir: Path, run10: Path, name: str = NAME) -> tuple[Path, Path]:
    return run_dir.with_name(f"{run_dir.name}_{name}"), run10.with_name(f"{run10.name}_{name}")


def select_original(segs, cfg: dict, margin: int) -> list[recording.Piece]:
    """The selection exactly as built at b38685e (kept only to verify the stored windows)."""
    win, step = recording.window_len(cfg)
    need = int(round(cfg["windowing"]["selection_s"] * segs[0].sfreq)) if segs else 0
    interiors = [(i, margin, s.n - margin) for i, s in enumerate(segs) if s.n - 2 * margin >= win]
    long_ = [t for t in interiors if t[2] - t[1] >= need]
    if long_:
        i, a, _ = long_[0]
        pieces = [recording.Piece(i, a, a + need)]
    else:
        pieces, got = [], 0
        for i, a, b in interiors:
            if got >= need:
                break
            length = min(b - a, max(need - got, win))
            pieces.append(recording.Piece(i, a, a + length))
            got += length
    for p in pieces:
        p.windows = list(range(p.start, p.stop - win + 1, step))
    return pieces


def _candidates(pieces, segs) -> list[tuple[int, int, int]]:
    """(piece, segment index, start_sample at 250 Hz from the recording start) of every candidate window."""
    return [(pi, segs[p.segment].index, segs[p.segment].orig_start + s) for pi, p in enumerate(pieces) for s in p.windows]


def _set(row: pd.Series, values: dict) -> None:
    for k, v in values.items():                  # (Series.update would ignore new columns)
        row[k] = v


def _read_qc(path: Path) -> pd.DataFrame:
    cols = ["window_index", "piece", "segment_index", "start_sample", "accepted", "max_ptp_uv"]
    return pd.read_csv(path) if path.stat().st_size > 2 else pd.DataFrame(columns=cols)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise CorrectionError(msg)


def participant_plan(row: pd.Series, qc_w: pd.DataFrame, cfg: dict, margin: int) -> dict:
    """Header-only reproduction + corrected selection of one participant (no signal)."""
    sf = float(cfg["preprocessing"]["sfreq"])
    raw = read_header(row["path"], cfg.get("input", {}).get("format", "edf"))
    sf_in = float(raw.info["sfreq"])
    _check(sf_in == float(row["orig_sfreq"]), f"{row['key']}: rate {sf_in} != stored {row['orig_sfreq']}")
    cuts, n_events = disc.boundary_cuts(raw)
    table, info = recording.segment_table(raw.n_times, cuts, n_events, sf_in, sf)
    for k in ("boundary_positions_s", "segment_durations_s"):
        _check(info[k] == json.loads(row[k]), f"{row['key']}: {k} not reproduced from the EEGLAB header")
    kept = [t for t in table if t["kept"]]
    _check([[t["orig_start"], t["orig_start"] + t["n"]] for t in kept] == json.loads(row["segments_samples"]),
           f"{row['key']}: segment table not reproduced")
    shapes = [recording.SegmentShape(t["index"], t["orig_start"], t["n"], sf) for t in kept]
    old = select_original(shapes, cfg, margin)
    old_c = _candidates(old, shapes)
    stored = list(zip(qc_w["piece"].astype(int), qc_w["segment_index"].astype(int), qc_w["start_sample"].astype(int)))
    _check(old_c == stored, f"{row['key']}: original selection does not reproduce the stored candidate windows")
    new, sel = recording.select(shapes, cfg, margin)
    new_c = _candidates(new, shapes)
    _check(set(new_c) <= set(old_c), f"{row['key']}: corrected selection is not a subset of the original")
    return {"info": info, "old": old, "new": new, "sel": sel, "old_c": old_c, "new_c": new_c,
            "changed": new_c != old_c}


def rejection_check(row: pd.Series, qc_w: pd.DataFrame, plan: dict, cfg: dict) -> dict:
    """Changed participants only: reload + segment the signal (no source model), reproduce the stored
    peak-to-peak values and decisions, re-apply the rule to the corrected candidates."""
    raw, _ = load_raw(row["path"], row["dataset"], cfg)
    segs, _ = recording.segments(raw, cfg)
    del raw
    c0, k0, ptp0, t0 = reject(segs, plan["old"], cfg)
    _check(len(c0) == len(qc_w) and np.array_equal(k0, qc_w["accepted"].astype(bool).to_numpy()),
           f"{row['key']}: stored accept/reject decisions not reproduced")
    _check(np.allclose(ptp0.max(axis=1) * 1e6, qc_w["max_ptp_uv"].to_numpy(), rtol=1e-6, atol=1e-9),
           f"{row['key']}: stored peak-to-peak values not reproduced (signal differs)")
    c1, k1, _, t1 = reject(segs, plan["new"], cfg)
    stored = dict(zip(plan["old_c"], k0))
    flips = [c for c, k in zip(_candidates(plan["new"], segs), k1) if bool(k) != bool(stored[c])]
    return {"threshold_original_uv": t0 * 1e6, "threshold_corrected_selection_uv": t1 * 1e6,
            "decision_flips": [list(map(int, c)) for c in flips]}


def correct(run_dir: Path, run10: Path, name: str = NAME) -> tuple[Path, Path, dict]:
    run_dir, run10 = Path(run_dir), Path(run10)
    dst_b, dst_a = corrected_dirs(run_dir, run10, name)
    for src in (run_dir, run10):
        _check((src / "assembly.done.json").exists(), f"{src} is not an assembled store")
    man_b = json.loads((run_dir / "manifest.json").read_text())
    man_a = json.loads((run10 / "manifest.json").read_text())
    cfg = man_b["config"]                                      # the frozen configuration of the original build
    _check(man_a["config"] == cfg, "mode A and mode B stores were built with different configurations")
    margin = recording.margin_samples(cfg)
    _check(margin == man_b["segmentation"]["margin_samples"], "margin differs from the original build")
    idents = {d: {"kind": "lx_correction", "name": name, "source_graph_run_id": m["graph_run_id"], "mode": m["mode"]}
              for d, m in ((dst_b, man_b), (dst_a, man_a))}
    states = {d: check_unit(d, "assembly", i)[0] for d, i in idents.items()}
    if all(s == "complete" for s in states.values()):
        return dst_b, dst_a, json.loads((dst_b / "correction.json").read_text())
    for d, s in states.items():
        _check(s == "complete" or not d.exists() or not any(d.iterdir()),
               f"{d} exists and is not this correction (never overwritten); move it to _superseded/ first")

    subjects = pd.read_csv(run_dir / "subjects.csv")
    windows = pd.read_csv(run_dir / "windows.csv")
    keep = np.ones(len(windows), bool)
    rows, report, dropped = [], [], []
    n_ten = int(cfg["windowing"]["tenfold_samples"])
    for _, row in subjects.iterrows():
        row = row.copy()
        key = row["key"]
        qc_w = _read_qc(run_dir / "subjects" / key / "qc_windows.csv")
        plan = participant_plan(row, qc_w, cfg, margin)
        row["boundary_cuts_in"] = json.dumps(plan["info"]["boundary_cuts_in"])
        row["segments_in"] = json.dumps(plan["info"]["segments_in"])
        row["lcmv_covariance_n_windows"] = row.get("n_windows_accepted")
        row["lcmv_covariance_selected_s"] = row.get("selected_duration_s")
        row["correction"] = ""
        rec = {"key": key, "changed": plan["changed"], "selected_s_original": row.get("selected_duration_s"),
               "selected_s_corrected": plan["sel"]["selected_duration_s"],
               "candidates_original": len(plan["old_c"]), "candidates_corrected": len(plan["new_c"])}
        if plan["changed"]:
            _check(row["status"] == "ok", f"{key}: selection changes for a participant with status {row['status']}")
            rc = rejection_check(row, qc_w, plan, cfg)
            _check(not rc["decision_flips"], f"{key}: re-applying the rejection rule to the corrected selection "
                   f"changes the decision of windows {rc['decision_flips']}; a decision is needed (nothing written)")
            rec.update(rc)
            acc = dict(zip(plan["old_c"], qc_w["accepted"].astype(bool)))
            n_acc = int(sum(acc[c] for c in plan["new_c"]))
            _set(row, {k: plan["sel"][k] for k in ("selection_mode", "n_pieces", "selected_duration_s",
                                                   "n_candidate_windows")})
            _set(row, {"n_windows_candidate": len(plan["new_c"]), "n_windows_accepted": n_acc,
                        "n_windows_rejected": len(plan["new_c"]) - n_acc,
                        "rejection_reasons": json.dumps({"peak_to_peak": len(plan["new_c"]) - n_acc}),
                        "rejection_threshold_corrected_selection_uv": rc["threshold_corrected_selection_uv"],
                        "correction": f"{name}: windows outside the 300 s selection removed (entry 53)"})
            _check(n_acc >= int(cfg["windowing"]["min_windows"]), f"{key}: fewer than the minimum windows remain")
        idx = np.flatnonzero(windows["key"].to_numpy() == key)
        if len(idx):
            new = set(plan["new_c"])
            w = windows.iloc[idx]
            ok = np.array([(int(p), int(j), int(s)) in new for p, j, s in
                           zip(w["piece"], w["segment_index"], w["start_sample"])])
            keep[idx] = ok
            for _, d in w[~ok].iterrows():
                dropped.append({"key": key, "window_id": d["window_id"], "start_sample": int(d["start_sample"]),
                                "stop_sample": int(d["stop_sample"]), "segment_index": int(d["segment_index"]),
                                "piece": int(d["piece"]), "xu_tenfold": bool(d["xu_tenfold"])})
            row["n_windows_kept"] = int(ok.sum())
            row["n_tenfold"] = int(min(n_ten, ok.sum()))
        rec.update({"windows_original": int(len(idx)), "windows_corrected": int(keep[idx].sum())})
        report.append(rec)
        rows.append(row)
    subjects_new = pd.DataFrame(rows)
    win_b = windows[keep].reset_index(drop=True)
    rank = win_b.groupby("key").cumcount()
    tenfold_new = (rank < n_ten).to_numpy()
    tenfold_changed = sorted(set(win_b.loc[tenfold_new != win_b["xu_tenfold"].astype(bool).to_numpy(), "key"]))
    win_b["xu_tenfold"] = tenfold_new
    win_b["sample_weight"] = 1.0 / win_b.groupby("key")["key"].transform("size")
    split = xu_split_record(subjects_new, cfg)
    _check(split == json.loads((run_dir / "xu_split.json").read_text()), "the frozen participant split would change")
    summary = {"name": name, "reason": REASON, "methodology_log_entry": 53,
               "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
               "correction_code": code_identity("source_graph"),
               "source": {"mode_B": {"run_dir": str(run_dir), "graph_run_id": man_b["graph_run_id"],
                                     "git_commit": man_b["git_commit"], "n_windows": int(len(windows))},
                          "mode_A": {"run_dir": str(run10), "graph_run_id": man_a["graph_run_id"],
                                     "git_commit": man_a["git_commit"]}},
               "beamforming_repeated": False, "connectivity_or_features_recomputed": False,
               "participants_changed": [r["key"] for r in report if r["changed"]],
               "windows_removed": dropped, "n_windows_mode_B": {"original": int(len(windows)), "corrected": int(len(win_b))},
               "tenfold_changed_participants": tenfold_changed,
               "participants": report}

    # ---- mode B: subset every aligned array with the same mask (original order kept)
    stacked = {}
    with np.load(run_dir / "graphs.npz") as z:
        meta = {k: z[k] for k in ("bands", "band_edges", "roi_names", "channels")}
        for a in ARRAYS:
            v = z[a]
            _check(len(v) == len(windows), f"{a}: {len(v)} rows for {len(windows)} windows")
            stacked[a] = v[keep]
            del v
    a_idx = np.flatnonzero(tenfold_new)
    stacked_a = {a: stacked[a][a_idx] for a in ARRAYS}
    win_a = win_b.iloc[a_idx].reset_index(drop=True)
    win_a["sample_weight"] = 1.0 / win_a.groupby("key")["key"].transform("size")
    # mode A must equal the original mode A store wherever its windows are unchanged
    old_a = pd.read_csv(run10 / "windows.csv")
    same = old_a["window_id"].isin(win_a["window_id"]).to_numpy()
    common = win_a["window_id"].isin(old_a["window_id"]).to_numpy()
    _check((old_a["window_id"][same].tolist() == win_a["window_id"][common].tolist()), "mode A window order differs")
    with np.load(run10 / "graphs.npz") as z:
        for a in ARRAYS:
            _check(np.array_equal(z[a][same], stacked_a[a][common]), f"mode A {a} differs from the original store")
    summary["n_windows_mode_A"] = {"original": int(len(old_a)), "corrected": int(len(win_a)),
                                   "identical_to_original": bool(same.all() and common.all())}
    bands, edges = meta["bands"].tolist(), meta["band_edges"]
    for dst, man, arr, win in ((dst_b, man_b, stacked, win_b), (dst_a, man_a, stacked_a, win_a)):
        if states[dst] == "complete":
            continue
        m = copy.deepcopy(man)
        m.update({"graph_run_id": stable_hash(idents[dst]), "assembled_at": datetime.now(timezone.utc).isoformat(),
                  "git_commit": git_commit(), "git_dirty": git_dirty(), "versions_at_correction": versions(),
                  "correction": {k: v for k, v in summary.items() if k != "participants"}})
        m["summary"] = {**man["summary"], "n_graphs": int(len(win))}
        store = GraphStore(arr, win, bands, edges, meta["roi_names"].tolist(), meta["channels"].tolist(),
                           subjects_new, m)
        store.save(dst)
        atomic_write_json(dst / "xu_split.json", split)
        atomic_write_csv(dst / "qc_report.csv", subjects_new)
        atomic_write_json(dst / "correction.json", summary)
        atomic_write_csv(dst / "correction_participants.csv", pd.DataFrame(report))
        atomic_write_csv(dst / "correction_removed_windows.csv",
                         pd.DataFrame(dropped, columns=["key", "window_id", "start_sample", "stop_sample",
                                                        "segment_index", "piece", "xu_tenfold"]))
        write_marker(dst, "assembly", idents[dst], ["graphs.npz", "windows.csv", "subjects.csv", "manifest.json",
                                                    "xu_split.json", "correction.json"])
        del store
    return dst_b, dst_a, summary
