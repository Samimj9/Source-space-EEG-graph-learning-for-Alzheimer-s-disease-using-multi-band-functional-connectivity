"""LCMV / Xu suite: five-minute selection never exceeds 300 s, exact boundary validation, and the targeted
correction ``corr1`` of an assembled store (methodology log entry 53). Synthetic headers and arrays only."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from adgraph.checkpoint import write_marker
from adgraph.config import load_yaml
from adgraph.evaluation.splits import xu_partition
from adgraph.lcmvxu import correct as corr
from adgraph.lcmvxu import recording
from adgraph.lcmvxu.validate import boundary_violations, require_validated, validate_run

CFG = load_yaml("configs/lcmv_xu/pipeline.yaml")
SF_IN, SF = 500.0, 250.0
M = recording.margin_samples(CFG)
WIN, STEP = recording.window_len(CFG)
NEED = int(CFG["windowing"]["selection_s"] * SF)


def header(lengths_in):
    """Fake EEGLAB header: consecutive segments of the given input-rate lengths, a boundary between each."""
    edges = np.cumsum(lengths_in)[:-1]
    onset = (edges - 0.5) / SF_IN                         # EEGLAB boundary between two samples
    return SimpleNamespace(info={"sfreq": SF_IN}, n_times=int(sum(lengths_in)), first_time=0.0,
                           annotations=SimpleNamespace(description=["boundary"] * len(edges), onset=onset))


# D3-23-like: the second segment starts at input sample 129800 (259.6 s, whose float product with 250 Hz is not
# exact); 252.999 s + 37.4 s < 300 s with less than one window still needed; a third segment would be over target
FRAGMENTED = [129800, 2 * (9350 + 2 * M), 2 * (5000 + 2 * M)]
CONTIGUOUS = [2 * (NEED + 2 * M + 1000)]


def shapes_of(raw):
    import adgraph.preprocessing.discontinuities as disc

    cuts, n_ev = disc.boundary_cuts(raw)
    table, info = recording.segment_table(raw.n_times, cuts, n_ev, SF_IN, SF)
    return [recording.SegmentShape(t["index"], t["orig_start"], t["n"], SF) for t in table if t["kept"]], info


def test_selection_never_exceeds_target():
    shapes, _ = shapes_of(header(FRAGMENTED))
    old = corr.select_original(shapes, CFG, M)
    new, info = recording.select(shapes, CFG, M)
    assert sum(p.length for p in old) > NEED                      # the defect: an extra 10 s piece
    assert len(old) == 3 and len(old[-1].windows) == 1
    assert sum(p.length for p in new) <= NEED and len(new) == 2
    assert info["selected_duration_s"] <= CFG["windowing"]["selection_s"]
    assert [p.windows for p in new] == [p.windows for p in old[:2]]
    shapes, _ = shapes_of(header(CONTIGUOUS))
    new, info = recording.select(shapes, CFG, M)
    assert info["selection_mode"] == "contiguous" and new[0].length == NEED and len(new[0].windows) == 291


def test_exact_validation_at_the_margin():
    raw = header(FRAGMENTED)
    shapes, info = shapes_of(raw)
    seg = shapes[1]
    assert seg.orig_start == 64900
    subj = pd.DataFrame([{"key": "D3-1", "orig_sfreq": SF_IN, "status": "ok",
                          "boundary_positions_s": json.dumps(info["boundary_positions_s"]),
                          "segments_samples": json.dumps([[s.orig_start, s.orig_start + s.n] for s in shapes])}])
    at = seg.orig_start + M                                       # exactly the margin after the boundary
    win = pd.DataFrame({"key": ["D3-1"], "segment_index": [1], "start_sample": [at], "stop_sample": [at + WIN]})
    assert boundary_violations(win, subj, M, SF)                  # legacy rounded seconds: flagged (false positive)
    subj["boundary_cuts_in"] = json.dumps(info["boundary_cuts_in"])
    subj["segments_in"] = json.dumps(info["segments_in"])
    assert boundary_violations(win, subj, M, SF) == []            # exact: lies exactly at the margin -> safe
    one_in = win.assign(start_sample=at - 1, stop_sample=at - 1 + WIN)
    assert boundary_violations(one_in, subj, M, SF)               # one sample closer: still a violation
    end = seg.orig_start + seg.n - M - WIN
    late = win.assign(start_sample=end + 1, stop_sample=end + 1 + WIN)
    assert boundary_violations(win.assign(start_sample=end, stop_sample=end + WIN), subj, M, SF) == []
    assert boundary_violations(late, subj, M, SF)
    other = win.assign(segment_index=0)                           # a window is checked in its own segment
    assert boundary_violations(other, subj, M, SF)


def make_store(tmp_path):
    """Original (b38685e-style) mode B and mode A stores of 10 participants, D3-1 fragmented."""
    rng = np.random.default_rng(0)
    roi, ch, k = ["a-lh", "b-lh", "a-rh", "b-rh"], ["Fp1", "Fp2", "Cz"], len(CFG["bands"])
    headers, rows, wins = {}, [], []
    for n in range(1, 11):
        key = f"D3-{n}"
        raw = header(FRAGMENTED if n == 1 else CONTIGUOUS)
        headers[f"/data/sub-{n:03d}.set"] = raw
        shapes, info = shapes_of(raw)
        pieces = corr.select_original(shapes, CFG, M)
        cands = [(pi, shapes[p.segment].index, shapes[p.segment].orig_start + s)
                 for pi, p in enumerate(pieces) for s in p.windows]
        unit = tmp_path / "graphsLX_v1_h" / "subjects" / key
        unit.mkdir(parents=True)
        pd.DataFrame([{"window_index": i, "piece": p, "segment_index": j, "start_sample": s, "accepted": True,
                       "max_ptp_uv": 50.0} for i, (p, j, s) in enumerate(cands)]).to_csv(unit / "qc_windows.csv",
                                                                                           index=False)
        for i, (p, j, s) in enumerate(cands):
            wins.append({"window_id": f"{key}_w{i:05d}", "key": key, "dataset": "D3", "subject_id": n,
                         "label": "AD" if n % 2 else "HC", "y": n % 2, "window_index": i, "start_sample": s,
                         "stop_sample": s + WIN, "tmin": s / SF, "tmax": (s + WIN) / SF, "segment_index": j,
                         "piece": p, "xu_tenfold": i < 10, "sample_weight": 1.0 / len(cands)})
        sel = round(sum(p.length for p in pieces) / SF, 3)
        rows.append({"key": key, "dataset": "D3", "subject_id": n, "label": "AD" if n % 2 else "HC", "y": n % 2,
                     "path": f"/data/sub-{n:03d}.set", "orig_sfreq": SF_IN, "status": "ok", "reason": "",
                     "boundary_positions_s": json.dumps(info["boundary_positions_s"]),
                     "segment_durations_s": json.dumps(info["segment_durations_s"]),
                     "segments_samples": json.dumps([[s.orig_start, s.orig_start + s.n] for s in shapes]),
                     "selection_mode": "multi_segment" if len(pieces) > 1 else "contiguous",
                     "n_pieces": len(pieces), "selected_duration_s": sel, "selection_target_s": 300,
                     "n_candidate_windows": len(cands), "n_windows_candidate": len(cands),
                     "n_windows_accepted": len(cands), "n_windows_rejected": 0, "n_windows_kept": len(cands),
                     "n_tenfold": 10, "rejection_reasons": "{}"})
    subjects, windows = pd.DataFrame(rows), pd.DataFrame(wins)
    n = len(windows)
    adj = rng.uniform(0, 1, (n, k, 4, 4)).astype(np.float32)
    adj = (adj + adj.transpose(0, 1, 3, 2)) / 2
    adj[:, :, range(4), range(4)] = 0
    arrays = {a: (adj if a in ("plv", "ciplv") else
                  rng.uniform(0, 1, (n, k, 3 if a.startswith("sensor") else 4)).astype(np.float32))
              for a in corr.ARRAYS}
    dev, test = xu_partition(subjects[["key", "dataset", "y"]])
    split = {"test_fraction": 0.2, "seed": 20250702, "stratified_by": "diagnosis", "development": dev, "test": test,
             "n_development": {"AD": int(subjects[subjects.key.isin(dev)].y.sum()),
                               "HC": int((subjects[subjects.key.isin(dev)].y == 0).sum())},
             "n_test": {"AD": int(subjects[subjects.key.isin(test)].y.sum()),
                        "HC": int((subjects[subjects.key.isin(test)].y == 0).sum())}}
    dirs = []
    for name, mode, mask in (("graphsLX_v1_h", "B_all_windows", np.ones(n, bool)),
                             ("graphsLX10_v1_h", "A_xu_tenfold", windows["xu_tenfold"].to_numpy())):
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        w = windows[mask].reset_index(drop=True)
        w["sample_weight"] = 1.0 / w.groupby("key")["key"].transform("size")
        np.savez(d / "graphs.npz", **{a: v[mask] for a, v in arrays.items()}, bands=np.array(list(CFG["bands"])),
                 band_edges=np.array(list(CFG["bands"].values()), float), roi_names=np.array(roi),
                 channels=np.array(ch))
        w.to_csv(d / "windows.csv", index=False)
        subjects.assign(n_windows_kept=subjects.n_windows_kept).to_csv(d / "subjects.csv", index=False)
        (d / "xu_split.json").write_text(json.dumps(split))
        (d / "manifest.json").write_text(json.dumps({
            "graph_run_id": f"orig_{mode}", "assembled_at": "2026-10-01", "config": CFG, "git_commit": "b38685e",
            "mode": mode, "summary": {"n_graphs": int(mask.sum())}, "registry": {"subset_limit": None},
            "segmentation": {"window_samples": WIN, "step_samples": STEP, "margin_samples": M},
            "source_operator": {"roi_names": roi, "channels": ch}}))
        write_marker(d, "assembly", {"orig": mode}, ["graphs.npz", "windows.csv", "subjects.csv", "manifest.json"])
        dirs.append(d)
    return dirs[0], dirs[1], headers, arrays, windows


@pytest.fixture
def store(tmp_path, monkeypatch):
    run_b, run_a, headers, arrays, windows = make_store(tmp_path)
    monkeypatch.setattr(corr, "read_header", lambda path, fmt="edf": headers[path])
    monkeypatch.setattr(corr, "rejection_check", lambda *a: {"threshold_original_uv": 80.0,
                                                             "threshold_corrected_selection_uv": 79.9,
                                                             "decision_flips": []})
    return run_b, run_a, arrays, windows


def test_correction_removes_only_over_target_windows(store):
    run_b, run_a, arrays, windows = store
    before = {p: p.read_bytes() for d in (run_b, run_a) for p in sorted(d.rglob("*")) if p.is_file()}
    assert not validate_run(run_b)["passed"]                      # original: legacy false positive + > 300 s
    dst_b, dst_a, summary = corr.correct(run_b, run_a)
    assert {p: p.read_bytes() for d in (run_b, run_a) for p in sorted(d.rglob("*")) if p.is_file()} == before
    assert summary["participants_changed"] == ["D3-1"]
    assert [w["window_id"] for w in summary["windows_removed"]] == [windows.loc[
        (windows.key == "D3-1") & (windows.piece == 2), "window_id"].item()]
    assert summary["beamforming_repeated"] is False
    wb = pd.read_csv(dst_b / "windows.csv")
    assert len(wb) == len(windows) - 1 and summary["n_windows_mode_A"]["identical_to_original"]
    keep = windows["window_id"].isin(wb["window_id"]).to_numpy()
    with np.load(dst_b / "graphs.npz") as z:
        for a in corr.ARRAYS:
            assert np.array_equal(z[a], arrays[a][keep])          # same rows, same order, nothing recomputed
    s = pd.read_csv(dst_b / "subjects.csv").set_index("key")
    assert s.loc["D3-1", "selected_duration_s"] <= 300 and s.loc["D3-1", "n_windows_kept"] == keep[
        windows.key.to_numpy() == "D3-1"].sum()
    assert np.allclose(wb.groupby("key")["sample_weight"].sum(), 1.0)
    for d in (dst_b, dst_a):
        res = validate_run(d)
        assert res["passed"], res["checks"]
        (d / "validation.json").write_text(json.dumps(res, default=str))
        require_validated(d)
    assert corr.correct(run_b, run_a)[2]["participants_changed"] == ["D3-1"]     # idempotent, never rewritten


def test_correction_stops_when_a_decision_would_change(store, monkeypatch):
    run_b, run_a, _, _ = store
    monkeypatch.setattr(corr, "rejection_check", lambda *a: {"threshold_original_uv": 80.0,
                                                             "threshold_corrected_selection_uv": 70.0,
                                                             "decision_flips": [[0, 0, 1650]]})
    with pytest.raises(corr.CorrectionError, match="decision"):
        corr.correct(run_b, run_a)
    assert not corr.corrected_dirs(run_b, run_a)[0].exists()


def test_unvalidated_or_failed_store_is_refused(store):
    run_b, _, _, _ = store
    with pytest.raises(SystemExit, match="not been validated"):
        require_validated(run_b)
    (run_b / "validation.json").write_text(json.dumps({**validate_run(run_b), "graph_run_id": "orig_B_all_windows"}))
    with pytest.raises(SystemExit, match="FAILED"):
        require_validated(run_b)
