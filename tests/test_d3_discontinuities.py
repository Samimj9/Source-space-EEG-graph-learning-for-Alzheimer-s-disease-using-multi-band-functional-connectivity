"""Dataset 3 discontinuity handling (EEGLAB 'boundary' events; methodology log entry 45), synthetic data only.

* real EEGLAB .set files (written with scipy) read by MNE: boundary position, zero / removed-length durations;
* no accepted window crosses a discontinuity or its margin: boundary inside a window, exactly at a window edge,
  multiple boundaries; short segments give no windows (and never a failure);
* per-segment resampling: changing one segment leaves every window of another segment bit-identical, and the
  resampled signal near segment ends matches the true band-limited signal;
* the frozen D1/D2 configuration does not split.
"""

import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io

from adgraph.config import load_yaml
from adgraph.preprocessing import discontinuities as disc

mne = pytest.importorskip("mne")
REPO = Path(__file__).resolve().parents[1]
CFG = load_yaml(REPO / "configs" / "pipeline_d3.yaml")
CH = CFG["channels"]["recorded"]
SF = 500
MARGIN = 412
WIN = 5000
Rec = namedtuple("Rec", "key dataset subject_id label y age sex mmse acer comment path size_bytes")


def write_set(path, data_uv, events, sf=SF):
    """Minimal EEGLAB .set (data embedded); events = [(type, latency 1-based, duration)]."""
    chanlocs = np.array([(c, np.nan, np.nan, np.nan, "EEG") for c in CH],
                        dtype=[("labels", "O"), ("X", "O"), ("Y", "O"), ("Z", "O"), ("type", "O")])
    ev = np.array(events or [("none", 1.0, 0.0)], dtype=[("type", "O"), ("latency", "O"), ("duration", "O")])
    n = data_uv.shape[1]
    scipy.io.savemat(path, {"EEG": {"setname": "x", "nbchan": len(CH), "trials": 1, "pnts": n, "srate": sf, "xmin": 0.0,
                                    "xmax": (n - 1) / sf, "data": data_uv.astype(np.float32), "chanlocs": chanlocs,
                                    "event": ev, "icawinv": np.array([]), "epoch": np.array([]), "ref": "common"}},
                     appendmat=False)


def signal(dur, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SF)) / SF
    return (np.sin(2 * np.pi * 10 * t) + rng.standard_normal((len(CH), t.size))) * 10.0      # uV


def make_raw(dur, cut_s=(), seed=0, data=None):
    """RawArray as MNE reads EEGLAB: boundary annotations at (k - 0.5) / sf with removed-length durations."""
    x = (signal(dur, seed) if data is None else data) * 1e-6
    raw = mne.io.RawArray(x, mne.create_info(CH, SF, "eeg"), verbose="error")
    k = [int(round(c * SF)) for c in cut_s]
    raw.set_annotations(mne.Annotations([(i - 0.5) / SF for i in k], [7 / SF if j % 2 else 0.0 for j in range(len(k))],
                                        ["boundary"] * len(k)))
    return raw


def run_subject(monkeypatch, raw, cfg=CFG):
    import adgraph.io.eeg as eeg
    import adgraph.pipeline as pl

    monkeypatch.setattr(eeg, "read_header", lambda path, fmt: raw.copy())
    W = np.random.default_rng(5).standard_normal((68, len(CH)))
    rec = Rec("D3-9", "D3", 9, "AD", 1, 70.0, "F", 20.0, np.nan, "", "x.set", 1)
    return pl.process_subject(rec, cfg, {"W": W, "ch_names": CH, "roi_names": [f"r{i}" for i in range(68)]})


def no_crossing(windows, row):
    cuts = json.loads(row["discontinuity_samples"])
    assert not disc.windows_crossing(windows["start_sample"], windows["stop_sample"], cuts, MARGIN).any()
    assert ((windows["start_sample"] >= windows["segment_start_sample"] + MARGIN)
            & (windows["stop_sample"] <= windows["segment_stop_sample"] - MARGIN)).all()
    assert ((windows["stop_sample"] - windows["start_sample"]) == WIN).all()
    for _, g in windows.groupby("segment_index"):                  # 50 % overlap grid inside each segment
        assert (np.diff(np.sort(g["start_sample"])) % 2500 == 0).all()


def test_mne_reads_eeglab_boundaries_from_a_real_set_file(tmp_path):
    from adgraph.io.eeg import read_header

    events = [("boundary", 10000.5, 12.0), ("stim", 15000.0, 0.0), ("boundary", 20000.5, 0.0),
              ("boundary", 0.5, 3.0)]                                   # one at the very start (no cut)
    write_set(tmp_path / "s.set", signal(60), events)
    raw = read_header(tmp_path / "s.set", "eeglab")
    assert raw.ch_names == CH and raw.info["sfreq"] == SF and np.abs(raw.get_data()).max() < 1e-3   # uV -> V
    on = dict(zip(raw.annotations.onset * SF, raw.annotations.duration * SF))
    assert on[9999.5] == 12.0 and on[19999.5] == 0.0                   # onset = latency - 1; duration = removed
    cuts, n_events = disc.boundary_cuts(raw)
    assert cuts.tolist() == [10000, 20000] and n_events == 3           # stim ignored; cut before sample k


def test_boundary_inside_windows_and_multiple_boundaries(monkeypatch):
    raw = make_raw(300, cut_s=(95.3, 151.0, 152.5, 230.0))
    arrays, windows, row, _, qc = run_subject(monkeypatch, raw)
    assert row["n_boundary_events"] == 4 and row["n_segments"] == 5 and row["status"] == "ok"
    assert row["n_segments_usable"] == 4                               # only 151.0-152.5 s is too short
    no_crossing(windows, row)
    assert row["n_windows_continuous_grid_crossing_discontinuity"] > 0  # the old grid would have crossed cuts
    assert row["n_windows_excluded_discontinuity"] == row["n_windows_continuous_grid"] - row["n_windows_total"] > 0
    assert row["n_windows_kept"] + row["n_windows_rejected"] == row["n_windows_total"] == len(qc["qc_windows"])
    assert arrays["plv"].shape[0] == len(windows) == row["n_windows_kept"]
    assert np.isfinite(arrays["plv"]).all() and np.isfinite(arrays["psd_log"]).all()


def test_boundary_exactly_at_a_window_edge(monkeypatch):
    # the continuous v4 grid would place a window ending exactly at this cut; it must not survive (margin)
    offset = 10 * 250
    edge = (offset + MARGIN + 3 * 2500 + WIN) / 250                    # end of the 4th continuous-grid window (s)
    raw = make_raw(200, cut_s=(edge,))
    _, windows, row, _, _ = run_subject(monkeypatch, raw)
    no_crossing(windows, row)
    cut = json.loads(row["discontinuity_samples"])[0]
    assert (windows["stop_sample"] <= cut - MARGIN).sum() + (windows["start_sample"] >= cut + MARGIN).sum() == len(windows)
    assert len(windows) > 0


def test_only_short_segments_give_no_windows_not_a_failure(monkeypatch):
    raw = make_raw(150, cut_s=tuple(np.arange(20, 140, 15.0)))         # every segment < 23.3 s
    arrays, windows, row, _, _ = run_subject(monkeypatch, raw)
    assert row["status"] == "below_min_windows" and row["n_segments_usable"] == 0 and len(windows) == 0
    assert arrays["plv"].shape == (0, 4, 68, 68) and arrays["sensor_psd_rel"].shape == (0, 4, 19)


def test_segments_are_independent(monkeypatch):
    cfg = json.loads(json.dumps(CFG))
    cfg["windowing"]["rejection"]["mad_k"] = 1e9                       # keep every window: compare all of them
    base = signal(200, seed=3)
    a = make_raw(200, cut_s=(80.0,), data=base)
    changed = base.copy()
    changed[:, :79 * SF] *= 7.0                                         # alter only the first segment
    changed[:, 10 * SF:79 * SF] += 60.0 * np.sin(np.arange(69 * SF) / SF * 2 * np.pi * 0.3)
    b = make_raw(200, cut_s=(80.0,), data=changed)
    xa, wa, ra, _, _ = run_subject(monkeypatch, a, cfg)
    xb, wb, rb, _, _ = run_subject(monkeypatch, b, cfg)
    seg2a, seg2b = (wa["segment_index"] == 1).to_numpy(), (wb["segment_index"] == 1).to_numpy()
    assert seg2a.sum() > 0 and (wa.loc[seg2a, "start_sample"].to_numpy() == wb.loc[seg2b, "start_sample"].to_numpy()).all()
    # source signals are referenced per sample (instantaneous) and the operator is fixed: identical windows
    for k in ("plv", "ciplv", "psd_abs", "sensor_psd_abs"):
        assert np.array_equal(xa[k][seg2a], xb[k][seg2b]), k


def test_per_segment_resampling_is_accurate_near_segment_ends():
    from adgraph.io.eeg import make_montage

    fr = np.array([3.0, 7.5, 11.0, 19.0, 44.0])                         # band-limited truth (<= 45 Hz)
    t = np.arange(120 * SF) / SF
    x = np.stack([np.sin(2 * np.pi * fr[(i % 5)] * t + i) for i in range(len(CH))]) * 1e-5
    raw = mne.io.RawArray(x, mne.create_info(CH, SF, "eeg"), verbose="error")
    raw.set_montage(make_montage("standard_1020"))
    cut = 47 * SF + 1                                                   # odd: segment 2 starts between 250 Hz samples
    segs = [(0, cut - 1), (cut, cut + disc.exact_length(120 * SF - cut, SF, 250))]   # as plan() trims them
    assert [b - a for a, b in segs] == [23500, 36498]
    out, info = disc.process_segments(raw, segs, CFG, offset_new=0)
    # even lengths resample exactly: output sample k of a segment starting at input sample a is at (a + 2k) / SF
    t_new = np.concatenate([(segs[j][0] + 2 * np.arange(s["proc_stop"] - s["proc_start"])) / SF
                            for j, s in enumerate(info)])
    truth = np.stack([np.sin(2 * np.pi * fr[(i % 5)] * t_new + i) for i in range(len(CH))]) * 1e-5
    err = np.zeros(len(t_new), dtype=bool)
    for s in info:                                                      # region a window can use: >= margin inside
        err[s["proc_start"] + MARGIN:s["proc_stop"] - MARGIN] = True
    rel = np.abs(out.get_data()[:, err] - truth[:, err]).max() / truth.std()
    assert rel < 1e-3, rel


def test_d1d2_configuration_does_not_split():
    assert not disc.enabled(load_yaml(REPO / "configs" / "pipeline.yaml")) and disc.enabled(CFG)
    assert disc.split(10, [0, 3, 3, 10, 12]) == [(0, 3), (3, 10)]
