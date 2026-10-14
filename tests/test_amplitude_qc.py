"""Amplitude QC on the analysed interval, unit safeguard, bad channels and window rejection
(methodology log entry 39). Synthetic signals only."""

import json
from pathlib import Path

import numpy as np
import pytest

from adgraph.config import common_channels, load_yaml
from adgraph.preprocessing.clean import bandpass_half_length, detect_bad_channels, preprocess

mne = pytest.importorskip("mne")

# These tests document the amplitude / bad-channel / rejection QC of entry 39 on UNCROPPED recordings,
# i.e. the pipeline-v3 path (analysed interval = edge-trimmed part of the processed recording). The QC
# logic is shared with v4, where the recording is cropped first; v4 is covered by test_boundary_processing.
CFG = load_yaml("configs/pipeline.yaml")
CFG.pop("boundary", None)
CFG["version"] = 3
CH = common_channels(CFG)
SF, T = 500, 150
W = np.random.default_rng(1).standard_normal((68, 30))


def _signal(seed=0, uv=10.0):
    """30 ~``uv`` µV channels (volts) like scalp EEG: four broadband sources, one with a 10 Hz
    rhythm, mixed with smooth positive weights (neighbouring channels correlate), plus 30 %
    independent noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(SF * T) / SF
    src = rng.standard_normal((4, t.size))
    src[0] += 3 * np.sin(2 * np.pi * 10 * t)
    mix = rng.uniform(0.5, 1.0, (30, 4))
    x = mix @ src + 0.3 * np.sqrt((mix ** 2).sum(1, keepdims=True)) * rng.standard_normal((30, t.size))
    return x / x.std() * uv * 1e-6


def _raw(x):
    from adgraph.io.eeg import make_montage

    raw = mne.io.RawArray(x.copy(), mne.create_info(CH, SF, "eeg"), verbose="error")
    raw.set_montage(make_montage("standard_1020"), match_case=False)
    return raw


def _edge_transients(x, start_mv=3.0, end_mv=3.0):
    """Large decaying transient in the first 3 s and a ramp in the last 4 s (both inside the 10 s
    edge trim), different on every channel, like amplifier settling / electrode movement."""
    t = np.arange(x.shape[1]) / SF
    g = np.random.default_rng(9).uniform(0.5, 1.5, (30, 1))
    y = x + g * start_mv * 1e-3 * np.exp(-t / 1.0) * (t < 3)
    return y + g[::-1] * end_mv * 1e-3 * np.clip((t - (T - 4)) / 4, 0, 1)


def _former_statistic(x):
    """The check before log entry 39: median std of the whole 0.1-45 Hz recording."""
    from adgraph.preprocessing.clean import resample_and_bandpass

    return float(np.median(resample_and_bandpass(_raw(x), CFG).get_data().std(axis=1)))


def test_edge_transients_no_longer_fail_the_check_and_do_not_change_it():
    x = _signal()
    y = _edge_transients(x)
    assert _former_statistic(y) > CFG["units"]["plausible_std_v"][1]     # the reported failure mode
    _, qa = preprocess(_raw(x), CFG)
    _, qb = preprocess(_raw(y), CFG)                                     # passes now
    # passes, far below the former statistic; it is larger than without transients because the slow
    # (< 0.1 Hz) band-pass leakage into the first/last ~6.5 s of the analysed interval is windowed
    # signal, and is measured rather than hidden (those windows are then rejected, test below)
    assert qa["median_std_uv"] < qb["median_std_uv"] < 0.05 * qb["whole_recording_std_uv"]
    assert qb["whole_recording_std_uv"] > 200 and qb["analysed_interval_s"] == [10.0, T - 10.0]
    # the MAD statistic barely notices transients that reach a fraction of the samples (unit errors are 10^3)
    assert 0.5 < qb["gross_robust_std_uv"] / qa["gross_robust_std_uv"] < 2


@pytest.mark.parametrize("factor", [1e-6, 1e-3, 1e3, 1e6])
def test_unit_safeguard_catches_scale_errors(factor):
    with pytest.raises(ValueError, match="unit safeguard"):
        preprocess(_raw(_signal() * factor), CFG)


def test_unit_safeguard_is_independent_of_edges():
    """A 10^3 error stays caught even if the edges were quiet, and huge edges alone do not trip it."""
    with pytest.raises(ValueError, match="unit safeguard"):
        preprocess(_raw(_signal() * 1e3), CFG)
    _, qc = preprocess(_raw(_edge_transients(_signal(), 50.0, 50.0)), CFG)       # 50 mV edge transients
    assert qc["gross_robust_std_uv"] < 100


def test_noisy_analysed_interval_still_fails():
    """Genuinely high amplitude where windows are cut is not accepted."""
    x = _signal(uv=10.0)
    x[:, 10 * SF:(T - 10) * SF] += np.random.default_rng(5).standard_normal((30, (T - 20) * SF)) * 500e-6
    with pytest.raises(ValueError, match="plausible range"):
        preprocess(_raw(x), CFG)


def test_bad_channels_come_from_the_analysed_interval():
    x = _signal()
    rng = np.random.default_rng(3)
    t = np.arange(x.shape[1]) / SF
    x[5] += rng.standard_normal(t.size) * 400e-6 * ((t > 40) & (t < 120))     # noisy where analysed
    x[20] += 5e-3 * np.exp(-t / 1.0) * (t < 3) * rng.standard_normal(t.size)   # noisy only in the edge
    raw, qc = preprocess(_raw(x), CFG)
    assert set(qc["bad_channels"]) == {CH[5]}
    assert CH[20] in qc["bad_channels_whole_recording"]                 # what the former rule would flag
    ch = qc["channels"].set_index("channel")
    assert ch.loc[CH[5], "interpolated"] and not ch.loc[CH[20], "interpolated"]
    assert ch.loc[CH[5], "bad_reason"].startswith("amplitude") and ch.loc[CH[20], "bad_reason"] == ""
    assert {"std_uv", "robust_std_uv", "log_std_z", "neighbor_corr_p90", "final_std_uv"} <= set(ch.columns)


def test_detect_bad_channels_rules_unchanged():
    """Same flags and order as the implementation before log entry 39 (rules are not changed)."""
    rules = CFG["preprocessing"]["bad_channels"]

    def former(data, names):
        std = np.std(data, axis=1)
        bads = {names[i]: "flat" for i in np.flatnonzero(std < rules["flat_std_v"])}
        ok = std >= rules["flat_std_v"]
        ls = np.log(std[ok])
        mad = np.median(np.abs(ls - np.median(ls))) * 1.4826
        if mad > 0:
            for i, zi in zip(np.flatnonzero(ok), (ls - np.median(ls)) / mad):
                if abs(zi) > rules["robust_z_log_std"]:
                    bads.setdefault(names[i], f"amplitude z={zi:.1f}")
        idx = np.flatnonzero(ok)
        corr = np.abs(np.corrcoef(data[idx]))
        np.fill_diagonal(corr, np.nan)
        for i, c in zip(idx, np.nanpercentile(corr, 90, axis=1)):
            if c < rules["min_neighbor_corr"]:
                bads.setdefault(names[i], f"low correlation {c:.2f}")
        return bads

    for seed in range(5):
        rng = np.random.default_rng(seed)
        common = rng.standard_normal(4000)
        data = (common + 0.4 * rng.standard_normal((30, 4000))) * 1e-5
        data[rng.integers(30)] = 0.0
        data[rng.integers(30)] *= 300
        data[rng.integers(30)] = rng.standard_normal(4000) * 1e-5
        names = [f"c{i}" for i in range(30)]
        assert list(detect_bad_channels(data, names, rules).items()) == list(former(data, names).items())


def test_window_rejection_is_active_and_reported():
    from adgraph.pipeline import edge_window_ptp_ratio, process_recording

    x = _signal()
    t = np.arange(x.shape[1]) / SF
    burst = (t > 70) & (t < 72)                                          # movement artifact, shared by all channels
    x[:, burst] += np.random.default_rng(4).uniform(0.5, 1.5, (30, 1)) * 300e-6 * np.sin(2 * np.pi * 2 * t[burst])
    raw, _ = preprocess(_raw(x), CFG)
    res = process_recording(raw, W, CFG)
    w = res["qc_windows"]
    assert len(w) == res["n_windows_total"] == 12 and w["kept"].sum() == res["n_windows_kept"]
    hit = (w["tmin"] < 72) & (w["tmax"] > 70)
    assert not w.loc[hit, "kept"].any() and w.loc[~hit, "kept"].all()  # the burst windows are rejected
    assert w.loc[hit, "decision"].str.startswith("rejected").all()
    reach = bandpass_half_length(CFG) / CFG["preprocessing"]["sfreq"]
    assert reach == pytest.approx(16.5, abs=0.01)                        # 0.1 Hz high-pass: 33 s FIR
    near = (w["tmin"] < 10 + reach) | (w["tmax"] > T - 10 - reach)
    assert (w["within_edge_filter_reach"] == near).all() and near.sum() == 4
    assert 0.7 < edge_window_ptp_ratio(w) < 1.5


def test_edge_transients_do_not_contaminate_retained_windows():
    """Leakage through the 33 s band-pass reaches into the first/last windows (below 0.1 Hz). The
    unchanged adaptive rejection removes the windows it makes abnormal; PSD features of the windows
    retained in both cases are unchanged, and the sensor signal in the middle is untouched.

    PLV/ciPLV change slightly in every window (here <= 0.0025 for 3 mV transients, ~300x the EEG),
    decaying with distance from the edges: the analytic signal is computed over the whole
    continuous band-passed ROI signal and the Hilbert kernel decays as 1/t (existing connectivity
    design, methodology log entry 39; not changed here)."""
    from adgraph.pipeline import edge_window_ptp_ratio, process_recording

    x = _signal()
    out = []
    for y in (x, _edge_transients(x)):
        raw, qc = preprocess(_raw(y), CFG)
        out.append((process_recording(raw, W, CFG), qc, raw.get_data()))
    (a, qa, da), (b, qb, db) = out
    assert qa["bad_channels"] == qb["bad_channels"] == {}
    wb = b["qc_windows"]
    assert a["qc_windows"]["kept"].all()
    rejected = wb.loc[~wb["kept"], "window_index"].tolist()
    assert rejected == [0, 11] and wb.loc[~wb["kept"], "within_edge_filter_reach"].all()
    both = np.isin(a["window_index"], b["window_index"])
    for k in ("psd_log", "psd_rel", "sensor_psd_log", "sensor_psd_rel"):
        assert np.abs(a[k][both] - b[k]).max() < 1e-4, k
    for k in ("plv", "ciplv"):
        d = np.abs(a[k][both] - b[k]).reshape(both.sum(), -1).max(1)
        assert d.max() < 5e-3 and d[len(d) // 2] < 0.25 * d.max(), k   # small, largest near the edges
    leak = np.abs(db - da)
    win = int(CFG["windowing"]["window_s"] * CFG["preprocessing"]["sfreq"])
    first, middle = a["start_sample"][0], a["start_sample"][len(a["start_sample"]) // 2]
    assert leak[:, middle:middle + win].max() < 1e-9                     # middle windows untouched
    assert leak[:, first:first + win].max() > 1e-6                       # leakage exists at the edge ...
    assert edge_window_ptp_ratio(wb) < 1.5                              # ... and no kept window carries it


def test_qc_tables_are_saved_with_the_subject(tmp_path, monkeypatch):
    pytest.importorskip("edfio")
    import adgraph.pipeline as pl
    from test_end_to_end import _write_edf

    import pandas as pd

    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir(), d2.mkdir()
    _write_edf(d1 / "sub1.edf", CFG["channels"]["recorded"], "V")
    _write_edf(d2 / "2_842019_1.edf", CFG["channels"]["recorded"], "uV")
    pd.DataFrame({"ID": ["Sub1"], "Age": [72], "Gender": ["M"], "Group": ["AD"], "MMSE": [21]}).to_excel(
        tmp_path / "p1.xlsx", index=False)
    pd.DataFrame({"Id": ["2_842019"], "Age": [71], "Gender": ["M"], "Group": ["HC"], "MMSE (/30)": [29]}).to_excel(
        tmp_path / "p2.xlsx", index=False)
    paths = {"dataset1": {"eeg_dir": str(d1), "participants": str(tmp_path / "p1.xlsx")},
             "dataset2": {"eeg_dir": str(d2), "participants": str(tmp_path / "p2.xlsx")}}
    monkeypatch.setattr(pl, "fetch_fsaverage", lambda d: None)
    monkeypatch.setattr(pl, "build_roi_operator", lambda info, cfg, d: {
        "W": W, "roi_names": [f"roi{i:02d}" for i in range(68)], "ch_names": list(info["ch_names"]),
        "n_sources": 20484, "fwd_ch_names": list(info["ch_names"]), "roi_coords": np.zeros((68, 3))})
    monkeypatch.setattr(pl, "verify_operator", lambda *a, **k: 0.0)
    run_dir = pl.run(paths, CFG, tmp_path / "out", tmp_path / "sd")
    unit = run_dir / "subjects" / "D1-1"
    marker = json.loads((unit / "subject.done.json").read_text())
    assert {"qc_channels.csv", "qc_windows.csv"} <= {Path(f).name for f in marker["files"]}
    ch, win = pd.read_csv(unit / "qc_channels.csv"), pd.read_csv(unit / "qc_windows.csv")
    assert len(ch) == 30 and ch.loc[ch["channel"] == "F3", "bad_reason"].str.startswith("amplitude").all()
    assert win["kept"].sum() == len(pd.read_csv(unit / "windows.csv"))
    subj = pd.read_csv(run_dir / "subjects.csv").set_index("key")
    for col in ("median_std_uv", "median_robust_std_uv", "gross_robust_std_uv", "whole_recording_std_uv",
                "bad_channels_whole_recording", "n_windows_rejected", "qc_edge_window_ptp_ratio"):
        assert col in subj.columns

    import subprocess
    import sys

    r = subprocess.run([sys.executable, "scripts/inspect_graph_subjects.py", "--run-dir", str(run_dir)],
                       cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "D1-1" in r.stdout and "D2-2" in r.stdout and "windows (all candidates)" in r.stdout
    (folder,) = run_dir.glob("inspection_*")
    assert {p.name for p in folder.iterdir()} == {"D1-1.png", "D2-2.png", "summary.json"}
