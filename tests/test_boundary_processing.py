"""Boundary-aware processing, pipeline v4 (methodology log entry 41). Synthetic data only."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from adgraph.config import common_channels, load_yaml
from adgraph.connectivity.plv import (analytic_kernel, analytic_kernel_report, band_analytic, band_analytic_fir,
                                      band_filter_kernels, phase_sync)
from adgraph.preprocessing.boundary import (crop_edges, expected_valid_windows, margins, report,
                                            valid_window_starts)

mne = pytest.importorskip("mne")

CFG = load_yaml("configs/pipeline.yaml")
CH = common_channels(CFG)
SF = float(CFG["preprocessing"]["sfreq"])
TBW = CFG["connectivity"]["transition_bandwidth_hz"]
REPO = Path(__file__).resolve().parents[1]


def _eeg(sf, seconds, seed=0, uv=10.0):
    """30 correlated channels (volts) with a 10 Hz rhythm."""
    rng = np.random.default_rng(seed)
    n = int(sf * seconds)
    t = np.arange(n) / sf
    src = rng.standard_normal((4, n))
    src[0] += 3 * np.sin(2 * np.pi * 10 * t)
    mix = rng.uniform(0.5, 1.0, (30, 4))
    x = mix @ src + 0.3 * rng.standard_normal((30, n))
    return x / x.std() * uv * 1e-6


def _raw(x, sf):
    from adgraph.io.eeg import make_montage

    raw = mne.io.RawArray(x.copy(), mne.create_info(CH, sf, "eeg"), verbose="error")
    raw.set_montage(make_montage("standard_1020"), match_case=False)
    return raw


def _process(x, sf, W, cfg=CFG):
    from adgraph.legacy.pipeline import process_recording
    from adgraph.preprocessing.clean import preprocess

    raw = _raw(x, sf)
    crop = crop_edges(raw, cfg)
    offset = int(round(crop["crop_start_s"] * cfg["preprocessing"]["sfreq"]))
    raw, qc = preprocess(raw, cfg)
    return process_recording(raw, W, cfg, offset=offset), qc, crop, raw.n_times


# ------------------------------------------------------------------ the analytic FIR
def test_config_is_v4_boundary_aware():
    assert CFG["version"] == 4 and CFG["boundary"] == {"crop_before_filtering": True, "filter_pad": "reflect_limited",
                                                       "analytic": "fir", "margin_policy": "full_support"}


@pytest.mark.parametrize("band", list(CFG["connectivity"]["bands"]))
def test_analytic_kernel_properties(band):
    lo, hi = CFG["connectivity"]["bands"][band]
    g = analytic_kernel(SF, float(lo), float(hi), TBW)
    h = band_filter_kernels(CFG)[band]
    assert len(g) == len(h) == 825 and np.array_equal(np.real(g), h)          # real part = production band-pass
    r = analytic_kernel_report(SF, float(lo), float(hi), TBW)
    assert r["hilbert_kernel_energy_outside_support"] < 1e-6 and r["negative_frequency_leakage_db"] < -70
    assert np.allclose(np.imag(g), -np.imag(g)[::-1], atol=1e-12)             # Hilbert of a symmetric kernel is odd


def test_interior_equivalence_with_fft_hilbert():
    """Away from the ends the finite analytic FIR reproduces filter + FFT Hilbert (PLV to < 1e-4)."""
    x = np.random.default_rng(0).standard_normal((6, int(SF * 300)))
    for band, (lo, hi) in CFG["connectivity"]["bands"].items():
        a = band_analytic_fir(x, SF, lo, hi, transition_bandwidth_hz=TBW)
        b = band_analytic(x, SF, lo, hi, transition_bandwidth_hz=TBW)
        w = slice(int(SF * 140), int(SF * 160))
        assert np.abs(phase_sync(a[:, w])[0] - phase_sync(b[:, w])[0]).max() < 1e-4, band
        assert np.abs(phase_sync(a[:, w])[1] - phase_sync(b[:, w])[1]).max() < 1e-4, band


def test_analytic_fir_is_local():
    """Output sample t depends only on x[t - c .. t + c]; changing anything farther away changes nothing."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 20000))
    c = len(analytic_kernel(SF, 4.0, 8.0, TBW)) // 2
    y = x.copy()
    y[:, :5000] = rng.standard_normal((3, 5000)) * 1e3                       # huge change at the start
    a, b = (band_analytic_fir(v, SF, 4.0, 8.0, transition_bandwidth_hz=TBW) for v in (x, y))
    assert np.allclose(a[:, 5000 + c:], b[:, 5000 + c:], rtol=0, atol=1e-9)
    assert not np.allclose(a[:, 5000 + c - 5:5000 + c], b[:, 5000 + c - 5:5000 + c])
    fft_a, fft_b = (band_analytic(v, SF, 4.0, 8.0, transition_bandwidth_hz=TBW) for v in (x, y))
    assert np.abs(fft_a[:, -2000:] - fft_b[:, -2000:]).max() > 1e-3          # the FFT Hilbert is not local


# ------------------------------------------------------------------ margins and window grid
def test_margins_from_the_actual_filters():
    m = margins(CFG)
    assert m["sensor_filter_half_samples"] == 4125 and m["analytic_half_samples"] == 412
    assert m["margin_samples"] == 4537 and m["margin_s"] == pytest.approx(18.148)
    fb = copy.deepcopy(CFG)
    fb["boundary"]["margin_policy"] = "feature_band"
    assert margins(fb)["margin_samples"] == 412 + 37
    bad = copy.deepcopy(CFG)
    bad["boundary"]["margin_policy"] = "none"
    with pytest.raises(ValueError, match="margin_policy"):
        margins(bad)
    rep = report(CFG)
    assert rep["sensor_bandpass_taps"] == 8251 and set(rep["analytic_kernels"]) == set(CFG["connectivity"]["bands"])


def test_valid_window_grid():
    m = margins(CFG)["margin_samples"]
    n = int(280 * SF)                                     # a 300 s recording after cropping 2 x 10 s
    s = valid_window_starts(n, CFG)
    win = int(20 * SF)
    assert s[0] == m and (np.diff(s) == int(10 * SF)).all()
    assert s[-1] + win <= n - m and s[-1] + int(10 * SF) + win > n - m
    assert len(s) == expected_valid_windows(300, CFG) == 23                  # v3: 27 for 300 s
    assert len(valid_window_starts(int(20 * SF), CFG)) == 0                  # too short: no window


@pytest.mark.parametrize("sf", [1000.0, 1024.0])
def test_crop_edges_at_the_original_rate(sf):
    raw = _raw(_eeg(sf, 60), sf)
    info = crop_edges(raw, CFG)
    assert info["crop_start_s"] == pytest.approx(10.0) and info["crop_stop_s"] == pytest.approx(50.0)
    assert raw.n_times == int(40 * sf) and info["orig_n_times"] == int(60 * sf)
    with pytest.raises(ValueError, match="twice the edge trim"):
        crop_edges(_raw(_eeg(sf, 20), sf), CFG)


# ------------------------------------------------------------------ the whole chain
@pytest.mark.parametrize("sf", [1000.0, 1024.0])
def test_discarded_edges_cannot_influence_any_output(sf):
    """The same rules for D1 (1000 Hz) and D2 (1024 Hz): any change inside the discarded 10 s, here a
    5 mV transient, leaves every feature, the bad channels and the windows bit-identical."""
    W = np.random.default_rng(2).standard_normal((68, 30))
    x = _eeg(sf, 150)
    y = x.copy()
    n = int(10 * sf)
    g = np.random.default_rng(3).uniform(0.3, 1.5, (30, 1))
    y[:, :n] += g * 5e-3 * np.exp(-np.arange(n) / (2 * sf))
    y[:, -n:] += g[::-1] * 5e-3 * np.linspace(0, 1, n)
    (a, qa, ca, na), (b, qb, cb, nb) = _process(x, sf, W), _process(y, sf, W)
    assert qa["bad_channels"] == qb["bad_channels"] and np.array_equal(a["start_sample"], b["start_sample"])
    for k in ("plv", "ciplv", "ciplv_signed", "psd_abs", "psd_rel", "sensor_psd_abs", "sensor_psd_rel"):
        assert np.array_equal(a[k], b[k]), k
    m = margins(CFG)["margin_samples"]
    off = int(round(10 * SF))
    assert (a["start_sample"] >= off + m).all() and (a["stop_sample"] <= off + na - m).all()
    assert a["n_windows_total"] == expected_valid_windows(150, CFG) == 8
    assert not a["qc_windows"]["within_edge_filter_reach"].any()


def test_v3_configuration_still_uses_the_original_path():
    """Removing the boundary section restores the v3 behaviour (windows from the 10 s trim, FFT Hilbert)."""
    from adgraph.legacy.pipeline import process_recording
    from adgraph.preprocessing.clean import preprocess

    v3 = copy.deepcopy(CFG)
    v3.pop("boundary")
    v3["version"] = 3
    raw, _ = preprocess(_raw(_eeg(1000.0, 150), 1000.0), v3)
    res = process_recording(raw, np.random.default_rng(2).standard_normal((68, 30)), v3)
    assert res["start_sample"][0] == int(10 * SF) and res["n_windows_total"] == 12


def test_rebuild_end_to_end_and_validation(tmp_path, monkeypatch):
    """Two synthetic recordings through pipeline.run (stubbed operator): new versioned directory,
    manifest with the boundary processing, validation script passes and writes the dataset report."""
    pytest.importorskip("edfio")
    import pandas as pd

    import adgraph.legacy.pipeline as pl
    from test_end_to_end import _write_edf

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
    W = np.random.default_rng(2).standard_normal((68, 30))
    monkeypatch.setattr(pl, "fetch_fsaverage", lambda d: None)
    monkeypatch.setattr(pl, "build_roi_operator", lambda info, cfg, d: {
        "W": W, "roi_names": [f"roi{i:02d}" for i in range(68)], "ch_names": list(info["ch_names"]),
        "n_sources": 20484, "fwd_ch_names": list(info["ch_names"]), "roi_coords": np.zeros((68, 3))})
    monkeypatch.setattr(pl, "verify_operator", lambda *a, **k: 0.0)
    run_dir = pl.run(paths, CFG, tmp_path / "out", tmp_path / "sd")
    assert run_dir.name.startswith("graphs_v4_")
    man = json.loads((run_dir / "manifest.json").read_text())
    assert man["boundary_processing"]["margin_samples"] == 4537 and man["config"]["boundary"]["analytic"] == "fir"
    subj = pd.read_csv(run_dir / "subjects.csv")
    assert (subj["sample_offset"] == 2500).all() and (subj["crop_start_s"] == 10).all()
    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(f"drive_root: {tmp_path}\nprocessed_dir: {tmp_path / 'out'}\n"
                          f"dataset1: {{eeg_dir: '{d1}', participants: '{tmp_path / 'p1.xlsx'}'}}\n"
                          f"dataset2: {{eeg_dir: '{d2}', participants: '{tmp_path / 'p2.xlsx'}'}}\n")
    r = subprocess.run([sys.executable, "scripts/legacy/validate_graph_run.py", "--paths", str(paths_yaml), "--allow-dirty"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "COMPLETION REPORT" in r.stdout and "FAIL" not in r.stdout
    (val,) = run_dir.glob("validation_*")
    rep = pd.read_csv(val / "dataset_report.csv")
    assert len(rep) == 2 and rep["graphs_in_dataset"].sum() == len(pd.read_csv(run_dir / "windows.csv"))
    assert json.loads((val / "validation.json").read_text())["totals"]["recordings_with_graphs"] == 2


def test_latest_run_prefers_the_highest_version(tmp_path):
    import adgraph.legacy.pipeline as pl
    from adgraph.legacy.synthetic import make_synthetic_store

    for name, version, stamp in (("graphs_v4_a", 4, "2026-10-14T00:00:00"), ("graphs_v3_b", 3, "2026-10-15T00:00:00")):
        st = make_synthetic_store()
        st.manifest = {**st.manifest, "config": {"version": version}, "assembled_at": stamp}
        st.save(tmp_path / name)
        (tmp_path / name / "assembly.done.json").write_text("{}")
    assert pl.latest_complete_graph_run(tmp_path).name == "graphs_v4_a"     # newer v3 is never preferred
