"""Read-only amplitude diagnostic on synthetic EDFs with known amplitudes and known faults (log entry 38)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from adgraph.config import load_yaml
from adgraph.io.eeg import load_raw
from adgraph.legacy.amplitude import amplitude_report
from adgraph.preprocessing.clean import preprocess

pytest.importorskip("mne")
pytest.importorskip("edfio")

CFG = load_yaml("configs/pipeline.yaml")
D1 = [f"EEG E{k:02d}" for k in range(1, 33)]
D2 = [f"Misc ExG{k}" for k in range(1, 33)]
REPO = Path(__file__).resolve().parents[1]


def _eeg(sf, seconds=80, uv=10.0, drift_uv=0.0, dc_uv=0.0, seed=0):
    """Independent ~``uv`` µV channels with a 10 Hz rhythm, optional shared slow drift and DC offsets (µV)."""
    rng = np.random.default_rng(seed)
    n = int(sf * seconds)
    t = np.arange(n) / sf
    x = rng.standard_normal((32, n)) * uv + uv * np.sin(2 * np.pi * 10 * t)
    if drift_uv:
        x += drift_uv * np.sin(2 * np.pi * 0.2 * t)                 # common to all channels
    return x + rng.normal(0, dc_uv, (32, 1)) if dc_uv else x


def _write(path, names, sf, unit, data, prange=(-20000, 20000)):
    import edfio

    edfio.Edf([edfio.EdfSignal(np.clip(x, *prange), sampling_frequency=sf, label=n, physical_dimension=unit,
                               physical_range=prange) for x, n in zip(data, names)]).write(path)
    return path


def test_stages_units_and_agreement_with_preprocess(tmp_path):
    """D1 convention (values in µV, header 'V'): MNE reads volts x1e6, the dataset scale fixes it; the
    reported QC value is exactly what preprocess() checks."""
    p = _write(tmp_path / "sub1.edf", D1, 1000, "V", _eeg(1000))
    r = amplitude_report(p, "D1", CFG)
    s = r["summary"]
    assert s["header_dimensions"] == ["V"] and s["mne_to_volt"] == [1.0] and s["dataset_scale"] == 1e-6
    assert np.isclose(s["median_std_uv"]["mne"], s["median_std_uv"]["scaled"] * 1e6, rtol=1e-6)
    assert 10 < s["median_std_uv"]["scaled"] < 16                    # sqrt(10^2 + 10^2/2) = 12.2 µV
    raw, _ = load_raw(p, "D1", CFG)
    if CFG.get("boundary"):                      # pipeline v4 crops the edges before preprocessing (entry 41)
        from adgraph.preprocessing.boundary import crop_edges
        crop_edges(raw, CFG)
    _, qc = preprocess(raw, CFG)
    ck = s["checks"]                                                 # the pipeline's own checks, same values
    assert np.isclose(ck["plausibility"]["value_uv"], qc["median_std_uv"], rtol=1e-9)
    assert np.isclose(ck["unit_safeguard"]["value_uv"], qc["gross_robust_std_uv"], rtol=1e-9)
    assert ck["plausibility"]["passes"] and ck["unit_safeguard"]["passes"] and s["former_check"]["passes"]
    assert len(ck["channels"]) == 30 and ck["bad_channels"] == {}
    assert any(e.startswith("UNITS: 1-45 Hz average-referenced std") and "no evidence" in e for e in s["evidence"])
    used = r["header"][r["header"]["electrode"] != ""]
    assert len(used) == 32 and used["electrode"].tolist() == CFG["channels"]["recorded"]
    assert set(r["channels"]["stage"]) == {"mne", "scaled", "qc", "hp1", "car", "car_hp1"}
    assert r["channels"].groupby("stage").size().eq(30).all()       # 30 channels after M1/M2 drop


def test_d2_header_microvolts(tmp_path):
    p = _write(tmp_path / "2_1_1.edf", D2, 1024, "uV", _eeg(1024))
    s = amplitude_report(p, "D2", CFG)["summary"]
    assert s["header_dimensions"] == ["µV"] and s["mne_to_volt"] == [1e-6] and s["dataset_scale"] == 1.0
    assert 10 < s["median_std_uv"]["scaled"] < 16 and s["orig_sfreq"] == 1024


def test_unit_error_is_identified(tmp_path):
    """µV values labelled 'V' but treated as correctly labelled (no rescale) -> 10^6 too large."""
    p = _write(tmp_path / "x.edf", D2, 1024, "V", _eeg(1024))
    s = amplitude_report(p, "D2", CFG)["summary"]
    assert not s["former_check"]["passes"] and not s["checks"]["unit_safeguard"]["passes"]
    assert not s["checks"]["plausibility"]["passes"] and "unit safeguard" in s["checks"]["plausibility"]["error"]
    assert any(e.startswith("UNITS:") and "OUTSIDE" in e for e in s["evidence"])


def test_drift_and_common_mode_are_identified_not_units(tmp_path):
    """Correct units, but a large shared slow drift and DC offsets: the check fails at its stage
    while the 1-45 Hz average-referenced EEG is ordinary."""
    p = _write(tmp_path / "sub1.edf", D1, 1000, "V", _eeg(1000, drift_uv=500.0, dc_uv=3000.0))
    s = amplitude_report(p, "D1", CFG)["summary"]
    assert not s["former_check"]["passes"] and s["median_std_uv"]["qc"] > 200
    assert s["checks"]["unit_safeguard"]["passes"] and s["checks"]["plausibility"]["passes"]
    assert s["median_std_uv"]["car_hp1"] < 16
    assert s["median_band_fraction_qc"]["0.1-1"] > 0.5 and s["common_mode_fraction"]["qc"] > 0.5
    ev = " ".join(s["evidence"])
    assert "no evidence of a unit-conversion error" in ev and "slow drift" in ev and "common-mode" in ev
    assert s["median_abs_dc_offset_uv"] > 500


def test_clipping_and_edges_are_reported(tmp_path):
    x = _eeg(1000)
    x[3, 5000:5500] = 1e6                                           # saturates the physical range
    x[:, :2000] += 3000.0 * np.exp(-np.arange(2000) / 400.0)        # settling transient at the start
    p = _write(tmp_path / "sub1.edf", D1, 1000, "V", x)
    s = amplitude_report(p, "D1", CFG)["summary"]
    assert s["channels_clipping"] == ["F7"] and s["max_clip_fraction"] > 1e-3      # E04 -> F7
    assert s["qc_trimmed_median_std_uv"] < s["median_std_uv"]["qc"]
    assert any(e.startswith("EDGES:") for e in s["evidence"])
    assert s["checks"]["plausibility"]["passes"]                   # transient only in the discarded edge


def test_line_noise_is_reported_before_and_after_the_bandpass(tmp_path):
    """50 Hz survives the 0.1-45 Hz FIR at about -4.5 dB; 60 Hz is removed."""
    x = _eeg(1000)
    t = np.arange(x.shape[1]) / 1000
    x += 20 * np.sin(2 * np.pi * 50 * t)                          # 14.1 µV RMS mains on every channel
    s = amplitude_report(_write(tmp_path / "sub1.edf", D1, 1000, "V", x), "D1", CFG)["summary"]
    ln = s["line_noise"]["50Hz"]
    assert 12 < ln["median_rms_uv_raw"] < 16
    assert -5.0 < ln["bandpass_gain_db"] < -4.0 and s["line_noise"]["60Hz"]["bandpass_gain_db"] < -50
    assert np.isclose(ln["median_rms_uv_bandpassed"] / ln["median_rms_uv_raw"], 10 ** (ln["bandpass_gain_db"] / 20),
                      rtol=0.15)
    assert any(e.startswith("LINE NOISE 50 Hz") for e in s["evidence"])


def test_cli_writes_only_a_new_diagnostics_folder(tmp_path):
    root = tmp_path / "AD-LEB"
    d1, d2 = root / "EEGs" / "Dataset 1 (2017)", root / "EEGs" / "Dataset 2 (2019)" / "RestingState"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    f1 = _write(d1 / "sub1.edf", D1, 1000, "V", _eeg(1000, seconds=40))
    f2 = _write(d2 / "2_1012019_1.edf", D2, 1024, "uV", _eeg(1024, seconds=40))
    before = {f: f.read_bytes() for f in (f1, f2)}
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"drive_root: {root}\n"
                     "dataset1: {eeg_dir: '{drive_root}/EEGs/Dataset 1 (2017)'}\n"
                     "dataset2: {eeg_dir: '{drive_root}/EEGs/Dataset 2 (2019)/RestingState'}\n")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "scripts/legacy/diagnose_amplitude.py", "--paths", str(paths), "--out", str(out),
                        "--allow-ephemeral-storage"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "D1-1" in r.stdout and "D2-2" in r.stdout and "evidence:" in r.stdout
    (folder,) = (out / "diagnostics").iterdir()
    names = {p.name for p in folder.iterdir()}
    assert {"summary.json", "manifest.json", "amplitude.png", "header_D1-1.csv", "channels_D2-2.csv",
            "spectral_D1-1.csv", "blocks_D2-2.csv"} <= names
    assert {f: f.read_bytes() for f in (f1, f2)} == before            # EDFs untouched
