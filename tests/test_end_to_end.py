"""Synthetic EDF -> load -> preprocess -> windows -> graphs (fsaverage replaced by a random operator)."""

import numpy as np
import pytest

from adgraph.config import common_channels, load_yaml
from adgraph.io.eeg import load_raw
from adgraph.pipeline import process_recording
from adgraph.preprocessing.clean import preprocess

mne = pytest.importorskip("mne")
pytest.importorskip("edfio")

CFG = load_yaml("configs/pipeline.yaml")


def _write_edf(path, names, unit):
    """EDF with ~10 uV signals. ``unit='V'`` mimics D1: values are uV but the header says V."""
    import edfio

    rng = np.random.default_rng(0)
    sf, dur = 500, 150
    t = np.arange(sf * dur) / sf
    data = (np.sin(2 * np.pi * 10 * t) + rng.standard_normal((len(names), t.size))) * 10.0  # in uV
    data[4] = rng.standard_normal(t.size) * 2000.0                                       # one noisy channel
    signals = [edfio.EdfSignal(x, sampling_frequency=sf, label=n, physical_dimension=unit,
                               physical_range=(-20000, 20000)) for x, n in zip(data, names)]
    edfio.Edf(signals).write(path)


@pytest.mark.parametrize("dataset,unit", [("D1", "V"), ("D2", "uV")])
def test_pipeline_on_synthetic_edf(tmp_path, dataset, unit):
    names = [f"{c.upper()}-REF" for c in CFG["channels"]["recorded"]] + ["ECG"]
    path = tmp_path / "rec.edf"
    _write_edf(path, names, unit)

    raw, load_qc = load_raw(path, dataset, CFG)
    assert raw.ch_names == common_channels(CFG) and load_qc["unrecognised_channels"] == ["ECG"]
    raw, qc = preprocess(raw, CFG)
    assert raw.info["sfreq"] == 250 and 5 < qc["median_std_uv"] < 20
    assert list(qc["bad_channels"]) == ["F3"]  # index 4 of the recorded list

    W = np.random.default_rng(1).standard_normal((68, 30))
    res = process_recording(raw, W, CFG)
    n = res["n_windows_kept"]
    assert res["n_windows_total"] == 5 and 1 <= n <= 5    # (150 - 2*10 - 40) / 20 + 1
    assert res["plv"].shape == (n, 4, 68, 68) and res["power"].shape == (n, 4, 68)
    assert np.allclose(res["plv"], res["plv"].transpose(0, 1, 3, 2), atol=1e-6)


def test_wrong_unit_scale_is_caught(tmp_path):
    names = CFG["channels"]["recorded"]
    path = tmp_path / "rec.edf"
    _write_edf(path, names, "uV")                # correct header, but treated as D1 -> rescaled twice
    raw, _ = load_raw(path, "D1", CFG)
    with pytest.raises(ValueError, match="plausible range"):
        preprocess(raw, CFG)


def test_run_writes_outputs(tmp_path, monkeypatch):
    import json

    import pandas as pd

    import adgraph.pipeline as pl

    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir(), d2.mkdir()
    names = CFG["channels"]["recorded"]
    for f, unit in [(d1 / "sub1.edf", "V"), (d1 / "sub2.edf", "V"), (d2 / "2_842019_1.edf", "uV"),
                    (d2 / "17_952019_1.edf", "uV")]:
        _write_edf(f, names, unit)
    (d2 / "17_952019_2.gdf").write_bytes(b"")  # task file in the same folder must be ignored
    pd.DataFrame({"ID": ["Sub1", "Sub2"], "Age": [72, 68], "Gender": ["M", "F"], "Group": ["AD", "HC"],
                  "MMSE": [21, 27]}).to_excel(tmp_path / "p1.xlsx", index=False)
    pd.DataFrame({"Id": ["2_842019", "17_952019", "99_1"], "Age": [71, 79, 70], "Gender": ["M", "F", "F"],
                  "Group": ["HC", "AD", "HC"], "MMSE (/30)": [29, 11, 30]}).to_excel(tmp_path / "p2.xlsx", index=False)
    paths = {"dataset1": {"eeg_dir": str(d1), "participants": str(tmp_path / "p1.xlsx")},
             "dataset2": {"eeg_dir": str(d2), "participants": str(tmp_path / "p2.xlsx")}}

    W = np.random.default_rng(2).standard_normal((68, 30))
    rois = [f"roi{i:02d}" for i in range(68)]
    monkeypatch.setattr(pl, "fetch_fsaverage", lambda d: None)
    monkeypatch.setattr(pl, "build_roi_operator", lambda info, cfg, d: {
        "W": W, "roi_names": rois, "ch_names": list(info["ch_names"]), "n_sources": 20484,
        "fwd_ch_names": list(info["ch_names"])})
    monkeypatch.setattr(pl, "verify_operator", lambda *a, **k: 0.0)

    out = pl.run(paths, CFG, tmp_path / "out", tmp_path / "subjects")
    npz = np.load(out / "connectivity.npz")
    windows = pd.read_csv(out / "windows.csv")
    subjects = pd.read_csv(out / "subjects.csv")
    manifest = json.loads((out / "manifest.json").read_text())

    assert subjects["key"].tolist() == ["D1-1", "D1-2", "D2-2", "D2-17"]
    assert (subjects["status"] == "ok").all()
    assert npz["plv"].shape[0] == npz["ciplv"].shape[0] == npz["power"].shape[0] == len(windows)
    assert npz["plv"].shape[1:] == (4, 68, 68) and npz["bands"].tolist() == ["theta", "alpha1", "alpha2", "beta"]
    assert np.allclose(windows.groupby("key")["sample_weight"].sum(), 1.0)
    assert manifest["registry"]["metadata_without_recording"] == ["D2-99"]
    assert manifest["source_operator"]["shared_across_datasets"] is True
    assert manifest["summary"]["n_graphs"] == len(windows)
