"""Index-only EDF channel labels -> electrodes of the AD-LEB 32-channel layout (log entry 37).

Synthetic only. The layout text below has the format of AD-LEB/electrodes_positions_32.loc
(EEGLAB polar, dot-padded labels, CRLF lines); no recording data are used.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adgraph.config import common_channels, load_yaml
from adgraph.io.eeg import ChannelMappingError, channel_mapping, inspect_headers, list_recordings, load_raw, \
    make_montage, mapping_mode
from adgraph.io.layout import check_layout, find_layout_file, indexed_number, parse_loc, spatial_consistency, \
    verify_layout

CFG = load_yaml("configs/pipeline.yaml")
RECORDED = CFG["channels"]["recorded"]
USED = common_channels(CFG)
REPO = Path(__file__).resolve().parents[1]

LOC = ("1 -18 0.479 Fp1.\r\n2 0 0.471 Fpz.\r\n3 18 0.479 Fp2.\r\n4 -54 0.491 F7..\r\n5 -39 0.318 F3..\r\n"
       "6 0 0.234 Fz..\r\n7 39 0.318 F4..\r\n8 54 0.491 F8..\r\n9 -69 0.382 FC5.\r\n10 -45 0.167 FC1.\r\n"
       "11 45 0.167 FC2.\r\n12 69 0.382 FC6.\r\n13 -100 0.500 M1..\r\n14 -90 0.500 T7..\r\n15 -90 0.247 C3..\r\n"
       "16 0 0.000 Cz..\r\n17 90 0.247 C4..\r\n18 90 0.500 T8..\r\n19 100 0.500 M2..\r\n20 -111 0.382 CP5.\r\n"
       "21 -136 0.168 CP1.\r\n22 136 0.168 CP2.\r\n23 111 0.382 CP6.\r\n24 -126 0.491 P7..\r\n"
       "25 -141 0.321 P3..\r\n26 180 0.239 Pz..\r\n27 141 0.321 P4..\r\n28 126 0.491 P8..\r\n"
       "29 180 0.35 POz.\r\n30 -169 0.482 O1..\r\n31 180 0.472 Oz..\r\n32 169 0.482 O2..")

D1_NAMES = [f"EEG E{k:02d}" for k in range(1, 33)]
D2_NAMES = [f"Misc ExG{k}" for k in range(1, 33)]


def _loc(tmp_path, text=LOC, name="electrodes_positions_32.loc"):
    p = tmp_path / name
    p.write_bytes(text.encode())
    return p


# ---------------------------------------------------------------- layout file

def test_parse_loc_real_format_matches_config(tmp_path):
    df = parse_loc(_loc(tmp_path))
    assert df["index"].tolist() == list(range(1, 33))
    assert df["label"].tolist() == RECORDED                  # dots and CR stripped, config order
    assert verify_layout(_loc(tmp_path), RECORDED) == RECORDED
    assert [c for c in RECORDED if c not in USED] == ["M1", "M2"] and len(USED) == 30


def test_loc_angles_agree_with_standard_1020():
    """The .loc polar angles point where the standard_1020 montage puts the same labels
    (EEGLAB: theta from the nose, positive to the right), so labels and positions agree."""
    import tempfile

    pytest.importorskip("mne")
    with tempfile.TemporaryDirectory() as d:
        df = parse_loc(_loc(Path(d)))
    pos = make_montage("standard_1020").get_positions()["ch_pos"]
    lookup = {k.lower(): v for k, v in pos.items()}
    for row in df.itertuples():
        if row.radius == 0:
            continue
        x, y, _ = lookup[row.label.lower()]
        az = np.degrees(np.arctan2(x, y))
        assert abs((az - row.theta + 180) % 360 - 180) < 20, row.label


@pytest.mark.parametrize("text,match", [
    (LOC.replace("2 0 0.471 Fpz.", "3 0 0.471 Fpz."), "indices"),          # repeated index
    (LOC.replace("\r\n32 169 0.482 O2..", ""), None),                       # 31 electrodes
    (LOC.replace("Fpz.", "Fp1."), "labels"),                                # repeated label
    (LOC.replace("4 -54 0.491 F7..", "4 -54 F7.."), "line"),                # missing field
    ("", "no electrodes"),
])
def test_malformed_loc_is_refused(tmp_path, text, match):
    p = _loc(tmp_path, text)
    if match is None:                                       # parses, but does not match the config
        with pytest.raises(ValueError, match="differs"):
            verify_layout(p, RECORDED)
    else:
        with pytest.raises(ValueError, match=match):
            parse_loc(p)


def test_layout_order_mismatch_is_refused(tmp_path):
    swapped = LOC.replace("Fp1.", "TMP.").replace("Fp2.", "Fp1.").replace("TMP.", "Fp2.")
    with pytest.raises(ValueError, match="differs"):
        verify_layout(_loc(tmp_path, swapped), RECORDED)


def test_find_layout_file(tmp_path):
    (tmp_path / "EEGs" / "Dataset 1 (2017)").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        find_layout_file(tmp_path)
    copy = _loc(tmp_path / "EEGs", name="electrodes_positions_32 (1).loc")
    assert find_layout_file(tmp_path) == copy                # browser-download copy accepted
    _loc(tmp_path / "EEGs" / "Dataset 1 (2017)", LOC.replace("Fp1.", "Fp9."), name="electrodes_positions_32.loc")
    with pytest.raises(ValueError, match="several different"):
        find_layout_file(tmp_path)
    exact = _loc(tmp_path)
    assert find_layout_file(tmp_path) == exact               # root file wins
    info = check_layout({"drive_root": str(tmp_path), "electrode_layout": str(tmp_path / "nope.loc")}, RECORDED)
    assert info["path"] == str(exact) and len(info["sha256"]) == 64


# ---------------------------------------------------------------- label mapping

@pytest.mark.parametrize("name,k", [("EEG E01", 1), ("EEG E32", 32), ("Misc ExG1", 1), ("Misc ExG32", 32),
                                    ("E7", 7), ("ExG 07", 7), ("EEG E05.", 5), ("eeg e5..", 5),
                                    ("Fp1", None), ("ECG", None), ("Status", None), ("EEG Fp1-REF", None)])
def test_indexed_number(name, k):
    assert indexed_number(name) == k


@pytest.mark.parametrize("names", [D1_NAMES, D2_NAMES, [n + "." for n in D2_NAMES]])
def test_indexed_mapping_both_conventions(names):
    mapping, unknown = channel_mapping(names + ["Status"], RECORDED)
    assert list(mapping.values()) == RECORDED and unknown == ["Status"]
    assert mapping[names[12]] == "M1" and mapping[names[18]] == "M2" and mapping[names[31]] == "O2"
    assert len(set(mapping.values())) == 32 and mapping_mode(mapping) == "indexed"


def test_d1_and_d2_map_identically():
    m1, _ = channel_mapping(D1_NAMES, RECORDED)
    m2, _ = channel_mapping(D2_NAMES, RECORDED)
    assert list(m1.values()) == list(m2.values())


def test_named_mapping_still_works():
    mapping, unknown = channel_mapping([f"{c.upper()}-REF" for c in RECORDED] + ["ECG"], RECORDED)
    assert list(mapping.values()) == RECORDED and unknown == ["ECG"] and mapping_mode(mapping) == "named"


def test_ambiguous_inventories_raise():
    with pytest.raises(ChannelMappingError, match="mixed"):
        channel_mapping(D1_NAMES[:-1] + ["O2"], RECORDED)
    with pytest.raises(ChannelMappingError, match="more than once"):
        channel_mapping(D1_NAMES + ["Misc ExG5"], RECORDED)
    with pytest.raises(ChannelMappingError, match="more than once"):
        channel_mapping(D1_NAMES[:4] + ["EEG E5", "EEG E05."], RECORDED)


def test_incomplete_and_out_of_range_inventory():
    names = [n for n in D1_NAMES if n != "EEG E05"] + ["EEG E33"]
    mapping, unknown = channel_mapping(names, RECORDED)
    assert "F3" not in mapping.values() and unknown == ["EEG E33"] and len(mapping) == 31


# ---------------------------------------------------------------- EDF loading (synthetic)

def _write_edf(path, names, sf, unit, seconds=60, data=None):
    import edfio

    rng = np.random.default_rng(0)
    if data is None:
        data = rng.standard_normal((len(names), sf * seconds)) * 10.0
    signals = [edfio.EdfSignal(x, sampling_frequency=sf, label=n, physical_dimension=unit,
                               physical_range=(-20000, 20000)) for x, n in zip(data, names)]
    edfio.Edf(signals).write(path)


@pytest.mark.parametrize("dataset,names,sf,unit", [("D1", D1_NAMES, 1000, "V"), ("D2", D2_NAMES, 1024, "uV")])
def test_load_raw_indexed_labels(tmp_path, dataset, names, sf, unit):
    mne = pytest.importorskip("mne")
    pytest.importorskip("edfio")
    from adgraph.preprocessing.clean import preprocess

    path = tmp_path / "rec.edf"
    _write_edf(path, names, sf, unit, seconds=40)
    raw, info = load_raw(path, dataset, CFG)
    assert raw.ch_names == USED                              # 30 channels, canonical order, M1/M2 dropped
    assert info["dropped_channels"] == ["M1", "M2"] and info["channel_naming"] == "indexed"
    assert info["orig_sfreq"] == sf and info["unrecognised_channels"] == []
    assert set(raw.get_channel_types()) == {"eeg"}          # D2 'Misc' channels become EEG
    ref_info = mne.create_info(USED, 250.0, "eeg")           # same names, standard_1020 positions
    ref_info.set_montage(make_montage("standard_1020"), match_case=False)
    ref = ref_info.get_montage().get_positions()["ch_pos"]
    got = raw.get_montage().get_positions()["ch_pos"]
    for ch in USED:
        assert np.allclose(got[ch], ref[ch]), ch
    assert np.isclose(np.median(np.abs(raw.get_data())), 6.7e-6, rtol=0.2)   # volts after the D1 rescale
    raw, _ = preprocess(raw, CFG)
    assert raw.info["sfreq"] == 250 and raw.ch_names == USED
    assert mne.pick_types(raw.info, eeg=True).size == 30


def test_load_raw_refuses_incomplete_inventory(tmp_path):
    pytest.importorskip("edfio")
    path = tmp_path / "rec.edf"
    _write_edf(path, [n for n in D2_NAMES if n != "Misc ExG5"], 1024, "uV", seconds=5)
    with pytest.raises(ValueError, match="F3"):
        load_raw(path, "D2", CFG)


def test_spatial_consistency_detects_wrong_mapping():
    pytest.importorskip("mne")
    pos = make_montage("standard_1020").get_positions()["ch_pos"]
    pos = {k.lower(): v for k, v in pos.items()}
    xyz = np.array([pos[c.lower()] for c in USED])
    rng = np.random.default_rng(0)
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    mix = np.exp(-(d / 0.04) ** 2)                            # volume conduction: nearby channels share signal
    data = mix @ rng.standard_normal((30, 5000))
    assert spatial_consistency(data, xyz) < -0.3
    assert abs(spatial_consistency(data, xyz[rng.permutation(30)])) < 0.2


# ---------------------------------------------------------------- strict header gate agrees with load_raw

def _dataset_tree(tmp_path, layout=LOC, drop_d2=None):
    root = tmp_path / "AD-LEB"
    d1, d2 = root / "EEGs" / "Dataset 1 (2017)", root / "EEGs" / "Dataset 2 (2019)" / "RestingState"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    _write_edf(d1 / "sub1.edf", D1_NAMES, 1000, "V", seconds=2)
    _write_edf(d2 / "3_1012019_1.edf", [n for n in D2_NAMES if n != drop_d2] + ["Status"], 1024, "uV", seconds=2)
    _loc(root, layout)
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"drive_root: {root}\n"
                     "dataset1: {eeg_dir: '{drive_root}/EEGs/Dataset 1 (2017)'}\n"
                     "dataset2: {eeg_dir: '{drive_root}/EEGs/Dataset 2 (2019)/RestingState'}\n"
                     "electrode_layout: '{drive_root}/electrodes_positions_32.loc'\n")
    return root, paths


def _strict(tmp_path, paths):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return subprocess.run([sys.executable, "scripts/inspect_raw.py", "--paths", str(paths), "--out", str(out),
                           "--strict", "--allow-ephemeral-storage"], cwd=REPO, capture_output=True, text=True)


def test_strict_header_check_agrees_with_load_raw(tmp_path):
    pytest.importorskip("edfio")
    root, paths = _dataset_tree(tmp_path)
    r = _strict(tmp_path, paths)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "header check passed" in r.stdout and "sha256" in r.stdout and "indexed labels" in r.stdout
    recs = pd.concat([list_recordings(root / "EEGs" / "Dataset 1 (2017)", "D1"),
                      list_recordings(root / "EEGs" / "Dataset 2 (2019)" / "RestingState", "D2")])
    report = inspect_headers(recs, RECORDED)
    assert (report["naming"] == "indexed").all() and (report["missing_canonical"] == "").all()
    assert report["unrecognised"].tolist() == ["", "Status"]
    for rec in recs.itertuples():                              # the gate's verdict holds for loading
        raw, _ = load_raw(rec.path, rec.dataset, CFG)
        assert raw.ch_names == USED


def test_strict_header_check_fails_on_missing_channel(tmp_path):
    pytest.importorskip("edfio")
    root, paths = _dataset_tree(tmp_path, drop_d2="Misc ExG5")
    r = _strict(tmp_path, paths)
    assert r.returncode != 0 and "F3" in r.stdout + r.stderr
    with pytest.raises(ValueError, match="F3"):
        load_raw(root / "EEGs" / "Dataset 2 (2019)" / "RestingState" / "3_1012019_1.edf", "D2", CFG)


def test_strict_header_check_fails_on_dropped_channel_only_when_used(tmp_path):
    pytest.importorskip("edfio")
    _, paths = _dataset_tree(tmp_path, drop_d2="Misc ExG13")      # M1 is not used
    r = _strict(tmp_path, paths)
    assert r.returncode == 0, r.stdout + r.stderr


def test_strict_header_check_fails_on_bad_layout(tmp_path):
    pytest.importorskip("edfio")
    _, paths = _dataset_tree(tmp_path, layout=LOC.replace("Fp1.", "TMP.").replace("Fp2.", "Fp1.")
                             .replace("TMP.", "Fp2."))
    r = _strict(tmp_path, paths)
    assert r.returncode != 0 and "electrode layout cannot be verified" in r.stdout + r.stderr
