"""Dataset 3 electrode coordinates from the EEGLAB .set files (methodology log entry 46), synthetic data only.

* a real EEGLAB .set (written with scipy) is read: labels, X/Y/Z, chaninfo.filename, the 'lookup' in EEG.history;
* frame handling: the stored (MNI / Colin27) positions reach fsaverage MRI space unchanged, whereas MNE's EEGLAB
  reader used as is would place them ~5 cm away (it labels them 'head' with invented fiducials);
* invalid or uninterpretable coordinates stop the subject (CoordinateError), never a template fallback;
* load_raw uses the file's positions; one operator per distinct geometry; a subject never runs on another
  geometry's operator; the template-based operator is not reused; D1/D2 unchanged.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io

from adgraph.config import common_channels, config_hash, load_yaml
from adgraph.io import bids, chanlocs
from synthetic_chanlocs import LOOKUP, patch_reader, provider_chanlocs, template_xyz_mm

mne = pytest.importorskip("mne")
REPO = Path(__file__).resolve().parents[1]
CFG = load_yaml(REPO / "configs" / "pipeline_d3.yaml")
CH = CFG["channels"]["recorded"]


def write_set(path, xyz_mm, lookup=LOOKUP, nosedir="+X", n=5000):
    cl = np.array([(c, "EEG", *p) for c, p in zip(CH, xyz_mm)],
                  dtype=[("labels", "O"), ("type", "O"), ("X", "O"), ("Y", "O"), ("Z", "O")])
    nod = np.array([("A1", "", -25.0, 86.0, -68.0)],
                   dtype=[("labels", "O"), ("type", "O"), ("X", "O"), ("Y", "O"), ("Z", "O")])
    data = np.random.default_rng(0).standard_normal((len(CH), n)).astype(np.float32) * 10
    hist = f"EEG=pop_chanedit(EEG, 'lookup','{lookup}');" if lookup else ""
    scipy.io.savemat(path, {"setname": "x", "nbchan": len(CH), "trials": 1, "pnts": n, "srate": 500, "xmin": 0.0,
                            "xmax": (n - 1) / 500, "data": data, "chanlocs": cl, "event": np.array([]),
                            "chaninfo": {"filename": lookup or "", "nosedir": nosedir, "nodatchans": nod},
                            "history": hist, "icawinv": np.array([]), "epoch": np.array([]), "ref": "A1 A2"},
                     appendmat=False)


def fsaverage_mri_mm(info):
    from mne.transforms import apply_trans

    trans = mne.read_trans(Path(mne.__file__).parent / "data" / "fsaverage" / "fsaverage-trans.fif")
    return apply_trans(trans, np.array([info["chs"][info["ch_names"].index(c)]["loc"][:3] for c in CH])) * 1000


def test_real_set_file_coordinates_frame_and_hash(tmp_path):
    write_set(tmp_path / "s.set", template_xyz_mm(CH))
    loc = chanlocs.read_chanlocs(tmp_path / "s.set")
    assert loc["labels"] == CH and loc["chaninfo_filename"] == LOOKUP and loc["history_lookups"] == [LOOKUP]
    assert loc["fiducials"] == {}                                   # A1 is not a fiducial
    g = chanlocs.establish(loc, CH, CFG)
    assert g["frame"] == "mri" and g["lookup"] == "standard_1005.elc" and len(g["hash"]) == 16
    assert g["template_deviation_mm"] < 1e-6                        # the stored positions ARE the template
    # MNE's reader used as is: the same numbers labelled 'head' with invented fiducials -> ~5 cm off
    raw = mne.io.read_raw_eeglab(tmp_path / "s.set", preload=False, verbose="error")
    truth = np.asarray(g["pos_m"]) * 1000
    off = np.linalg.norm(fsaverage_mri_mm(raw.info) - truth, axis=1)
    assert off.min() > 30
    info = mne.create_info(CH, 250.0, "eeg")
    info.set_montage(chanlocs.montage(g))                          # ours: exactly the stored positions
    assert np.linalg.norm(fsaverage_mri_mm(info) - truth, axis=1).max() < 1e-3


@pytest.mark.parametrize("change, message", [
    (lambda x: x.__setitem__((3, 1), np.nan), "non-finite"),
    (lambda x: None, "cannot be established"),                    # lookup removed below
    (lambda x: x.__imul__(1e-3), "radius"),                       # metres stored, mm configured
    (lambda x: x.__setitem__((slice(None), 1), -x[:, 1]), "sides"),   # left/right mirrored
])
def test_invalid_or_uninterpretable_coordinates_stop(change, message):
    xyz = template_xyz_mm(CH)
    change(xyz)
    loc = provider_chanlocs(CH, xyz, lookup=None if message == "cannot be established" else LOOKUP)
    with pytest.raises(chanlocs.CoordinateError, match=message):
        chanlocs.establish(loc, CH, CFG)
    with pytest.raises(chanlocs.CoordinateError, match="no coordinates"):
        chanlocs.establish(provider_chanlocs(CH[:-1]), CH, CFG)
    with pytest.raises(chanlocs.CoordinateError, match="nosedir"):
        chanlocs.establish({**provider_chanlocs(CH), "nosedir": "+Y"}, CH, CFG)
    with pytest.raises(chanlocs.CoordinateError, match="cannot be established"):
        chanlocs.establish(provider_chanlocs(CH, lookup=r"C:\caps\my_cap.elc"), CH, CFG)


def test_load_raw_uses_the_file_positions(tmp_path):
    from adgraph.io.eeg import load_raw

    xyz = template_xyz_mm(CH)
    xyz[CH.index("Cz")] += [0.0, 0.0, -7.0]                         # a geometry that is not the template
    write_set(tmp_path / "s.set", xyz)
    raw, info = load_raw(tmp_path / "s.set", "D3", CFG)
    g = info["geometry"]
    assert raw.ch_names == CH and g["template_deviation_mm"] == pytest.approx(7.0, abs=1e-6)
    assert np.allclose(fsaverage_mri_mm(raw.info), np.asarray(g["pos_m"]) * 1000, atol=1e-3)


def _run(tmp_path, monkeypatch, shifted=()):
    import adgraph.io.eeg as eeg
    import adgraph.pipeline as pl
    from test_d3 import synthetic_raw, write_participants

    root = tmp_path / "ds004504"
    write_participants(root, {1: "A", 2: "C", 4: "A"})
    for i in (1, 2, 4):
        f = root / "derivatives" / f"sub-{i:03d}" / "eeg" / f"sub-{i:03d}_task-eyesclosed_eeg.set"
        f.parent.mkdir(parents=True)
        f.write_text(str(i))
    (root / "dataset_description.json").write_text("{}")
    monkeypatch.setattr(bids, "EXPECTED", {"AD": 2, "HC": 1})
    monkeypatch.setattr(eeg, "read_header", lambda path, fmt: synthetic_raw(int(Path(path).read_text())))

    def per_file(path):
        xyz = template_xyz_mm(CH)
        if int(Path(path).read_text()) in shifted:
            xyz[CH.index("Pz")] += [-4.0, 0.0, 0.0]
        return xyz
    patch_reader(monkeypatch, CH, per_file)
    built = []

    def build(info, cfg, d):
        built.append(np.array([ch["loc"][:3] for ch in info["chs"]]))
        return {"W": np.random.default_rng(len(built)).standard_normal((68, 19)), "roi_names": [f"r{i}" for i in range(68)],
                "ch_names": list(info["ch_names"]), "n_sources": 20484, "fwd_ch_names": list(info["ch_names"])}
    monkeypatch.setattr(pl, "fetch_fsaverage", lambda d: None)
    monkeypatch.setattr(pl, "build_roi_operator", build)
    monkeypatch.setattr(pl, "verify_operator", lambda *a, **k: 1e-12)
    out = pl.run({"dataset3": {"root": str(root)}}, CFG, tmp_path / "processed", tmp_path / "subjects",
                 allow_ephemeral=True)
    return pl, out, built


def test_one_operator_per_distinct_geometry(tmp_path, monkeypatch):
    from adgraph.graphs.store import GraphStore

    pl, out, built = _run(tmp_path, monkeypatch, shifted=(4,))
    assert len(built) == 2 and not np.allclose(built[0], built[1])    # two geometries -> two operators
    s = GraphStore.load(out).subjects.set_index("key")
    assert s.loc["D3-1", "montage_hash"] == s.loc["D3-2", "montage_hash"] != s.loc["D3-4", "montage_hash"]
    assert s.loc["D3-1", "operator_id"] == s.loc["D3-2", "operator_id"] != s.loc["D3-4", "operator_id"]
    assert (s["coord_frame"] == "mri").all() and (s["coord_source"] == "eeglab_chanlocs").all()
    m = json.loads((out / "manifest.json").read_text())
    assert len(m["source_operators"]) == 2 and m["source_operator"]["geometry"]["montage_hash"] == s.loc["D3-1",
                                                                                                         "montage_hash"]
    st = pl.status({"dataset3": {"root": str(tmp_path / "ds004504")}}, CFG, tmp_path / "processed")
    assert (st["subjects"]["state"] == "completed").all()


def test_subject_never_uses_another_geometrys_operator(monkeypatch):
    import adgraph.io.eeg as eeg
    import adgraph.pipeline as pl
    from test_d3 import synthetic_raw
    from test_d3_discontinuities import Rec

    patch_reader(monkeypatch, CH)
    monkeypatch.setattr(eeg, "read_header", lambda path, fmt: synthetic_raw(1))
    rec = Rec("D3-1", "D3", 1, "AD", 1, 70.0, "F", 20.0, np.nan, "", "x.set", 1)
    op = {"W": np.zeros((68, 19)), "ch_names": CH, "roi_names": [f"r{i}" for i in range(68)], "operator_id": "o",
          "geometry": {"montage_hash": "someothergeometry"}}
    with pytest.raises(chanlocs.CoordinateError, match="differs from the operator"):
        pl.process_subject(rec, CFG, op)


def test_template_operator_not_reused_and_d1d2_unchanged():
    import adgraph.pipeline as pl

    g = chanlocs.establish(provider_chanlocs(CH), common_channels(CFG), CFG)
    old = json.loads(json.dumps(CFG))
    old["channels"]["montage"] = "standard_1020"
    del old["channels"]["coordinates"]
    assert pl.stable_hash(pl.operator_identity(CFG, g)) != pl.stable_hash(pl.operator_identity(old))
    assert config_hash(CFG) != config_hash(old)                     # new D3 graph-run identity
    d12 = load_yaml(REPO / "configs" / "pipeline.yaml")
    assert not chanlocs.enabled(d12) and "geometry" not in pl.operator_identity(d12)
    assert config_hash(d12) == "33faa4b89a5d"                       # frozen D1/D2 graph run unchanged
    assert pl.subject_geometries(pd.DataFrame({"key": ["D1-1"], "path": ["x"]}), d12) == {}


def test_input_check_reports_coordinates_and_stops_on_bad_ones(tmp_path, monkeypatch):
    import runpy
    import sys

    import yaml
    from test_d3 import write_participants

    root = tmp_path / "ds004504"
    write_participants(root, {1: "A", 2: "C"})
    for i in (1, 2):
        f = root / "derivatives" / f"sub-{i:03d}" / "eeg" / f"sub-{i:03d}_task-eyesclosed_eeg.set"
        f.parent.mkdir(parents=True)
        write_set(f, template_xyz_mm(CH), lookup=LOOKUP if i == 1 else "")
    (tmp_path / "processed").mkdir()
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump({"drive_root": str(tmp_path), "processed_dir":
                                                         str(tmp_path / "processed"), "cache_dir": str(tmp_path / "c"),
                                                         "dataset3": {"root": str(root)}}))
    monkeypatch.setattr(bids, "EXPECTED", {"AD": 1, "HC": 1})
    monkeypatch.setattr(sys, "argv", ["d3_data.py", "check", "--paths", str(tmp_path / "paths.yaml")])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(REPO / "scripts" / "d3_data.py"), run_name="__main__")
    rep = pd.read_csv(next((tmp_path / "processed").glob("d3_header_report_*.csv"))).set_index("key")
    assert e.value.code == 1                                        # D3-2: no lookup record -> frame unknown
    assert rep.loc["D3-1", "coord_frame"] == "mri" and rep.loc["D3-1", "coord_template_deviation_mm"] < 1e-6
    assert "cannot be established" in rep.loc["D3-2", "coord_error"]
