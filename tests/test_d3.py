"""Dataset 3 (OpenNeuro ds004504, AD/HC only; methodology log entry 44), synthetic data only.

* participants: groups A -> AD, C -> HC, F (FTD) excluded by design; unknown groups refused;
* download: FTD never fetched, MD5-verified, resumable (present files skipped, corrupt ones re-fetched);
* preprocessing: no second band-pass, flagged channels reported but not interpolated, average reference;
* boundary margin = analytic-kernel half only (412 samples);
* graph build writes a graphsD3_v4_* run that D1/D2 selection never picks; FTD recorded as excluded;
* experiments: within_D3 setting, graph_family D3 -> graphsD3 runs and experiments_D3 results.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from adgraph.config import load_yaml
from adgraph.io import bids

mne = pytest.importorskip("mne")
REPO = Path(__file__).resolve().parents[1]
CFG = load_yaml(REPO / "configs" / "pipeline_d3.yaml")
CH = CFG["channels"]["recorded"]


def write_participants(root: Path, groups: dict) -> pd.DataFrame:
    rows = [{"participant_id": f"sub-{i:03d}", "Gender": "F" if i % 2 else "M", "Age": 60 + i, "Group": g,
             "MMSE": 20 + i % 10} for i, g in groups.items()]
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "participants.tsv", sep="\t", index=False)
    return pd.DataFrame(rows)


def test_participants_groups_and_exclusion(tmp_path):
    write_participants(tmp_path, {1: "A", 2: "C", 3: "F", 4: "A"})
    kept, excl = bids.load_participants_tsv(tmp_path / "participants.tsv")
    assert kept[["key", "label", "y"]].values.tolist() == [["D3-1", "AD", 1], ["D3-2", "HC", 0], ["D3-4", "AD", 1]]
    assert excl["key"].tolist() == ["D3-3"] and excl["group"].tolist() == ["F"]
    with pytest.raises(ValueError, match="cohort is"):
        bids.check_cohort(kept)                               # release cohort is 36 AD / 29 HC
    bids.check_cohort(kept, {"AD": 2, "HC": 1})
    write_participants(tmp_path, {1: "A", 2: "X"})
    with pytest.raises(ValueError, match="unknown Group"):
        bids.load_participants_tsv(tmp_path / "participants.tsv")


def test_download_skips_ftd_verifies_and_resumes(tmp_path, monkeypatch):
    from adgraph.io import openneuro

    mirror = tmp_path / "mirror" / "ds004504"
    write_participants(mirror, {1: "A", 2: "C", 3: "F"})
    (mirror / "dataset_description.json").write_text(json.dumps({"DatasetDOI": "doi:x", "License": "CC0"}))
    (mirror / "CHANGES").write_text("1.0.9 2026-05-20\n")
    for i in (1, 2, 3):
        f = mirror / "derivatives" / f"sub-{i:03d}" / "eeg" / f"sub-{i:03d}_task-eyesclosed_eeg.set"
        f.parent.mkdir(parents=True)
        f.write_bytes(bytes([i]) * 1000)
    listing = {f"ds004504/{p.relative_to(mirror).as_posix()}": {"size": p.stat().st_size,
                                                               "etag": hashlib.md5(p.read_bytes()).hexdigest()}
               for p in mirror.rglob("*") if p.is_file()}
    monkeypatch.setattr(openneuro, "list_objects", lambda prefix, base_url=None: listing)
    monkeypatch.setattr(bids, "EXPECTED", {"AD": 1, "HC": 1})
    base = (tmp_path / "mirror").as_uri()
    root = tmp_path / "d3"
    m = openneuro.download_ds004504(root, base)
    sets = sorted(p.name for p in root.rglob("*.set"))
    assert sets == ["sub-001_task-eyesclosed_eeg.set", "sub-002_task-eyesclosed_eeg.set"]   # never FTD
    assert m["cohort"]["excluded"][0]["key"] == "D3-3" and m["release"] == "1.0.9 2026-05-20"
    target = root / "derivatives" / "sub-001" / "eeg" / "sub-001_task-eyesclosed_eeg.set"
    target.write_bytes(b"\x00" * 1000)                         # same size, wrong content -> re-fetched
    m2 = openneuro.download_ds004504(root, base)
    status = {Path(f["path"]).name: f["status"] for f in m2["files"]}
    assert status["sub-001_task-eyesclosed_eeg.set"] == "downloaded"
    assert status["sub-002_task-eyesclosed_eeg.set"] == "present"
    assert target.read_bytes() == bytes([1]) * 1000


def synthetic_raw(seed=0, dur=150, sf=500, noisy=None):
    rng = np.random.default_rng(seed)
    t = np.arange(dur * sf) / sf
    x = (np.sin(2 * np.pi * 10 * t) + rng.standard_normal((len(CH), t.size))) * 10e-6
    if noisy is not None:
        x[noisy] *= 300
    return mne.io.RawArray(x, mne.create_info(CH, sf, "eeg"), verbose="error")


def test_preprocessing_without_bandpass_or_interpolation():
    from adgraph.io.eeg import make_montage
    from adgraph.preprocessing.boundary import crop_edges, expected_valid_windows, margins
    from adgraph.preprocessing.clean import preprocess

    raw = synthetic_raw(noisy=3)
    raw.set_montage(make_montage("standard_1020"))
    crop_edges(raw, CFG)
    ref = raw.copy().resample(250, verbose="error", pad="reflect_limited")
    ref.set_eeg_reference("average", projection=True, verbose="error").apply_proj(verbose="error")
    out, qc = preprocess(raw, CFG)
    assert np.allclose(out.get_data(), ref.get_data())        # resample + average reference only: no filter
    assert list(qc["bad_channels"]) == ["F4"] and not qc["channels"]["interpolated"].any()
    assert out.info["bads"] == []
    m = margins(CFG)
    assert m["sensor_filter_half_samples"] == 0 and m["margin_samples"] == 412
    assert expected_valid_windows(150, CFG) == 11


def test_graph_build_family_and_exclusion(tmp_path, monkeypatch):
    import adgraph.io.eeg as eeg
    import adgraph.pipeline as pl
    from adgraph.graphs.store import GraphStore

    root = tmp_path / "ds004504"
    write_participants(root, {1: "A", 2: "C", 3: "F", 4: "A"})
    for i in (1, 2, 3, 4):
        f = root / "derivatives" / f"sub-{i:03d}" / "eeg" / f"sub-{i:03d}_task-eyesclosed_eeg.set"
        f.parent.mkdir(parents=True)
        f.write_text(str(i))
    (root / "dataset_description.json").write_text("{}")
    monkeypatch.setattr(bids, "EXPECTED", {"AD": 2, "HC": 1})
    monkeypatch.setattr(eeg, "read_header", lambda path, fmt: synthetic_raw(int(Path(path).read_text())))
    W = np.random.default_rng(2).standard_normal((68, 19))
    monkeypatch.setattr(pl, "fetch_fsaverage", lambda d: None)
    monkeypatch.setattr(pl, "build_roi_operator", lambda info, cfg, d: {
        "W": W, "roi_names": [f"r{i}" for i in range(68)], "ch_names": list(info["ch_names"]), "n_sources": 20484,
        "fwd_ch_names": list(info["ch_names"])})
    monkeypatch.setattr(pl, "verify_operator", lambda *a, **k: 0.0)
    processed = tmp_path / "processed"
    out = pl.run({"dataset3": {"root": str(root)}}, CFG, processed, tmp_path / "subjects", allow_ephemeral=True)
    assert out.name.startswith("graphsD3_v4_")
    store = GraphStore.load(out)
    assert store.subjects["key"].tolist() == ["D3-1", "D3-2", "D3-4"] and (store.subjects["status"] == "ok").all()
    assert store.arrays["sensor_psd_log"].shape[1:] == (4, 19) and store.arrays["plv"].shape[1:] == (4, 68, 68)
    m = store.manifest
    assert m["registry"]["excluded_by_design"]["D3"][0]["key"] == "D3-3"
    assert m["boundary_processing"]["margin_samples"] == 412 and not m["boundary_processing"]["sensor_bandpass_applied"]
    assert set(m["auxiliary_inputs"]) == {"participants_D3", "dataset_description_D3"}
    assert m["summary"]["per_dataset"]["D3"]["subjects"] == 3
    assert pl.latest_complete_graph_run(processed, "graphsD3") == out
    with pytest.raises(FileNotFoundError):
        pl.latest_complete_graph_run(processed)              # D1/D2 selection never sees a D3 run


def test_d3_settings_and_configs(tmp_path):
    from adgraph.experiments.config import expand_variants, load_experiment
    from adgraph.experiments.runner import run_variant
    from adgraph.experiments.config import set_dotted
    from adgraph.synthetic import make_synthetic_store

    v = expand_variants(load_experiment(REPO / "configs/experiments/d3/d3_exp1_source_psd.yaml"))[1]
    bad = dict(v, settings=["within_D3", "within_D1"])
    with pytest.raises(ValueError, match="never mixed"):
        from adgraph.experiments.config import validate
        validate(bad)
    store = make_synthetic_store({"D3": (5, 5)}, windows_per_subject=(3, 3), psd_effect=0.8, seed=3)
    for k, val in {"evaluation.n_repeats": 2, "evaluation.k": 3, "evaluation.inner_k": 2,
                   "evaluation.n_bootstrap": 20}.items():
        v = set_dotted(v, k, val)
    s = run_variant(store, v, "within_D3", tmp_path)
    assert s["n_subjects"] == 10 and "roc_auc" in s["subject_level"]["all"]


def test_experiment_cli_uses_d3_family(tmp_path):
    from adgraph.synthetic import make_synthetic_store

    processed = tmp_path / "processed"
    for name, cohort in (("graphs_v4_synth", {"D1": (4, 4), "D2": (4, 5)}), ("graphsD3_v4_synth", {"D3": (5, 5)})):
        g = processed / name
        make_synthetic_store(cohort, windows_per_subject=(3, 3), psd_effect=0.6).save(g)
        (g / "assembly.done.json").write_text("{}")
        man = json.loads((g / "manifest.json").read_text())
        man["assembled_at"] = "2026-01-01T00:00:00+00:00"
        (g / "manifest.json").write_text(json.dumps(man))
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump({"drive_root": str(tmp_path), "processed_dir": str(processed),
                                                         "cache_dir": str(processed / "cache")}))
    cfg = tmp_path / "age.yaml"
    cfg.write_text(yaml.safe_dump({"extends": str(REPO / "configs/experiments/d3/d3_baseline_age.yaml"),
                                   "evaluation": {"n_repeats": 1, "k": 3, "inner_k": 2, "n_bootstrap": 20}}))
    r = subprocess.run([sys.executable, "scripts/experiment.py", "run", "--config", str(cfg), "--paths",
                        str(tmp_path / "paths.yaml"), "--allow-ephemeral-storage"], capture_output=True, text=True,
                       cwd=REPO)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2000:]
    assert "graphsD3_v4_synth" in r.stdout
    assert list((processed / "experiments_D3" / "d3_baseline_age").rglob("final.done.json"))
    assert not (processed / "experiments").exists()


# ---------------------------------------------------------------- flat-channel stop and 19-channel inspection
def test_confirmed_flat_channel_stops_d3_but_low_amplitude_eeg_passes():
    from adgraph.io.eeg import make_montage
    from adgraph.preprocessing.boundary import crop_edges
    from adgraph.preprocessing.clean import FlatChannelError, preprocess

    raw = synthetic_raw()
    raw._data[CH.index("Cz")] = 0.0                                         # flatlined electrode
    raw.set_montage(make_montage("standard_1020"))
    crop_edges(raw, CFG)
    with pytest.raises(FlatChannelError, match=r"flat channel\(s\) \['Cz'\]"):
        preprocess(raw, CFG)
    low = synthetic_raw(seed=4)
    low._data[CH.index("Cz")] *= 0.1                                        # ~1 uV: low but physiological
    low.set_montage(make_montage("standard_1020"))
    crop_edges(low, CFG)
    _, qc = preprocess(low, CFG)
    assert not any(str(r).startswith("flat") for r in qc["bad_channels"].values())


def test_flat_channel_is_still_interpolated_for_d1d2():
    from adgraph.io.eeg import make_montage
    from adgraph.preprocessing.boundary import crop_edges
    from adgraph.preprocessing.clean import preprocess

    cfg = load_yaml(REPO / "configs" / "pipeline.yaml")                     # frozen D1/D2 behaviour unchanged
    names = [c for c in cfg["channels"]["recorded"] if c not in cfg["channels"]["drop"]]
    rng = np.random.default_rng(0)
    t = np.arange(150 * 500) / 500
    x = (np.sin(2 * np.pi * 10 * t) + rng.standard_normal((len(names), t.size))) * 10e-6
    x[names.index("Cz")] = 0.0
    raw = mne.io.RawArray(x, mne.create_info(names, 500, "eeg"), verbose="error")
    raw.set_montage(make_montage("standard_1020"))
    crop_edges(raw, cfg)
    out, qc = preprocess(raw, cfg)
    assert qc["bad_channels"]["Cz"] == "flat" and bool(qc["channels"].set_index("channel").loc["Cz", "interpolated"])
    assert np.std(out.get_data(picks=["Cz"])) > 1e-7


def _d3_run(tmp_path, monkeypatch, flat_subject=None, limit=None):
    import adgraph.io.eeg as eeg
    import adgraph.pipeline as pl

    root = tmp_path / "ds004504"
    write_participants(root, {1: "A", 2: "C", 3: "F", 4: "A"})
    for i in (1, 2, 3, 4):
        f = root / "derivatives" / f"sub-{i:03d}" / "eeg" / f"sub-{i:03d}_task-eyesclosed_eeg.set"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(str(i))
    (root / "dataset_description.json").write_text("{}")
    monkeypatch.setattr(bids, "EXPECTED", {"AD": 2, "HC": 1})

    def header(path, fmt):
        i = int(Path(path).read_text())
        raw = synthetic_raw(i)
        if i == flat_subject:
            raw._data[CH.index("O2")] = 0.0
        return raw
    monkeypatch.setattr(eeg, "read_header", header)
    W = np.random.default_rng(2).standard_normal((68, 19))
    monkeypatch.setattr(pl, "fetch_fsaverage", lambda d: None)
    monkeypatch.setattr(pl, "build_roi_operator", lambda info, cfg, d: {
        "W": W, "roi_names": [f"r{i}" for i in range(68)], "ch_names": list(info["ch_names"]), "n_sources": 20484,
        "fwd_ch_names": list(info["ch_names"])})
    monkeypatch.setattr(pl, "verify_operator", lambda *a, **k: 1e-12)
    paths = {"dataset3": {"root": str(root)}}
    return pl, paths, pl.run(paths, CFG, tmp_path / "processed", tmp_path / "subjects", limit=limit,
                             allow_ephemeral=True)


def test_flat_subject_fails_with_subject_and_channel_named(tmp_path, monkeypatch):
    import adgraph.pipeline as pl

    with pytest.raises(pl.GraphRunIncomplete):
        pl_, paths, _ = _d3_run(tmp_path, monkeypatch, flat_subject=2)
    failed = json.loads(next((tmp_path / "processed").glob("graphsD3_v4_*/subjects/D3-2/failed.json")).read_text())
    assert "subject D3-2" in failed["status"] and "['O2']" in failed["status"]
    done = list((tmp_path / "processed").glob("graphsD3_v4_*/subjects/D3-1/subject.done.json"))
    assert done                                                             # other subjects unaffected


def test_inspection_uses_active_channel_count_and_smoke_checks(tmp_path, monkeypatch):
    import runpy

    _, paths, run = _d3_run(tmp_path, monkeypatch, limit=1)
    unit = run / "subjects" / "D3-1"
    z = dict(np.load(unit / "arrays.npz"))
    np.savez(unit / "arrays.npz", **{k: v[:0] for k, v in z.items()})      # a subject without kept windows
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump({"drive_root": str(tmp_path), "processed_dir":
                                                         str(tmp_path / "processed"), "cache_dir": str(tmp_path / "c"),
                                                         "dataset3": paths["dataset3"]}))
    r = subprocess.run([sys.executable, "scripts/inspect_graph_subjects.py", "--paths", str(tmp_path / "paths.yaml"),
                        "--config", "configs/pipeline_d3.yaml", "--run-dir", str(run)], capture_output=True, text=True,
                       cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg"})
    assert r.returncode == 0, r.stderr[-2000:]
    # explicit-selection smoke checks: frozen D1/D2 manifest by graph_run_id, D3 operator from the smoke manifest
    ref = tmp_path / "processed" / "graphs_v4_frozen"
    ref.mkdir()
    (ref / "manifest.json").write_text(json.dumps({"graph_run_id": "frozen123", "config": {"version": 4},
                                                   "assembled_at": "x",
                                                   "source_operator": {"roi_names": [f"r{i}" for i in range(68)]}}))
    (ref / "assembly.done.json").write_text("{}")
    pd.DataFrame({"key": ["D3-1", "D3-2", "D3-4"], "n_boundary_events": [0, 0, 0]}).to_csv(
        tmp_path / "processed" / "d3_header_report_20260101_000000.csv", index=False)

    def verify(extra=()):
        monkeypatch.setattr(sys, "argv", ["d3_verify_smoke.py", "--paths", str(tmp_path / "paths.yaml"),
                                          "--d1d2-graph-run-id", "frozen123", *extra])
        with pytest.raises(SystemExit) as e:
            runpy.run_path(str(REPO / "scripts" / "d3_verify_smoke.py"), run_name="__main__")
        return e.value.code, json.loads((run / "d3_montage_checks.json").read_text())["checks"]

    code, checks = verify()
    assert code == 0 and all(c["passed"] for c in checks) and len(checks) == 13
    assert "(5) ZERO accepted windows crossing a discontinuity or its margin" in [c["check"] for c in checks]
    m = json.loads((ref / "manifest.json").read_text())
    m["source_operator"]["roi_names"] = m["source_operator"]["roi_names"][::-1]          # different ROI order
    (ref / "manifest.json").write_text(json.dumps(m))
    code, checks = verify()
    assert code == 1 and [c["check"] for c in checks if not c["passed"]] == [
        "(4) 68 ROI names and order identical to the frozen D1/D2 run"]
