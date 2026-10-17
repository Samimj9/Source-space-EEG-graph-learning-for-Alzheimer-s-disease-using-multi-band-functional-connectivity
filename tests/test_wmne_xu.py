"""Matched wMNE / Xu suite (methodology log entry 55): configuration identity with the LCMV suite, isolated paths and
pins, identical experiment definitions and unit counts, the frozen participant split and folds, and the
window-by-window store match. Synthetic participant tables and arrays only (no EEG, no fsaverage)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adgraph.config import config_hash, load_yaml
from adgraph.evaluation.splits import xu_partition
from adgraph.experiments.config import FAMILIES, expand_variants, load_experiment
from adgraph.experiments.plan import experiment_entries, plan_units
from adgraph.lcmvxu import build, protocol
from adgraph.lcmvxu.match import match, require_matched

LX_CFG, WX_CFG = load_yaml("configs/lcmv_xu/pipeline.yaml"), load_yaml("configs/wmne_xu/pipeline.yaml")
EXPECTED_VARIANTS = {"exp0": 4, "exp1": 4, "exp2": 10, "exp3": 10, "exp4": 4, "exp5": 8, "exp6": 10, "exp7": 6,
                     "exp8": 10}
EXPECTED_HIST = {"exp0": 2, "exp1": 2, "exp2": 8, "exp3": 8, "exp4": 2, "exp5": 4, "exp6": 2, "exp7": 6}


# ------------------------------------------------------------------------------------------- configuration, paths
def test_wmne_config_differs_from_lcmv_only_in_the_inverse():
    diff = {k for k in set(LX_CFG) | set(WX_CFG) if LX_CFG.get(k) != WX_CFG.get(k)}
    assert diff == {"output_subdir", "store_names", "source"}
    for k in ("subject", "spacing", "bem", "mindist_mm", "parcellation", "roi_mode"):
        assert WX_CFG["source"][k] == LX_CFG["source"][k]
    hist = load_yaml("configs/pipeline_d3.yaml")["source"]          # the historical wMNE settings
    for k in ("method", "loose", "depth", "lambda2", "noise_cov", "pick_ori"):
        assert WX_CFG["source"]["wmne"][k] == hist[k]
    assert build.method(WX_CFG) == "wmne" and build.method(LX_CFG) == "lcmv"


def test_store_paths_are_isolated_and_lcmv_unchanged(tmp_path):
    assert config_hash(LX_CFG) == "811690218bfa"                     # the running LCMV stores' configuration
    lb, la = build.run_dirs(tmp_path, LX_CFG)
    wb, wa = build.run_dirs(tmp_path, WX_CFG)
    assert (lb.name, la.name) == ("graphsLX_v1_811690218bfa", "graphsLX10_v1_811690218bfa")
    assert lb.parent.name == "D3_LCMV_XU" and wb.parent.name == wa.parent.name == "D3_WMNE_XU"
    assert wb.name.startswith("graphsWX_v1_") and wa.name.startswith("graphsWX10_v1_")
    roots = {f: FAMILIES[f][1] for f in ("LX", "LX10", "WX", "WX10", "D3H")}
    assert len(set(roots.values())) == 5
    assert all(roots[f].startswith("D3_WMNE_XU/") for f in ("WX", "WX10", "D3H"))
    assert FAMILIES["WX"][0] == "D3_WMNE_XU/graphsWX" and FAMILIES["D3H"][0] == "graphsD3"
    setup = Path("scripts/colab_setup.py").read_text()
    assert '"WX": "D3_WMNE_XU/CODE_COMMIT.txt"' in setup and '"LX": "D3_LCMV_XU/CODE_COMMIT.txt"' in setup


def _unit_glob(prefix, names):
    return [n for n in names if Path(n).match(f"{prefix.split('/')[-1]}_v*")]


def test_store_prefixes_never_select_another_family():
    names = ["graphsWX_v1_a", "graphsWX10_v1_a", "graphsLX_v1_b_corr1", "graphsLX10_v1_b_corr1"]
    assert _unit_glob(FAMILIES["WX"][0], names) == ["graphsWX_v1_a"]
    assert _unit_glob(FAMILIES["WX10"][0], names) == ["graphsWX10_v1_a"]


# ------------------------------------------------------------------------------------------- experiments
@pytest.mark.parametrize("wx", sorted(Path("configs/experiments/wx").rglob("*.yaml")), ids=lambda p: p.stem)
def test_matched_experiments_are_the_lcmv_definitions(wx):
    lx = Path(str(wx).replace("/wx/", "/lx/").replace("wx10_", "lx10_").replace("/wx_", "/lx_"))
    a, b = expand_variants(load_experiment(wx)), expand_variants(load_experiment(lx))
    drop = ("experiment", "graph_family", "name", "config")
    assert [{k: v for k, v in x.items() if k not in drop} for x in a] == \
           [{k: v for k, v in y.items() if k not in drop} for y in b]
    units = plan_units(experiment_entries(str(wx)))
    exp = wx.stem.split("_")[1]
    assert len(units) == 2 * EXPECTED_VARIANTS[exp]
    assert {u[3] for u in units} == {"xu_dev_cv", "xu_holdout"}
    assert {u[2]["graph_family"] for u in units} == {"WX10" if "tenfold" in str(wx) else "WX"}


@pytest.mark.parametrize("h", sorted(Path("configs/experiments/wx_hist").glob("*.yaml")), ids=lambda p: p.stem)
def test_historical_experiments_use_the_frozen_protocol(h):
    units = plan_units(experiment_entries(str(h)))
    assert len(units) == 2 * EXPECTED_HIST[h.stem.split("_")[1]]
    ev = units[0][2]["evaluation"]
    assert (ev["n_repeats"], ev["inner_k"], ev["seed"]) == (1, 5, 0)
    assert {u[2]["graph_family"] for u in units} == {"D3H"}


# ------------------------------------------------------------------------------------------- split and folds
def windows_meta(n_windows: dict, labels: dict) -> pd.DataFrame:
    rows = [{"key": k, "dataset": "D3", "y": labels[k], "window_id": f"{k}_w{i}"} for k, n in n_windows.items()
            for i in range(n)]
    return pd.DataFrame(rows)


LABELS = {f"D3-{n}": int(n <= 36) for n in range(1, 66)}           # ds004504: 36 AD, 29 HC
EV = {"k": 5, "n_repeats": 1, "seed": 0}


def saved_split(meta):
    s = meta.groupby("key").agg(dataset=("dataset", "first"), y=("y", "first")).reset_index()
    dev, test = xu_partition(s)
    return {"development": dev, "test": test}


def test_identical_participants_partition_and_folds():
    ref = windows_meta({k: 250 for k in LABELS}, LABELS)
    hist = windows_meta({k: 40 + int(k[3:]) % 7 for k in LABELS}, LABELS)   # other windows, same participants
    res = protocol.compare(hist, ref, saved_split(ref), EV)
    assert res["passed"], res["problems"]
    assert (res["n_participants"], res["n_development"], res["n_holdout"]) == (65, 52, 13)
    folds = res["dev_folds"]
    assert len(folds) == 5 and sorted(sum(folds, [])) == saved_split(ref)["development"]
    assert not set(sum(folds, [])) & set(saved_split(ref)["test"])           # no holdout participant in the CV


def test_protocol_mismatches_are_refused():
    ref = windows_meta({k: 10 for k in LABELS}, LABELS)
    split = saved_split(ref)
    missing = windows_meta({k: 10 for k in list(LABELS)[:-1]}, LABELS)
    assert not protocol.compare(missing, ref, split, EV)["passed"]
    flipped = windows_meta({k: 10 for k in LABELS}, {**LABELS, "D3-1": 0})
    assert any("labels differ" in p for p in protocol.compare(flipped, ref, split, EV)["problems"])
    other_split = {"development": split["development"][1:] + split["test"][:1],
                   "test": split["test"][1:] + split["development"][:1]}
    assert not protocol.compare(ref, ref, other_split, EV)["passed"]
    assert protocol.compare(ref, ref, split, {**EV, "seed": 1})["passed"]   # the seed is applied to both stores


# ------------------------------------------------------------------------------------------- store match
def _store_pair(tmp_path, monkeypatch):
    """Corrected LCMV-style store (reference) and a matched wMNE copy (target) of 10 synthetic participants."""
    import test_lx_correction as t

    from adgraph.lcmvxu import correct as corr

    b, a, headers, _, _ = t.make_store(tmp_path / "orig")
    monkeypatch.setattr(corr, "read_header", lambda path, fmt="edf": headers[path])
    monkeypatch.setattr(corr, "rejection_check", lambda *x: {"threshold_original_uv": 1.0, "decision_flips": [],
                                                             "threshold_corrected_selection_uv": 1.0})
    ref, _, _ = corr.correct(b, a)
    m = json.loads((ref / "manifest.json").read_text())
    m["source_operator"]["method"] = "lcmv"
    (ref / "manifest.json").write_text(json.dumps(m))
    tgt = tmp_path / "graphsWX_v1_x"
    shutil.copytree(ref, tgt)
    m["config"] = {**m["config"], "output_subdir": "D3_WMNE_XU", "store_names": ["graphsWX", "graphsWX10"],
                   "source": {"method": "wmne"}}
    m["source_operator"]["method"], m["graph_run_id"] = "wmne", "wx1"
    (tgt / "manifest.json").write_text(json.dumps(m))
    with np.load(ref / "graphs.npz") as z:
        arrays = {k: z[k] for k in z.files}
    for k in ("psd_log", "psd_rel", "psd_de", "psd_abs"):              # the source side differs (the comparison)
        arrays[k] = arrays[k] * 1.5
    np.savez(tgt / "graphs.npz", **arrays)
    return ref, tgt, arrays


def test_match_passes_for_identical_windows_and_sensor_features(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend("tests")
    ref, tgt, _ = _store_pair(tmp_path, monkeypatch)
    res = match(tgt, ref)
    assert res["passed"], res["checks"]
    (tgt / "match.json").write_text(json.dumps(res, default=str))
    require_matched(tgt, ref)
    assert res["arrays"]["psd_log"]["mean_abs_diff"] > 0                # source arrays differ and are reported


@pytest.mark.parametrize("change", ["window_start", "sensor_feature", "roi_order", "config", "split"])
def test_match_detects_every_difference(tmp_path, monkeypatch, change):
    monkeypatch.syspath_prepend("tests")
    ref, tgt, arrays = _store_pair(tmp_path, monkeypatch)
    if change == "window_start":
        w = pd.read_csv(tgt / "windows.csv")
        w.loc[5, "start_sample"] += 1
        w.to_csv(tgt / "windows.csv", index=False)
    elif change == "sensor_feature":
        arrays["sensor_psd_log"] = arrays["sensor_psd_log"] + 1e-3
        np.savez(tgt / "graphs.npz", **arrays)
    elif change == "roi_order":
        arrays["roi_names"] = arrays["roi_names"][::-1]
        np.savez(tgt / "graphs.npz", **arrays)
    elif change == "config":
        m = json.loads((tgt / "manifest.json").read_text())
        m["config"]["windowing"] = {**m["config"]["windowing"], "step_s": 2.0}
        (tgt / "manifest.json").write_text(json.dumps(m))
    else:
        s = json.loads((tgt / "xu_split.json").read_text())
        s["development"], s["test"] = s["development"][1:], s["test"] + s["development"][:1]
        (tgt / "xu_split.json").write_text(json.dumps(s))
    res = match(tgt, ref)
    assert not res["passed"]
    (tgt / "match.json").write_text(json.dumps(res, default=str))
    with pytest.raises(SystemExit):
        require_matched(tgt, ref)
