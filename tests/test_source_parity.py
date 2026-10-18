"""LCMV and wMNE through ONE code path (methodology log entries 55, 58): the unified store configuration reproduces
both validated stores, their paths are isolated, every final experiment variant is identical for both inverses
except the inverse, the frozen participant split and folds, and the window-by-window store match. Synthetic
participant tables and arrays only (no EEG, no fsaverage)."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from adgraph.config import config_hash, load_yaml
from adgraph.evaluation.splits import xu_partition
from adgraph.experiments.config import expand_variants, experiment_configs, load_experiment, with_context
from adgraph.sourcegraphs import build, protocol
from adgraph.sourcegraphs.config import KNOWN_HASHES, resolve
from adgraph.sourcegraphs.match import match, require_matched
from adgraph.stores import store_dir

LX_CFG, WX_CFG = resolve("lcmv"), resolve("wmne")


# ------------------------------------------------------------------------------------------- configuration, paths
@pytest.mark.parametrize("inverse,legacy", [("lcmv", "legacy/configs/lcmv_xu/pipeline.yaml"),
                                            ("wmne", "legacy/configs/wmne_xu/pipeline.yaml")])
def test_unified_configuration_reproduces_the_validated_stores(inverse, legacy):
    cfg = resolve(inverse)
    assert cfg == load_yaml(legacy)                                   # the very configuration the store was built with
    assert config_hash(cfg) == KNOWN_HASHES[(inverse, "dk68")] == {"lcmv": "811690218bfa",
                                                                   "wmne": "c32bc6c67f0e"}[inverse]


def test_wmne_config_differs_from_lcmv_only_in_the_inverse():
    diff = {k for k in set(LX_CFG) | set(WX_CFG) if LX_CFG.get(k) != WX_CFG.get(k)}
    assert diff == {"output_subdir", "store_names", "source"}
    for k in ("subject", "spacing", "bem", "mindist_mm", "parcellation", "roi_mode"):
        assert WX_CFG["source"][k] == LX_CFG["source"][k]
    hist = load_yaml("configs/pipeline_d3.yaml")["source"]          # the historical wMNE settings
    for k in ("method", "loose", "depth", "lambda2", "noise_cov", "pick_ori"):
        assert WX_CFG["source"]["wmne"][k] == hist[k]
    assert build.method(WX_CFG) == "wmne" and build.method(LX_CFG) == "lcmv"


def test_store_paths_are_isolated(tmp_path):
    names = {(inv, m, a): store_dir(tmp_path, inv, m, a).relative_to(tmp_path).as_posix()
             for inv in ("lcmv", "wmne") for m in ("B", "A") for a in ("dk68", "r18")}
    assert names[("lcmv", "B", "dk68")] == "D3_LCMV_XU/graphsLX_v1_811690218bfa_corr1"
    assert names[("lcmv", "A", "dk68")] == "D3_LCMV_XU/graphsLX10_v1_811690218bfa_corr1"
    assert names[("wmne", "B", "dk68")] == "D3_WMNE_XU/graphsWX_v1_c32bc6c67f0e"
    assert names[("wmne", "A", "dk68")] == "D3_WMNE_XU/graphsWX10_v1_c32bc6c67f0e"
    assert len(set(names.values())) == len(names)                   # a coarse atlas never shares a 68-ROI store
    assert all("-r18_v1_" in n for k, n in names.items() if k[2] == "r18")


# ------------------------------------------------------------------------------------------- experiments
@pytest.mark.parametrize("exp", sorted(experiment_configs()))
@pytest.mark.parametrize("mode", ["B", "A"])
def test_every_variant_is_identical_for_both_inverses(exp, mode):
    vs = expand_variants(load_experiment(experiment_configs()[exp]))
    for v in vs:
        a, b = with_context(v, "lcmv", mode), with_context(v, "wmne", mode)
        if exp == "exp0":                                             # sensor space: one run, inverse "sensor"
            assert a == b and a["context"]["inverse"] == "sensor"
            continue
        assert {**a, "context": None} == {**b, "context": None}
        assert (a["context"]["inverse"], b["context"]["inverse"]) == ("lcmv", "wmne")


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

    from adgraph.sourcegraphs import correct as corr

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
