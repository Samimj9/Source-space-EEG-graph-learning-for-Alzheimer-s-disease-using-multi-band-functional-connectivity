from pathlib import Path

import pytest

from adgraph.experiments.config import expand_variants, hparam_grid, load_experiment

EXP_DIR = Path("configs/experiments")
EXPERIMENTS = sorted(p for p in EXP_DIR.glob("*.yaml") if "base" not in p.name)
EXPECTED_COUNTS = {"exp0_sensor_psd": 2, "exp1_source_psd": 2, "exp2_conn_minimal": 24, "exp3_conn_psd": 24,
                   "exp4_band_mean": 2, "exp5_multiband": 16, "exp6_multiband_psd_strength": 16,
                   "baseline_age": 1, "probe_dataset": 1, "baseline_conn_classical": 4}


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_config_expands(path):
    cfg = load_experiment(path)
    variants = expand_variants(cfg)
    assert len(variants) == EXPECTED_COUNTS[cfg["experiment"]]
    for v in variants:
        assert v["evaluation"]["k"] == 5 and v["evaluation"]["n_repeats"] == 20
        assert v["evaluation"]["threshold"] == 0.5
        if v["input"] == "graph":
            assert v["graph"]["connectivity"] in ("plv", "ciplv")
        assert hparam_grid(v)


def test_plv_and_ciplv_are_separate_variants():
    vs = expand_variants(load_experiment(EXP_DIR / "exp5_multiband.yaml"))
    plv = {v["name"].replace("plv", "X") for v in vs if v["graph"]["connectivity"] == "plv"}
    ci = {v["name"].replace("ciplv", "X") for v in vs if v["graph"]["connectivity"] == "ciplv"}
    assert plv == ci and len(plv) == 8


def test_exp6_ablations():
    vs = {v["variant"]: v for v in expand_variants(load_experiment(EXP_DIR / "exp6_multiband_psd_strength.yaml"))}
    assert vs["plv__full"]["node_features"]["components"] == ["psd", "strength"]
    assert vs["plv__full"]["model"]["fusion"] == "attention"
    assert vs["plv__no_psd"]["node_features"]["components"] == ["strength"]
    assert vs["ciplv__no_strength"]["node_features"]["components"] == ["psd"]
    assert vs["plv__no_learned_fusion"]["model"]["fusion"] == "mean"
    assert vs["plv__drop_theta"]["graph"]["bands"] == ["alpha1", "alpha2", "beta"]


def test_variant_names_are_readable():
    names = [v["variant"] for v in expand_variants(load_experiment(EXP_DIR / "exp5_multiband.yaml"))]
    assert "plv__shared_encoder=True__attention" in names and "ciplv__shared_encoder=False__gate" in names


def test_exp2_uses_constant_node_features_and_three_encoders():
    vs = expand_variants(load_experiment(EXP_DIR / "exp2_conn_minimal.yaml"))
    assert {v["model"]["encoder"] for v in vs} == {"gcn", "gat", "graphsage"}
    assert all(v["node_features"]["components"] == ["constant"] and v["graph"]["band_mode"] == "single" for v in vs)
