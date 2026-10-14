from pathlib import Path

import pytest

from adgraph.experiments.config import expand_variants, hparam_grid, load_experiment

EXP_DIR = Path("configs/experiments")
EXPERIMENTS = sorted(p for p in EXP_DIR.glob("**/*.yaml") if "base" not in p.name)
EXPECTED_COUNTS = {"exp0_sensor_psd": 2, "exp1_source_psd": 2, "exp2_conn_minimal": 24, "exp3_conn_psd": 24,
                   "exp3b_conn_psd_allbands": 24, "exp4_band_mean": 2, "exp5_multiband": 16, "exp6_multiband_psd_strength": 16,
                   "baseline_age": 1, "probe_dataset": 1, "baseline_conn_classical": 4,
                   # fixed-hyperparameter GNN protocol (methodology log entry 42): Tier 1 and Tier 2
                   "exp2_conn_minimal_fixedhp": 8, "exp3_conn_psd_fixedhp": 8, "exp3b_conn_psd_allbands_fixedhp": 8,
                   "exp4_band_mean_fixedhp": 2, "exp5_multiband_fixedhp": 4, "exp6_multiband_psd_strength_fixedhp": 9,
                   "exp2_conn_minimal_fixedhp_tier2": 16, "exp3_conn_psd_fixedhp_tier2": 16,
                   "exp5_multiband_fixedhp_tier2": 12, "exp6_multiband_psd_strength_fixedhp_tier2": 7}


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_config_expands(path):
    cfg = load_experiment(path)
    variants = expand_variants(cfg)
    base_name = cfg["experiment"].removesuffix("_plus_age")
    assert len(variants) == EXPECTED_COUNTS[base_name]
    assert cfg["covariates"] == (["age"] if cfg["experiment"].endswith("_plus_age") else [])
    for v in variants:
        fixed = (v.get("protocol") or {}).get("hyperparameter_search") == "none"
        assert v["evaluation"]["k"] == 5 and v["evaluation"]["n_repeats"] == (10 if fixed else 20)
        if fixed:
            assert hparam_grid(v) == [{}]
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
    assert {v["model"]["encoder"] for v in vs} == {"gcn", "edge_weighted_gat", "weighted_graphsage"}
    assert all(v["node_features"]["components"] == ["constant"] and v["graph"]["band_mode"] == "single" for v in vs)


def test_exp3_primary_uses_matching_band_psd():
    v = expand_variants(load_experiment(EXP_DIR / "exp3_conn_psd.yaml"))[0]
    assert v["node_features"]["psd_bands"] == "matched" and v["node_features"]["psd_kinds"] == ["log", "rel"]
    v = expand_variants(load_experiment(EXP_DIR / "exp3b_conn_psd_allbands.yaml"))[0]
    assert v["node_features"]["psd_bands"] == "all"


def test_age_configs_cover_eeg_experiments():
    names = {p.stem for p in (EXP_DIR / "age").glob("*.yaml")}
    for e in ("exp0_sensor_psd", "exp1_source_psd", "exp2_conn_minimal", "exp3_conn_psd", "exp4_band_mean",
              "exp5_multiband", "exp6_multiband_psd_strength"):
        assert f"{e}_plus_age" in names
