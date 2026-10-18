"""Fixed-hyperparameter GNN protocol (methodology log entry 42, configs/experiments/gnn_fixed_base.yaml).

* every fixed-protocol config defines exactly one hyperparameter configuration, and a grid is refused;
* the Tier 1 schedule (variants, settings, repeats, encoder) is exactly the approved one;
* a run trains exactly ONE model per outer fold (no inner CV, no inner checkpoints), on training
  subjects only, and predicts each test subject's windows once per repeat;
* repeats r use the same subject splits as the nested-CV / classical experiments (paired comparisons).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from adgraph.evaluation.splits import make_splits
from adgraph.experiments.config import expand_variants, hparam_grid, load_experiment, set_dotted
from adgraph.experiments.plan import experiment_entries, plan_units
from adgraph.experiments.runner import run_variant
from adgraph.legacy.synthetic import make_synthetic_store

EXP = Path("configs/experiments")
S124 = ["within_D1", "within_D2", "cross_D1_to_D2", "cross_D2_to_D1"]
ALL = ["within_D1", "within_D2", "pooled", "cross_D1_to_D2", "cross_D2_to_D1"]
FIXED = {"hidden": 32, "n_layers": 2, "dropout": 0.3, "readout": "mean_max", "mlp_hidden": 32}
TRAIN = {"lr": 0.001, "weight_decay": 0.0001, "epochs": 60, "batch_size": 32}


def units(cfg):
    return [(u[2]["variant"], u[3]) for u in plan_units(experiment_entries(EXP / cfg))]


def test_fixed_configs_have_no_search():
    paths = sorted(EXP.glob("*fixedhp*.yaml")) + sorted((EXP / "age").glob("*fixedhp*.yaml"))
    assert len(paths) == 15
    for p in paths:
        for v in expand_variants(load_experiment(p)):
            assert v["protocol"] == {"name": "gnn_fixed_hparams_v1", "hyperparameter_search": "none"}
            assert hparam_grid(v) == [{}]
            assert {k: v["model"][k] for k in FIXED} == FIXED
            assert {k: v["training"][k] for k in TRAIN} == TRAIN
            assert v["training"]["early_stopping"]["enabled"] is False
            assert v["evaluation"]["n_repeats"] == 10
            assert set(v["settings"]) <= set(ALL)


def test_grid_is_refused_under_fixed_protocol(tmp_path):
    cfg = load_experiment(EXP / "exp2_conn_minimal_fixedhp.yaml")
    cfg["hparams"] = {"model.hidden": [16, 32]}
    with pytest.raises(ValueError, match="forbids hyperparameter search"):
        expand_variants(cfg)


def test_tier1_schedule():
    for cfg in ("exp2_conn_minimal_fixedhp.yaml", "exp3_conn_psd_fixedhp.yaml"):
        vs = expand_variants(load_experiment(EXP / cfg))
        assert {(v["graph"]["connectivity"], v["graph"]["bands"][0]) for v in vs} == {
            (c, b) for c in ("plv", "ciplv") for b in ("theta", "alpha1", "alpha2", "beta")}
        assert {v["model"]["encoder"] for v in vs} == {"gcn"}
        assert len(units(cfg)) == 8 * 4 and {s for _, s in units(cfg)} == set(S124)
    assert len(units("exp4_band_mean_fixedhp.yaml")) == 2 * 5
    v5 = expand_variants(load_experiment(EXP / "exp5_multiband_fixedhp.yaml"))
    assert {(v["graph"]["connectivity"], v["model"]["fusion"], v["model"]["shared_encoder"]) for v in v5} == {
        (c, f, True) for c in ("plv", "ciplv") for f in ("mean", "attention")}
    assert len(units("exp5_multiband_fixedhp.yaml")) == 4 * 5
    u6 = units("exp6_multiband_psd_strength_fixedhp.yaml")
    assert sorted(s for v, s in u6 if v == "plv__full") == sorted(ALL)
    assert sorted(s for v, s in u6 if v == "ciplv__full") == sorted(ALL)
    ablations = {v for v, _ in u6} - {"plv__full", "ciplv__full"}
    assert ablations == {f"plv__{a}" for a in ("no_psd", "no_strength", "no_learned_fusion", "drop_theta",
                                                "drop_alpha1", "drop_alpha2", "drop_beta")}
    assert all(s in S124 for v, s in u6 if v in ablations) and len(u6) == 2 * 5 + 7 * 4


@pytest.fixture(scope="module")
def store():
    return make_synthetic_store({"D1": (5, 5), "D2": (5, 6)}, windows_per_subject=(3, 4), psd_effect=0.8,
                                conn_effect=0.8, seed=1)


@pytest.mark.parametrize("cfg", ["exp2_conn_minimal_fixedhp.yaml", "exp6_multiband_psd_strength_fixedhp.yaml"])
def test_one_fit_per_outer_fold_and_no_leakage(store, tmp_path, monkeypatch, cfg):
    pytest.importorskip("torch")
    from adgraph.models import trainer

    fitted = []
    real_fit = trainer.GNNModel.fit

    def counting_fit(self, view):
        fitted.append(set(view.meta["key"]))
        return real_fit(self, view)

    monkeypatch.setattr(trainer.GNNModel, "fit", counting_fit)
    v = expand_variants(load_experiment(EXP / cfg))[0]
    for k, val in {"evaluation.n_repeats": 2, "evaluation.k": 3, "evaluation.n_bootstrap": 50,
                   "training.epochs": 2}.items():
        v = set_dotted(v, k, val)
    run_variant(store, v, "within_D1", tmp_path)
    out = tmp_path / v["experiment"] / v["variant"] / "within_D1"
    folds = pd.read_csv(out / "folds.csv")
    pred = pd.read_csv(out / "predictions_windows.csv")
    hp = pd.read_csv(out / "hparams.csv")
    assert len(fitted) == 2 * 3 == len(hp)                       # one fit per outer fold, nothing else
    assert not list((out / "units").glob("inner_*"))             # no inner-CV trials
    assert (hp["selected"] == "{}").all() and (hp["scores"] == "[]").all()
    for (r, f), g in folds.groupby(["repeat", "fold"]):
        train, test = set(g.loc[g.side == "train", "key"]), set(g.loc[g.side == "test", "key"])
        assert not train & test
        assert fitted[r * 3 + f] == train                        # the model saw training subjects only
        assert set(pred.query("repeat == @r and fold == @f")["key"]) == test
    assert not pred.duplicated(["repeat", "window_id"]).any()
    # repeat r uses the subject splits of every other experiment's repeat r (same seed + r)
    meta = pd.read_csv(out / "folds.csv").query("repeat == 0 and side == 'test'")
    ref = make_splits(store.windows.query("dataset == 'D1'").reset_index(drop=True), "within_D1", 3, 1, 0, "label")
    assert {frozenset(sp.test_keys) for sp in ref} == {frozenset(g["key"]) for _, g in meta.groupby("fold")}
    assert json.loads((out / "manifest.json").read_text())["variant_config"]["protocol"]["name"] == \
        "gnn_fixed_hparams_v1"
