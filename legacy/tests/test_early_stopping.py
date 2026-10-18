"""200-epoch / patience-10 early-stopping protocol (methodology log entry 56): stopping logic, best-checkpoint
restoration, subject-disjoint inner validation drawn only from outer training participants (never the outer CV fold
or the 13 holdout participants), separate versioned results, and identical LCMV / wMNE protocols.
Synthetic data only."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adgraph.checkpoint import IncompatibleCheckpoint
from adgraph.evaluation.splits import xu_partition
from adgraph.experiments.config import expand_variants, load_experiment
from adgraph.graphs.dataset import build_graph
from adgraph.models import trainer as tr
from adgraph.legacy.synthetic import make_synthetic_store

ES_CFG = {"epochs": 200, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4,
          "early_stopping": {"enabled": True, "patience": 10, "min_delta": 1e-4, "val_subject_fraction": 0.2,
                             "restore_best": True, "refit": False}}
MODEL = {"encoder": "gcn", "hidden": 8, "n_layers": 1, "fusion": "mean", "readout": "mean_max", "mlp_hidden": 8}


@pytest.fixture(scope="module")
def d3():
    """65 participants as in ds004504 (36 AD, 29 HC), a few small graphs each."""
    return make_synthetic_store(cohort={"D3": (36, 29)}, windows_per_subject=(3, 4), n_rois=8, n_channels=6,
                                psd_effect=0.5, conn_effect=0.5, seed=1)


def _data(store):
    return build_graph(store, {"connectivity": "plv", "bands": ["theta"], "band_mode": "single"},
                       {"components": ["constant"]})


def _scripted(monkeypatch, losses):
    """Validation loss sequence per epoch; snapshot of the weights at every epoch."""
    snaps, it = [], iter(losses)

    def fake(self, view):
        snaps.append(copy.deepcopy(self.net.state_dict()))
        return float(next(it))

    monkeypatch.setattr(tr.GNNModel, "_weighted_loss", fake)
    return snaps


def test_stops_after_patience_and_restores_best(d3, monkeypatch):
    data = _data(d3)
    # improves until epoch 6, then a tiny "improvement" below min_delta and noise: stop at 6 + 10 = 16
    losses = [0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.65 - 5e-5] + [0.66, 0.67] * 20
    snaps = _scripted(monkeypatch, losses)
    m = tr.GNNModel(MODEL, ES_CFG, seed=3).fit(data.view(np.arange(len(data.meta)), "train"))
    s = m.training_summary()
    assert (s["best_epoch"], s["epochs_run"], s["stopped_early"]) == (6, 16, True)
    assert s["best_val_loss"] == pytest.approx(0.65) and s["patience"] == 10 and s["min_delta"] == 1e-4
    best = snaps[s["best_epoch"] - 1]
    assert all(np.array_equal(v.cpu().numpy(), best[k].cpu().numpy()) for k, v in m.net.state_dict().items())
    assert len(json.loads(s["history_val_loss"])) == len(json.loads(s["history_train_loss"])) == 16


def test_runs_to_the_maximum_when_always_improving(d3, monkeypatch):
    _scripted(monkeypatch, [1.0 - 1e-3 * i for i in range(200)])
    m = tr.GNNModel(MODEL, ES_CFG, seed=0).fit(_data(d3).view(np.arange(len(d3.windows)), "train"))
    s = m.training_summary()
    assert (s["epochs_run"], s["best_epoch"], s["stopped_early"]) == (200, 200, False)


def test_inner_validation_is_subject_disjoint_and_inside_the_training_view(d3):
    data = _data(d3)
    keys = data.meta["key"].to_numpy()
    train_keys = sorted(set(keys))[:50]                       # an outer training set
    idx = np.flatnonzero(np.isin(keys, train_keys))
    m = tr.GNNModel(MODEL, {**ES_CFG, "epochs": 15}, seed=5).fit(data.view(idx, "train"))
    val = set(json.loads(m.training_summary()["inner_val_keys"]))
    assert val and val <= set(train_keys)
    assert m.es_info["n_inner_fit_participants"] + m.es_info["n_inner_val_participants"] == 50
    yv = data.meta.drop_duplicates("key").set_index("key").loc[sorted(val), "y"]
    assert set(yv) == {0, 1}                                    # both classes, whole participants


def test_refit_trains_on_all_training_participants_for_the_selected_epochs(d3, monkeypatch):
    _scripted(monkeypatch, [0.7, 0.6, 0.5] + [0.6] * 30)
    cfg = copy.deepcopy(ES_CFG)
    cfg["early_stopping"]["refit"] = True
    m = tr.GNNModel(MODEL, cfg, seed=0).fit(_data(d3).view(np.arange(len(d3.windows)), "train"))
    s = m.training_summary()
    assert (s["best_epoch"], s["epochs_run"], s["refit_epochs"]) == (3, 13, 3)   # stopping run, then the refit
    assert len(m.history) == 3                                  # the final model trained 3 epochs on all participants
    assert len(json.loads(s["history_val_loss"])) == 13         # the selection curve is kept


def test_disabled_early_stopping_is_unchanged(d3):
    """The fixed 60-epoch protocol: no validation split, all epochs, same result as before (seeded)."""
    data = _data(d3)
    view = data.view(np.arange(len(data.meta)), "train")
    cfg = {"epochs": 3, "batch_size": 32, "early_stopping": {"enabled": False}}
    a = tr.GNNModel(MODEL, cfg, seed=0).fit(view)
    b = tr.GNNModel(MODEL, cfg, seed=0).fit(view)
    assert a.training_summary()["epochs_run"] == 3 and "best_epoch" not in a.training_summary()
    test = data.view(np.arange(len(data.meta)), "test")
    assert np.allclose(a.predict_proba(test), b.predict_proba(test))


# ------------------------------------------------------------------------------------------- runner, Xu settings
def _variant(path, name="plv__theta", **training):
    v = next(x for x in expand_variants(load_experiment(path)) if x["variant"] == name)
    v["training"] = {**v["training"], **training}
    return v


def test_outer_test_and_holdout_participants_never_used_for_stopping(d3, tmp_path):
    from adgraph.experiments.runner import run_variant

    v = _variant("configs/experiments/lx_es/tenfold/lx10_es_exp2_conn_minimal.yaml")
    assert (v["training"]["epochs"], v["training"]["early_stopping"]["patience"]) == (200, 10)
    subj = d3.windows.drop_duplicates("key")[["key", "dataset", "y"]]
    dev, holdout = xu_partition(subj)
    for setting in ("xu_dev_cv", "xu_holdout"):
        run_variant(d3, v, setting, tmp_path, allow_ephemeral=True)
        d = tmp_path / v["experiment"] / v["variant"] / setting
        hp, folds = pd.read_csv(d / "hparams.csv"), pd.read_csv(d / "folds.csv")
        assert len(hp) == (5 if setting == "xu_dev_cv" else 1)
        for row in hp.itertuples():
            f = folds[(folds.repeat == row.repeat) & (folds.fold == row.fold)]
            train, test = set(f[f.side == "train"].key), set(f[f.side == "test"].key)
            inner = set(json.loads(row.inner_val_keys))
            assert inner and inner <= train and not inner & test           # outer CV fold never used
            assert not inner & set(holdout)                                 # holdout never used
            assert row.epochs_run <= 200 and row.best_epoch <= row.epochs_run
            if row.stopped_early:
                assert row.epochs_run - row.best_epoch == 10
        if setting == "xu_holdout":
            assert set(f[f.side == "train"].key) == set(dev) and set(f[f.side == "test"].key) == set(holdout)


def test_protocols_never_share_results(d3, tmp_path):
    from adgraph.experiments.runner import run_variant

    es = _variant("configs/experiments/lx_es/tenfold/lx10_es_exp2_conn_minimal.yaml", epochs=30)
    fixed = _variant("configs/experiments/lx/tenfold/lx10_exp2_conn_minimal.yaml", epochs=3)
    run_variant(d3, fixed, "xu_holdout", tmp_path, allow_ephemeral=True)
    run_variant(d3, es, "xu_holdout", tmp_path, allow_ephemeral=True)
    assert es["experiment"] != fixed["experiment"]                       # separate result folders
    assert (tmp_path / fixed["experiment"]).exists() and (tmp_path / es["experiment"]).exists()
    mixed = copy.deepcopy(es)
    mixed["experiment"], mixed["name"] = fixed["experiment"], fixed["name"]
    with pytest.raises(IncompatibleCheckpoint):                          # resuming into the other protocol's folder
        run_variant(d3, mixed, "xu_holdout", tmp_path, allow_ephemeral=True)
    again = run_variant(d3, es, "xu_holdout", tmp_path, allow_ephemeral=True)   # resume: reused, not recomputed
    assert again is not None


# ------------------------------------------------------------------------------------------- LCMV == wMNE protocol
@pytest.mark.parametrize("lx", sorted(Path("configs/experiments/lx_es").rglob("*.yaml")), ids=lambda p: p.stem)
def test_lcmv_and_wmne_use_the_identical_training_protocol(lx):
    wx = Path(str(lx).replace("lx_es", "wx_es").replace("lx10_es", "wx10_es"))
    drop = ("experiment", "graph_family", "name", "config")
    a, b = expand_variants(load_experiment(lx)), expand_variants(load_experiment(wx))
    assert [{k: v for k, v in x.items() if k not in drop} for x in a] == \
           [{k: v for k, v in y.items() if k not in drop} for y in b]
    for x in a:
        t = x["training"]
        assert (t["epochs"], t["early_stopping"]["enabled"], t["early_stopping"]["patience"],
                t["early_stopping"]["min_delta"], t["early_stopping"]["restore_best"],
                t["early_stopping"]["refit"]) == (200, True, 10, 1e-4, True, False)
        assert x["input"] == "graph"                                       # GNNs only (Exp 2-8)


def test_no_early_stopping_for_classical_models_or_existing_configs():
    assert not list(Path("configs/experiments/lx_es").rglob("*exp[01]_*.yaml"))
    for p in list(Path("configs/experiments/lx").rglob("*.yaml")) + list(Path("configs/experiments/wx").rglob("*.yaml")):
        for v in expand_variants(load_experiment(p)):
            if v["input"] == "graph":                                      # the fixed 60-epoch protocol, unchanged
                assert v["training"]["epochs"] == 60 and not v["training"]["early_stopping"]["enabled"]
