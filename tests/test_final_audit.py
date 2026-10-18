"""Correctness audit of the final experiments Exp 0-8 (methodology log entries 57, 58). Synthetic data only.

1. Labels and score orientation: AD = 1 / HC = 0 from the provider codes; every model family (LR, SVM margin,
   calibrated SVM, every GNN architecture of Exp 2-8) scores AD-positive signal higher (P(AD), not P(HC));
   window -> participant aggregation keeps keys and labels aligned; thresholds and targets in every config.
2. Band identity: every multiband variant of every final config is sensitive to permuting the band graphs (the
   band-permutation-invariant shared-encoder variants of the earlier suites are not part of the final experiments,
   and the config validation refuses them unless declared as a baseline).
3. Band identity works for the reason intended: with a band-direction signal (one band up, another down in AD)
   a shared encoder without band identity cannot separate the classes, the band-identity variant can.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from adgraph.evaluation.metrics import aggregate_subjects
from adgraph.experiments.config import expand_variants, load_experiment
from adgraph.graphs.dataset import Dataset
from adgraph.models.multiband import MultiBandGNN
from adgraph.models.trainer import GNNModel
from adgraph.models.xu_dual import DK_LOBES

CONFIGS = sorted(Path("configs/final/experiments").glob("exp*.yaml"))
DK = sorted(f"{r}-{h}" for regions in DK_LOBES.values() for r in regions for h in ("lh", "rh"))   # 68, store order
K, R = 5, 68


# ------------------------------------------------------------------------------------------- 1. labels, scores
def test_provider_groups_map_to_ad_1_hc_0(tmp_path):
    from adgraph.io.bids import load_participants_tsv

    tsv = tmp_path / "participants.tsv"
    tsv.write_text("participant_id\tGender\tAge\tGroup\tMMSE\nsub-001\tF\t57\tA\t16\nsub-037\tM\t57\tC\t30\n"
                   "sub-066\tM\t70\tF\t20\n")
    kept, excluded = load_participants_tsv(tsv)
    assert dict(zip(kept["key"], zip(kept["label"], kept["y"]))) == {"D3-1": ("AD", 1), "D3-37": ("HC", 0)}
    assert list(excluded["key"]) == ["D3-66"]                     # FTD excluded by design


def _participants(n_ad=24, n_hc=21, windows=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"key": f"D3-{i}", "dataset": "D3", "subject_id": i, "y": int(i <= n_ad), "window": w}
            for i in range(1, n_ad + n_hc + 1) for w in range(windows)]
    meta = pd.DataFrame(rows)
    meta["window_id"] = meta["key"] + "_w" + meta["window"].astype(str)
    meta["label"] = np.where(meta["y"] == 1, "AD", "HC")
    return meta.sample(frac=1.0, random_state=seed).reset_index(drop=True), rng   # shuffled order


def _split(meta, frac=0.6, seed=0):
    keys = sorted(meta["key"].unique())
    rng = np.random.default_rng(seed)
    ad = [k for k in keys if meta.loc[meta.key == k, "y"].iloc[0] == 1]
    hc = [k for k in keys if k not in ad]
    tr = set(rng.choice(ad, int(frac * len(ad)), replace=False)) | set(rng.choice(hc, int(frac * len(hc)), replace=False))
    m = meta["key"].isin(tr).to_numpy()
    return np.flatnonzero(m), np.flatnonzero(~m)


def _participant_auc(meta, idx, score):
    out = meta.iloc[idx][["key", "dataset", "y"]].copy()
    out["score"] = score
    s = aggregate_subjects(out)
    return roc_auc_score(s["y"], s["score"])


@pytest.mark.parametrize("model", [{"name": "logreg"}, {"name": "svm", "calibration": "none"},
                                   {"name": "svm", "calibration": "grouped_platt"}], ids=lambda m: str(m))
def test_classical_scores_are_ad_positive(model):
    from adgraph.models.classical import ClassicalModel

    meta, rng = _participants()
    x = rng.standard_normal((len(meta), 10))
    x[:, 3] += 1.5 * meta["y"].to_numpy()                          # AD has larger feature 3
    data = Dataset(meta, {"X": x}, "classical", {"feature_names": [f"f{i}" for i in range(10)]}, "y")
    tr, te = _split(meta)
    m = ClassicalModel(model["name"], {"C": 1.0}, {}, seed=0,
                       calibration=model.get("calibration", "none")).fit(data.view(tr, "train"))
    score = m.predict_score(data.view(te, "test"))
    assert _participant_auc(meta, te, score) > 0.9                 # oriented to AD, never inverted


def _graph_data(signal="level", seed=0):
    meta, rng = _participants(seed=seed)
    n = len(meta)
    x = rng.standard_normal((n, K, R, 2)).astype(np.float32)
    a = rng.uniform(0, 1, (n, K, R, R)).astype(np.float32)
    a = (a + a.transpose(0, 1, 3, 2)) / 2
    a[:, :, range(R), range(R)] = 0
    s = (2 * meta["y"].to_numpy() - 1).astype(np.float32)[:, None]
    if signal == "level":                                          # AD: higher feature 0 in band 1
        x[:, 1, :, 0] += 1.0 * (s + 1) / 2
    else:                                                          # band direction: band 1 up, band 2 down in AD
        x[:, 1, :, 1] += 1.0 * s
        x[:, 2, :, 1] -= 1.0 * s
    return meta, Dataset(meta, {"x": x, "adj": a}, "graph", {"graph_bands": list("12345")}, "y")


ARCHS = {
    "gcn_shared_mean": {"encoder": "gcn", "shared_encoder": True, "fusion": "mean"},
    "gcn_shared_attention": {"encoder": "gcn", "shared_encoder": True, "fusion": "attention"},
    "gcn_separate_mean": {"encoder": "gcn", "shared_encoder": False, "fusion": "mean"},
    "gcn_shared_attention_bandid": {"encoder": "gcn", "shared_encoder": True, "fusion": "attention",
                                    "band_identity": "onehot"},
    "earm": {"architecture": "earm_gnn", "hidden": 16, "n_blocks": 2},
    "xu_dual": {"architecture": "xu_dual_gcn", "hidden": 16, "fc1": 16, "fc2": 16, "roi_names": DK},
}


@pytest.mark.parametrize("arch", ARCHS, ids=str)
def test_gnn_probabilities_are_p_ad(arch):
    meta, data = _graph_data("level")
    tr, te = _split(meta)
    gcn = {"hidden": 16, "n_layers": 2, "dropout": 0.1, "readout": "mean_max", "mlp_hidden": 16}
    cfg = ARCHS[arch] if "architecture" in ARCHS[arch] else {**gcn, **ARCHS[arch]}
    m = GNNModel(cfg, {"epochs": 30, "batch_size": 32, "lr": 3e-3},
                 seed=0).fit(data.view(tr, "train"))
    p = m.predict_proba(data.view(te, "test"))
    assert ((p >= 0) & (p <= 1)).all()
    assert _participant_auc(meta, te, p) > 0.85                    # P(AD) rises with the AD signal


def test_aggregation_keeps_participants_and_labels_aligned():
    meta, rng = _participants()
    pred = meta[["key", "dataset", "y"]].copy()
    pred["score"] = rng.uniform(size=len(pred))
    s = aggregate_subjects(pred.sample(frac=1.0, random_state=3))
    for k, row in s.set_index("key").iterrows():
        assert row["y"] == meta.loc[meta.key == k, "y"].iloc[0]
        assert np.isclose(row["score"], pred.loc[pred.key == k, "score"].mean())
    bad = pred.copy()
    bad.loc[bad.index[0], "y"] = 1 - bad.loc[bad.index[0], "y"]
    with pytest.raises(ValueError):
        aggregate_subjects(bad)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda p: p.stem)
def test_every_config_uses_the_diagnosis_target_and_its_score_threshold(cfg):
    from adgraph.experiments.runner import score_threshold

    for v in expand_variants(load_experiment(cfg)):
        assert v["target"] == "label" and "covariates" not in v
        thr = score_threshold(v)
        if v["input"] == "classical" and v["model"]["name"] == "svm" and v["model"].get("calibration", "none") == "none":
            assert thr == 0.0                                       # decision-function margin
        else:
            assert thr == 0.5                                       # probability of AD


# ------------------------------------------------------------------------------------------- 2. band identity
def _built(v, store):
    from adgraph.graphs.dataset import build_graph

    cfg = copy.deepcopy(v["model"])
    x = build_graph(store, v["graph"], v["node_features"]).payload["x"]           # the real input dimensions
    k, f = x.shape[1], x.shape[-1]
    torch.manual_seed(0)
    m = GNNModel(cfg, v["training"]).build_net(f, k, R) if cfg.get("architecture") != "xu_dual_gcn" else \
        GNNModel({**cfg, "roi_names": DK}, v["training"]).build_net(f, k, R)
    with torch.no_grad():                                          # learned band parameters away from their init
        for name, p in m.named_parameters():
            if name in ("band_emb", "gate.s", "gate.b"):
                p.add_(torch.randn_like(p))
    return m.eval(), f, k


def _permutation_change(m, f, k):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, k, R, f, generator=g)
    a = torch.rand(4, k, R, R, generator=g)
    a = (a + a.transpose(-1, -2)) / 2
    perm = torch.roll(torch.arange(k), 1)
    with torch.no_grad():
        return float((m(x, a) - m(x[:, perm], a[:, perm])).abs().max())


MULTI = [(c, v["variant"]) for c in CONFIGS for v in expand_variants(load_experiment(c))
         if v["input"] == "graph" and v["graph"]["band_mode"] == "multi"]


@pytest.mark.parametrize("cfg,variant", MULTI, ids=lambda x: x.stem if isinstance(x, Path) else x)
def test_every_final_multiband_variant_is_band_aware(cfg, variant, d3_store):
    v = next(x for x in expand_variants(load_experiment(cfg)) if x["variant"] == variant)
    m, f, k = _built(v, d3_store)
    assert _permutation_change(m, f, k) > 1e-4


def test_multiband_variants_cover_exp5_to_exp8():
    assert {c.stem.split("_")[0] for c, _ in MULTI} == {"exp5", "exp6", "exp7", "exp8"}
    assert len(MULTI) == 8 + 10 + 6 + 10


def test_a_band_invariant_multiband_config_is_refused_unless_declared():
    from adgraph.experiments.config import validate

    v = next(x for x in expand_variants(load_experiment(Path("configs/final/experiments/exp5_multibranch.yaml")))
             if x["variant"] == "plv__shared_bandid__mean")
    bad = copy.deepcopy(v)
    bad["model"]["band_identity"] = "none"
    with pytest.raises(ValueError, match="invariant"):
        validate(bad)
    validate({**bad, "model": {**bad["model"], "band_invariant_baseline": True}})


# ------------------------------------------------------------------------------------------- 3. the correction
def _paired_direction_data(n_pairs=24, windows=3, seed=0):
    """AD participant i and HC participant i + n_pairs have IDENTICAL band graphs except that bands 1 and 2 are
    swapped; AD: band 1 relative power up and band 2 down, so HC is the reverse. A band-permutation-invariant model
    gives each pair the same score (AUC exactly 0.5); a band-aware model can separate them."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_pairs, windows, K, R, 2)).astype(np.float32)
    x[:, :, 1, :, 1] += 1.0
    x[:, :, 2, :, 1] -= 1.0
    a = rng.uniform(0, 1, (n_pairs, windows, K, R, R)).astype(np.float32)
    a = (a + np.swapaxes(a, -1, -2)) / 2
    a[..., range(R), range(R)] = 0
    swap = [0, 2, 1, 3, 4]
    X = np.concatenate([x, x[:, :, swap]]).reshape(-1, K, R, 2)
    A = np.concatenate([a, a[:, :, swap]]).reshape(-1, K, R, R)
    rows = [{"key": f"D3-{i + 1}", "dataset": "D3", "subject_id": i + 1, "y": int(i < n_pairs), "pair": i % n_pairs,
             "window_id": f"D3-{i + 1}_w{w}"} for i in range(2 * n_pairs) for w in range(windows)]
    meta = pd.DataFrame(rows)
    return meta, Dataset(meta, {"x": X, "adj": A}, "graph", {"graph_bands": list("12345")}, "y")


@pytest.mark.parametrize("bandid", ["none", "onehot"])
def test_band_identity_restores_band_direction_signals(bandid):
    meta, data = _paired_direction_data()
    test_pairs = set(range(0, 24, 3))                              # whole pairs on one side of the split
    te = np.flatnonzero(meta["pair"].isin(test_pairs).to_numpy())
    tr = np.flatnonzero(~meta["pair"].isin(test_pairs).to_numpy())
    cfg = {"encoder": "gcn", "hidden": 16, "n_layers": 2, "dropout": 0.1, "readout": "mean_max", "mlp_hidden": 16,
           "shared_encoder": True, "fusion": "mean", "band_identity": bandid}
    m = GNNModel(cfg, {"epochs": 40, "batch_size": 32, "lr": 3e-3},
                 seed=0).fit(data.view(tr, "train"))
    auc = _participant_auc(meta, te, m.predict_proba(data.view(te, "test")))
    if bandid == "none":
        assert auc == pytest.approx(0.5, abs=1e-6)                 # provably blind to the band direction
    else:
        assert auc > 0.9, auc


def test_band_identity_default_leaves_the_original_model_unchanged():
    torch.manual_seed(0)
    a = MultiBandGNN("gcn", 2, K, hidden=8)
    torch.manual_seed(0)
    b = MultiBandGNN("gcn", 2, K, hidden=8, band_identity="none")
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    with pytest.raises(ValueError):
        MultiBandGNN("gcn", 2, K, band_identity="learned_typo")
