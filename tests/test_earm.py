"""EARM-GNN (Experiment 7, methodology log entry 43): focused checks of the model and its experiment.

* shapes, softmax band weights, four distinct band representations;
* symmetric gates in (0, 1), gated adjacency <= EEG adjacency, bounded operator, finite values;
* gradients reach the edge gates (including through the penalty);
* the fixed-adjacency ablation has no gate parameters and uses D^-1/2 A D^-1/2;
* one fit per outer fold on training subjects only; fitted state (gates, history) saved per fold.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from adgraph.experiments.config import expand_variants, hparam_grid, load_experiment, set_dotted  # noqa: E402
from adgraph.experiments.runner import run_variant  # noqa: E402
from adgraph.models.earm import EARMGNN, propagation  # noqa: E402
from adgraph.synthetic import make_synthetic_store  # noqa: E402

CFG = Path("configs/experiments/exp7_earm_gnn.yaml")
B, K, R, F = 5, 4, 68, 3


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(B, K, R, R, generator=g)
    a = (a + a.transpose(-1, -2)) / 2
    a = a * (1 - torch.eye(R))
    return torch.randn(B, K, R, F, generator=g), a


def net(**kw):
    torch.manual_seed(0)
    return EARMGNN(in_dim=F, n_bands=K, n_rois=R, **kw)


def test_shapes_and_band_weights():
    m = net().eval()
    x, a = inputs()
    logits, w = m(x, a, return_band_weights=True)
    assert logits.shape == (B,) and w.shape == (B, K)
    assert torch.allclose(w.sum(1), torch.ones(B), atol=1e-6) and (w > 0).all()
    assert m.node_embeddings(x, a).shape == (B, K, R, 32)
    z, roi_att = m.band_embeddings(x, a, return_roi_attention=True)
    assert z.shape == (B, K, 96) and roi_att.shape == (B, K, R)
    assert torch.allclose(roi_att.sum(-1), torch.ones(B, K), atol=1e-6)
    assert m.fused_embedding(x, a).shape == (B, 96)
    assert sum(p.numel() for p in m.parameters()) == 18214


def test_gates_symmetric_bounded_and_operator_finite():
    m = net()
    g = m.gates()
    assert g.shape == (K, R, R) and torch.allclose(g, g.transpose(-1, -2)) and ((g > 0) & (g < 1)).all()
    x, a = inputs()
    p = m.operator(a)
    assert torch.isfinite(p).all() and torch.allclose(p, p.transpose(-1, -2), atol=1e-6)
    assert ((a * g) <= a + 1e-7).all()
    rho = torch.linalg.eigvalsh(p.double()).abs().amax()
    assert rho <= 1 + 1e-6                                     # bounded propagation (no blow-up with depth)


def test_gradients_reach_gates():
    m = net()
    x, a = inputs()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x, a), torch.tensor([0., 1, 0, 1, 1]))
    (loss + m.regularization()).backward()
    for p in (m.gate.U, m.gate.s, m.gate.b):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert m.regularization() > 0


def test_fixed_adjacency_ablation():
    m = net(adaptive_adjacency=False)
    assert m.gate is None and m.gates() is None and float(m.regularization()) == 0.0
    _, a = inputs()
    d = a.sum(-1).pow(-0.5)
    assert torch.allclose(m.operator(a), d.unsqueeze(-1) * a * d.unsqueeze(-2))
    assert torch.allclose(propagation(a, torch.ones(K, R, R)), propagation(a))
    assert sum(p.numel() for p in m.parameters()) == 18214 - (68 * 8 + 4 * 8 + 4)


def test_bands_are_encoded_separately():
    m = net().eval()
    x, a = inputs()
    z0 = m.band_embeddings(x, a)
    a2 = a.clone()
    a2[:, 0] = a2[:, 0].flip(-1).flip(-2)                       # change only the theta graph
    z1 = m.band_embeddings(x, a2)
    assert not torch.allclose(z0[:, 0], z1[:, 0]) and torch.allclose(z0[:, 1:], z1[:, 1:])
    with pytest.raises(ValueError):
        EARMGNN(in_dim=F, n_bands=K, n_rois=R, fusion="concat")


def test_config_variants():
    vs = {v["variant"]: v for v in expand_variants(load_experiment(CFG))}
    assert set(vs) == {f"{c}__{n}" for c in ("plv", "ciplv") for n in ("full", "fixed_adjacency", "mean_fusion")}
    for v in vs.values():
        assert hparam_grid(v) == [{}] and v["evaluation"]["n_repeats"] == 10
        assert v["graph"]["band_mode"] == "multi" and v["graph"]["bands"] == ["theta", "alpha1", "alpha2", "beta"]
        assert v["node_features"]["components"] == ["psd", "strength"] and v["sparsify"]["method"] == "none"
    assert vs["plv__fixed_adjacency"]["model"]["adaptive_adjacency"] is False
    assert vs["plv__mean_fusion"]["model"]["fusion"] == "mean" and vs["plv__full"]["model"]["fusion"] == "attention"


def test_experiment_one_fit_per_fold_saves_state(tmp_path, monkeypatch):
    from adgraph.models import trainer

    store = make_synthetic_store({"D1": (5, 5), "D2": (5, 6)}, windows_per_subject=(3, 4), psd_effect=0.8,
                                 conn_effect=0.8, seed=1)
    fitted = []
    real_fit = trainer.GNNModel.fit
    monkeypatch.setattr(trainer.GNNModel, "fit", lambda self, view: (fitted.append(set(view.meta["key"])),
                                                                     real_fit(self, view))[1])
    v = expand_variants(load_experiment(CFG))[0]
    for k, val in {"evaluation.n_repeats": 1, "evaluation.k": 3, "evaluation.n_bootstrap": 50,
                   "training.epochs": 2}.items():
        v = set_dotted(v, k, val)
    run_variant(store, v, "within_D2", tmp_path)
    out = tmp_path / "exp7_earm_gnn" / "plv__full" / "within_D2"
    folds = pd.read_csv(out / "folds.csv")
    pred = pd.read_csv(out / "predictions_windows.csv")
    assert len(fitted) == 3
    for f, g in folds.groupby("fold"):
        train, test = set(g.loc[g.side == "train", "key"]), set(g.loc[g.side == "test", "key"])
        assert not train & test and fitted[f] == train and set(pred.query("fold == @f")["key"]) == test
    assert [c for c in pred if c.startswith("band_weight_")] == [f"band_weight_{b}" for b in
                                                                 ("theta", "alpha1", "alpha2", "beta")]
    states = sorted((out / "units").glob("fold_*.model.npz"))
    assert len(states) == 3
    s = np.load(states[0])
    assert s["edge_gates"].shape == (4, 68, 68) and len(s["history_train_loss"]) == 2
    assert json.loads((out / "manifest.json").read_text())["variant_config"]["model"]["architecture"] == "earm_gnn"


def test_comparison_and_dataset_diagnostic_from_saved_outputs(tmp_path):
    from adgraph.evaluation.dataset_shift import diagnose
    from adgraph.experiments.compare import write_comparison

    store = make_synthetic_store({"D1": (4, 4), "D2": (4, 5)}, windows_per_subject=(3, 3), psd_effect=0.8,
                                 conn_effect=0.6, seed=2)
    fast = {"evaluation.n_repeats": 2, "evaluation.k": 3, "evaluation.inner_k": 2, "evaluation.n_bootstrap": 50,
            "training.epochs": 1}

    def run(cfg, keep):
        for v in expand_variants(load_experiment(cfg)):
            if keep(v):
                for k, val in fast.items():
                    v = set_dotted(v, k, val)
                for s in ("within_D1", "within_D2"):
                    run_variant(store, v, s, tmp_path / "results")

    run(Path("configs/experiments/exp1_source_psd.yaml"), lambda v: v["model"]["name"] == "svm")
    run(CFG, lambda v: v["graph"]["connectivity"] == "plv" and v["variant"] != "plv__mean_fusion")
    res = write_comparison(tmp_path / "results", tmp_path / "cmp")
    t = res["table"]
    assert {"EARM-GNN full", "EARM-GNN fixed adjacency (abl. A)", "Exp1 source PSD SVM"} == set(t["model"])
    d = res["differences"]
    assert set(d["comparator"]) == {"EARM-GNN fixed adjacency (abl. A)", "Exp1 source PSD SVM"}
    assert (d["n_shared_repeats"] == 2).all()
    assert any(f.startswith("E7_roc_within_D1_plv") for f in res["figures"])
    diag = diagnose(store, tmp_path / "results", tmp_path / "diag")
    sep = diag["separability"]
    assert {"auc_D1", "auc_D2", "smd_D1_minus_D2_HC"} <= set(sep) and len(sep) == 4 * 4
    assert set(diag["predictions"]["setting"]) == {"within_D1", "within_D2"}
    assert (tmp_path / "diag" / "report.md").exists()
