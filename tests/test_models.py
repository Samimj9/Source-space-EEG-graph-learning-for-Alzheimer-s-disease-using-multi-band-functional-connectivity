import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adgraph.models.gcn import DenseGCNConv  # noqa: E402
from adgraph.models.multiband import FUSIONS, BandFusion, MultiBandGNN  # noqa: E402

B, K, R = 3, 5, 68


def rand_graphs(b=B, k=K, r=R, f=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(b, k, r, r, generator=g)
    a = (a + a.transpose(-1, -2)) / 2
    a = a * (1 - torch.eye(r))
    return torch.randn(b, k, r, f, generator=g), a


def test_conv_shapes():
    x, a = rand_graphs(k=1)
    assert DenseGCNConv(2, 8)(x[:, 0], a[:, 0]).shape == (B, R, 8)


@pytest.mark.parametrize("k,shared,fusion,band_identity", [(1, True, "mean", "none"), (5, True, "mean", "onehot"),
                                                           (5, False, "concat", "none"), (5, True, "gate", "none"),
                                                           (5, False, "attention", "none"),
                                                           (4, True, "attention", "onehot")])
def test_forward_shapes_and_gradients(k, shared, fusion, band_identity):
    x, a = rand_graphs(k=k)
    net = MultiBandGNN("gcn", in_dim=2, n_bands=k, hidden=16, shared_encoder=shared, fusion=fusion,
                       band_identity=band_identity)
    logits, w = net(x, a, return_band_weights=True)
    assert logits.shape == (B,) and w.shape == (B, k)
    assert torch.allclose(w.sum(1), torch.ones(B), atol=1e-5)
    logits.sum().backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)                  # gradients propagate to the parameters


def test_node_permutation_invariance():
    torch.manual_seed(0)
    x, a = rand_graphs(k=2)
    net = MultiBandGNN("gcn", 2, 2, hidden=16, fusion="attention").eval()
    perm = torch.randperm(R)
    assert torch.allclose(net(x, a), net(x[:, :, perm], a[:, :, perm][:, :, :, perm]), atol=1e-5)


def test_constant_features_still_see_connectivity():
    """Experiment 2: with constant node features the output must depend on the adjacency."""
    torch.manual_seed(0)
    x = torch.ones(2, 1, R, 1)
    _, a = rand_graphs(b=2, k=1, f=1, seed=1)
    a[1] = a[1] ** 3                                              # different connectivity, same features
    net = MultiBandGNN("gcn", 1, 1, hidden=16).eval()
    out = net(x, a)
    assert not torch.allclose(out[0], out[1], atol=1e-6)


def test_shared_vs_band_specific_parameters():
    shared = MultiBandGNN("gcn", 2, 5, hidden=16, shared_encoder=True)
    specific = MultiBandGNN("gcn", 2, 5, hidden=16, shared_encoder=False)
    n = lambda m: sum(p.numel() for p in m.encoders.parameters())   # noqa: E731
    assert len(shared.encoders) == 1 and len(specific.encoders) == 5 and n(specific) == 5 * n(shared)


@pytest.mark.parametrize("fusion", FUSIONS)
def test_fusion_output_dims(fusion):
    f = BandFusion(fusion, dim=10, n_bands=5)
    out, w = f(torch.randn(6, 5, 10))
    assert out.shape == (6, 50 if fusion == "concat" else 10) and w.shape == (6, 5)


def test_shared_encoder_band_symmetry_and_band_identity():
    torch.manual_seed(0)
    x, a = rand_graphs(k=5)
    order = [3, 1, 0, 4, 2]
    plain = MultiBandGNN("gcn", 2, 5, hidden=8, shared_encoder=True, fusion="mean").eval()
    assert torch.allclose(plain(x, a), plain(x[:, order], a[:, order]), atol=1e-5)     # invariant (entry 57)
    ident = MultiBandGNN("gcn", 2, 5, hidden=8, shared_encoder=True, fusion="mean", band_identity="onehot").eval()
    assert not torch.allclose(ident(x, a), ident(x[:, order], a[:, order]), atol=1e-5)


def test_wrong_number_of_bands_rejected():
    x, a = rand_graphs(k=3)
    with pytest.raises(ValueError):
        MultiBandGNN("gcn", 2, 5)(x, a)


@pytest.mark.parametrize("name", ["gat", "graphsage", "edge_weighted_gat", "weighted_graphsage"])
def test_only_the_gcn_encoder_is_offered(name):
    from adgraph.models.encoders import build_encoder

    with pytest.raises(ValueError, match="final framework"):
        build_encoder(name, 2, 8, 1, 0.0)


def _fold(d3_store, cfg):
    from adgraph.evaluation.splits import fold_indices, make_splits
    from adgraph.graphs.dataset import build_graph

    data = build_graph(d3_store, cfg["graph"], cfg["node_features"])
    sp = make_splits(data.meta, "xu_dev_cv", 5, 1, 0)[0]
    tr, te = fold_indices(data.meta, sp)
    return data, tr, te


GRAPH = {"graph": {"connectivity": "ciplv", "bands": ["delta", "theta", "alpha", "beta", "gamma"],
                   "band_mode": "multi"},
         "node_features": {"components": ["psd", "strength"], "psd_kinds": ["log"], "psd_bands": "matched"}}


def test_gnn_model_fixed_epochs_reproducible(d3_store):
    from adgraph.models.trainer import GNNModel

    data, tr, te = _fold(d3_store, GRAPH)
    mcfg = {"encoder": "gcn", "hidden": 8, "n_layers": 1, "fusion": "attention", "band_identity": "onehot"}
    m = GNNModel(mcfg, {"epochs": 3, "batch_size": 16}, seed=0).fit(data.view(tr, "train"))
    p, w = m.predict_proba(data.view(te, "test"), return_band_weights=True)
    assert p.shape == (len(te),) and ((p >= 0) & (p <= 1)).all() and w.shape == (len(te), 5)
    assert [h["epoch"] for h in m.history] == [1, 2, 3]                     # exactly the fixed epochs
    s = m.training_summary()
    assert s["epochs"] == 3 and len(eval(s["history_train_loss"])) == 3 and np.isfinite(s["final_grad_norm"])
    p2 = GNNModel(mcfg, {"epochs": 3, "batch_size": 16}, seed=0).fit(data.view(tr, "train"))
    assert np.allclose(p, p2.predict_proba(data.view(te, "test")), atol=1e-6)   # seeded -> reproducible


def test_gnn_model_refuses_early_stopping_and_test_views(d3_store):
    from adgraph.models.trainer import GNNModel

    with pytest.raises(ValueError, match="early stopping"):
        GNNModel({"encoder": "gcn"}, {"epochs": 60, "early_stopping": {"enabled": True}})
    data, tr, te = _fold(d3_store, GRAPH)
    with pytest.raises(Exception):
        GNNModel({"encoder": "gcn", "hidden": 8, "n_layers": 1}, {"epochs": 1}).fit(data.view(te, "test"))
