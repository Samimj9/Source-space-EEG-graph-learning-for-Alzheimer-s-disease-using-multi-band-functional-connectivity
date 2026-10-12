import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adgraph.models.edge_weighted_gat import EdgeWeightedGATConv  # noqa: E402
from adgraph.models.gcn import DenseGCNConv  # noqa: E402
from adgraph.models.weighted_graphsage import WeightedSAGEConv  # noqa: E402
from adgraph.models.multiband import FUSIONS, BandFusion, MultiBandGNN  # noqa: E402

ENCODERS = ["gcn", "edge_weighted_gat", "weighted_graphsage"]
B, K, R = 3, 4, 68


def rand_graphs(b=B, k=K, r=R, f=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(b, k, r, r, generator=g)
    a = (a + a.transpose(-1, -2)) / 2
    a = a * (1 - torch.eye(r))
    return torch.randn(b, k, r, f, generator=g), a


@pytest.mark.parametrize("conv", [DenseGCNConv(2, 8), EdgeWeightedGATConv(2, 8, heads=2), WeightedSAGEConv(2, 8)])
def test_conv_shapes(conv):
    x, a = rand_graphs(k=1)
    assert conv(x[:, 0], a[:, 0]).shape == (B, R, 8)


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("k,shared,fusion", [(1, True, "mean"), (4, True, "mean"), (4, False, "concat"),
                                             (4, True, "gate"), (4, False, "attention"), (3, True, "attention")])
def test_forward_shapes(encoder, k, shared, fusion):
    x, a = rand_graphs(k=k)
    net = MultiBandGNN(encoder, in_dim=2, n_bands=k, hidden=16, shared_encoder=shared, fusion=fusion)
    logits, w = net(x, a, return_band_weights=True)
    assert logits.shape == (B,) and w.shape == (B, k)
    assert torch.allclose(w.sum(1), torch.ones(B), atol=1e-5)
    logits.sum().backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad and p.grad is not None)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_node_permutation_invariance(encoder):
    torch.manual_seed(0)
    x, a = rand_graphs(k=2)
    net = MultiBandGNN(encoder, 2, 2, hidden=16, fusion="attention").eval()
    perm = torch.randperm(R)
    out = net(x, a)
    out_p = net(x[:, :, perm], a[:, :, perm][:, :, :, perm])
    assert torch.allclose(out, out_p, atol=1e-5)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_constant_features_still_see_connectivity(encoder):
    """Experiment 2: with constant node features the output must depend on the adjacency."""
    torch.manual_seed(0)
    x = torch.ones(2, 1, R, 1)
    _, a = rand_graphs(b=2, k=1, f=1, seed=1)
    a[1] = a[1] ** 3                                              # different connectivity, same features
    net = MultiBandGNN(encoder, 1, 1, hidden=16).eval()
    out = net(x, a)
    assert not torch.allclose(out[0], out[1], atol=1e-6)


def test_shared_vs_band_specific_parameters():
    shared = MultiBandGNN("gcn", 2, 4, hidden=16, shared_encoder=True)
    specific = MultiBandGNN("gcn", 2, 4, hidden=16, shared_encoder=False)
    n = lambda m: sum(p.numel() for p in m.encoders.parameters())   # noqa: E731
    assert len(shared.encoders) == 1 and len(specific.encoders) == 4 and n(specific) == 4 * n(shared)


@pytest.mark.parametrize("fusion", FUSIONS)
def test_fusion_output_dims(fusion):
    f = BandFusion(fusion, dim=10, n_bands=4)
    z = torch.randn(5, 4, 10)
    out, w = f(z)
    assert out.shape == (5, 40 if fusion == "concat" else 10) and w.shape == (5, 4)


def test_mean_fusion_is_band_symmetric_with_shared_encoder():
    torch.manual_seed(0)
    x, a = rand_graphs(k=4)
    net = MultiBandGNN("gcn", 2, 4, hidden=8, shared_encoder=True, fusion="mean").eval()
    order = [3, 1, 0, 2]
    assert torch.allclose(net(x, a), net(x[:, order], a[:, order]), atol=1e-5)


def test_wrong_number_of_bands_rejected():
    x, a = rand_graphs(k=3)
    with pytest.raises(ValueError):
        MultiBandGNN("gcn", 2, 4)(x, a)


def test_gat_masks_absent_edges():
    conv = EdgeWeightedGATConv(2, 4, heads=1)
    x, a = rand_graphs(k=1)
    a = a[:, 0]
    a[:, :, 10:] = 0
    a[:, 10:, :] = 0
    _, alpha = conv(x[:, 0], a, return_attention=True)
    assert torch.all(alpha[:, :10, 10:] == 0)
    assert torch.allclose(alpha.sum(2), torch.ones_like(alpha.sum(2)))


def test_gnn_model_fit_predict(store):
    from adgraph.evaluation.splits import fold_indices, make_splits, setting_subset
    from adgraph.graphs.dataset import build_graph
    from adgraph.models.trainer import GNNModel

    data = build_graph(store, {"connectivity": "ciplv", "bands": ["theta", "alpha1", "alpha2", "beta"],
                               "band_mode": "multi"},
                       {"components": ["psd", "strength"], "psd_kinds": ["log"], "psd_bands": "matched"})
    sub = setting_subset(data.meta, "within_D1")
    meta = data.meta.iloc[sub].reset_index(drop=True)
    tr, te = fold_indices(meta, make_splits(meta, "within_D1", 5, 1, 0)[0])
    model = GNNModel({"encoder": "edge_weighted_gat", "hidden": 8, "n_layers": 1, "fusion": "gate", "encoder_kwargs": {"heads": 2}},
                     {"epochs": 2, "batch_size": 16,
                      "early_stopping": {"enabled": True, "patience": 1, "val_subject_fraction": 0.25}}, seed=0)
    model.fit(data.view(sub[tr], "train"))
    p, w = model.predict_proba(data.view(sub[te], "test"), return_band_weights=True)
    assert p.shape == (len(te),) and ((p >= 0) & (p <= 1)).all() and w.shape == (len(te), 4)
    assert "val_loss" in model.history[0]
    p2 = GNNModel({"encoder": "edge_weighted_gat", "hidden": 8, "n_layers": 1, "fusion": "gate", "encoder_kwargs": {"heads": 2}},
                  {"epochs": 2, "batch_size": 16,
                   "early_stopping": {"enabled": True, "patience": 1, "val_subject_fraction": 0.25}}, seed=0)
    p2 = p2.fit(data.view(sub[tr], "train")).predict_proba(data.view(sub[te], "test"))
    assert np.allclose(p, p2, atol=1e-5)                                  # seeded -> reproducible


@pytest.mark.parametrize("name", ["gat", "graphsage"])
def test_canonical_names_are_not_offered(name):
    from adgraph.models.encoders import build_encoder

    with pytest.raises(ValueError, match="weighted"):
        build_encoder(name, 2, 8, 1, 0.0)


def test_covariates_enter_only_the_head():
    torch.manual_seed(0)
    x, a = rand_graphs(k=4)
    net = MultiBandGNN("gcn", 2, 4, hidden=8, fusion="attention", n_covariates=1).eval()
    assert net.head[0].in_features == net.fusion.out_dim + 1
    out1 = net(x, a, torch.zeros(B, 1))
    out2 = net(x, a, torch.ones(B, 1) * 3)
    assert not torch.allclose(out1, out2)                                # age changes the logit ...
    _, w1 = net(x, a, torch.zeros(B, 1), return_band_weights=True)
    _, w2 = net(x, a, torch.ones(B, 1) * 3, return_band_weights=True)
    assert torch.allclose(w1, w2)                                        # ... but not the graph part
    with pytest.raises(ValueError):
        net(x, a)
    with pytest.raises(ValueError):
        MultiBandGNN("gcn", 2, 4, hidden=8)(x, a, torch.zeros(B, 1))


def test_gnn_model_with_age(store):
    from adgraph.evaluation.splits import fold_indices, make_splits, setting_subset
    from adgraph.graphs.dataset import add_covariates, build_graph
    from adgraph.models.trainer import GNNModel

    data = add_covariates(build_graph(store, {"connectivity": "plv", "bands": ["theta", "alpha1", "alpha2", "beta"],
                                              "band_mode": "multi"},
                                      {"components": ["psd", "strength"], "psd_kinds": ["log", "rel"]}), ["age"])
    sub = setting_subset(data.meta, "within_D2")
    meta = data.meta.iloc[sub].reset_index(drop=True)
    tr, te = fold_indices(meta, make_splits(meta, "within_D2", 5, 1, 0)[0])
    m = GNNModel({"encoder": "weighted_graphsage", "hidden": 8, "n_layers": 1, "fusion": "attention"},
                 {"epochs": 1}, seed=0).fit(data.view(sub[tr], "train"))
    ages = data.view(sub[tr], "train").meta.drop_duplicates("key")["age"]
    assert m.cov_scaler.mean_[0] == pytest.approx(ages.mean())
    assert m.net.n_covariates == 1
    assert m.predict_proba(data.view(sub[te], "test")).shape == (len(te),)
