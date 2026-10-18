"""Graph construction before GNN classification (graphs.construction, graphs.diagnostics; docs/final/graph_construction.md).
Synthetic adjacencies and the synthetic ds004504 store only."""

import numpy as np
import pandas as pd
import pytest

from adgraph.experiments.config import graph_constructions
from adgraph.graphs.construction import (GraphConstruction, feature_similarity_adjacency, knn_mask, n_components,
                                         proportional_mask, structure_stats)
from adgraph.graphs.dataset import DataView
from adgraph.source.atlas import region_names


def adj(n=6, k=5, r=68, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 1, (n, k, r, r)).astype(np.float32)
    a = (a + np.swapaxes(a, -1, -2)) / 2
    a[..., np.arange(r), np.arange(r)] = 0
    return a


def view(a, role="train"):
    meta = pd.DataFrame({"key": [f"D3-{i}" for i in range(len(a))], "dataset": "D3", "y": 0,
                         "window_id": [f"w{i}" for i in range(len(a))]})
    return DataView(meta, {"adj": a}, role, "y")


@pytest.mark.parametrize("name", list(graph_constructions()))
def test_every_configured_construction_is_a_valid_graph(name):
    cfg = dict(graph_constructions()[name])
    cfg.pop("adjacency", None)
    a = adj()
    gc = GraphConstruction.from_config(cfg, region_names("dk68")).fit(view(a))
    out = gc.transform(a)
    assert out.shape == a.shape and out.dtype == np.float32
    assert np.allclose(out, np.swapaxes(out, -1, -2))                       # symmetric
    assert (np.diagonal(out, axis1=-2, axis2=-1) == 0).all()                # no self-loops (added by the GCN norm)
    assert out.min() >= 0                                                   # no negative weights
    kept = out > 0
    assert np.array_equal(out[kept], a[kept])                               # retained edges keep their weights
    st = structure_stats(out)
    assert st["symmetric"] and st["zero_diagonal"] and st["min_weight"] >= 0
    if cfg["method"] == "knn" and cfg.get("anatomical", "none") == "none":
        assert st["isolated_nodes_per_graph"] == 0                          # no node loses every edge


def test_dense_is_the_identity():
    a = adj()
    assert GraphConstruction().transform(a) is a


@pytest.mark.parametrize("r,f,k", [(68, 0.10, 7), (68, 0.20, 13), (68, 0.30, 20), (34, 0.20, 7), (18, 0.20, 3),
                                   (12, 0.10, 1)])
def test_k_is_relative_to_the_number_of_rois(r, f, k):
    assert GraphConstruction("knn", k_fraction=f).k_for(r) == k


def test_union_knn_keeps_each_nodes_k_strongest_edges():
    a = adj(n=2, k=1)[:, 0]
    k = 5
    m = knn_mask(a, k, "union")
    for g in range(len(a)):
        for i in range(a.shape[-1]):
            top = np.argsort(-np.where(np.eye(68, dtype=bool)[i], -1, a[g, i]))[:k]
            assert m[g, i, top].all()                                       # i's own k strongest are kept
    assert (m.sum(-1) >= k).all() and np.array_equal(m, np.swapaxes(m, -1, -2))


def test_mutual_knn_is_a_subset_of_union_and_repairs_isolated_nodes():
    a = adj(n=3, k=1)[:, 0]
    union, mutual = knn_mask(a, 4, "union"), knn_mask(a, 4, "mutual", min_degree=1)
    raw = knn_mask(a, 4, "mutual", min_degree=0)
    assert not (raw & ~union).any() and (raw.sum(-1) <= 4).all()
    assert (mutual.sum(-1) >= 1).all() and np.array_equal(mutual, np.swapaxes(mutual, -1, -2))
    assert not (mutual & ~union).any()                                      # repair edges are kNN edges too


def test_directed_knn_is_reported_as_not_symmetric():
    gc = GraphConstruction("knn", k=3, symmetrize="none")
    assert not gc.symmetric
    out = knn_mask(adj(n=1, k=1)[:, 0], 3, "none")
    assert (out.sum(-1) == 3).all()


def test_zero_weight_edges_are_never_kept():
    a = adj(n=1, k=1)[:, 0]
    a[:, 0, :] = 0
    a[:, :, 0] = 0
    assert not knn_mask(a, 5, "union")[:, 0].any() and not proportional_mask(a, 0.2)[:, 0].any()


def test_proportional_threshold_density():
    a = adj(n=2, k=1)[:, 0]
    m = proportional_mask(a, 0.2)
    iu = np.triu_indices(68, 1)
    assert (m[:, iu[0], iu[1]].sum(-1) == round(0.2 * len(iu[0]))).all()


def test_quantile_threshold_is_fitted_on_training_views_only():
    a = adj()
    gc = GraphConstruction("quantile", q=0.9)
    with pytest.raises(Exception):
        gc.fit(view(a, "test"))
    gc.fit(view(a, "train"))
    t = gc.threshold_.copy()
    gc.transform(adj(seed=5))                                               # applying to other data changes nothing
    assert np.array_equal(t, gc.threshold_)


def test_bands_and_rois_stay_aligned():
    a = adj(n=2, k=5)
    a[:, 3] *= 0.1                                                          # one band much weaker
    out = GraphConstruction("knn", k=5, symmetrize="union").transform(a)
    for b in range(5):                                                      # per band and window, independently
        assert np.array_equal(out[:, b], GraphConstruction("knn", k=5).transform(a[:, b:b + 1])[:, 0])
    perm = np.random.default_rng(0).permutation(68)
    p = GraphConstruction("knn", k=5).transform(a[:, :, perm][:, :, :, perm])
    assert np.array_equal(p, out[:, :, perm][:, :, :, perm])                # ROI relabelling commutes


def test_contiguity_mask_restricts_to_cortical_neighbours():
    from adgraph.source.atlas import contiguity

    names = region_names("dk68")
    out = GraphConstruction("dense", anatomical="contiguity", roi_names=names).transform(adj())
    c = contiguity(names).astype(bool)
    assert not (out > 0)[..., ~c].any() and (out > 0)[..., c].mean() > 0.99
    with pytest.raises(ValueError):
        GraphConstruction("dense", anatomical="contiguity")                 # needs the store's ROI names


def test_feature_similarity_graph_is_a_valid_but_different_graph():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((4, 68, 15))
    s = feature_similarity_adjacency(f, 5)
    assert s.shape == (4, 5, 68, 68) and np.allclose(s[:, 0], s[:, 3])     # the same graph for every band
    assert np.allclose(s, np.swapaxes(s, -1, -2)) and (np.diagonal(s, axis1=-2, axis2=-1) == 0).all()
    assert s.min() >= 0 and s.max() <= 1
    shift = feature_similarity_adjacency(f * 3 + 7, 5)                      # z-scored per window: scale-free
    assert np.allclose(s, shift, atol=1e-5)


def test_feature_similarity_graphs_keep_connectivity_strength(d3_store):
    from adgraph.graphs.dataset import build_graph, node_strength

    g = {"connectivity": "plv", "bands": ["delta", "theta", "alpha", "beta", "gamma"], "band_mode": "multi"}
    nf = {"components": ["psd", "strength"], "psd_kinds": ["log"], "psd_bands": "matched"}
    conn = build_graph(d3_store, g, nf)
    sim = build_graph(d3_store, {**g, "adjacency": "feature_similarity"}, nf)
    assert not np.allclose(conn.payload["adj"], sim.payload["adj"])
    assert np.array_equal(conn.payload["x"], sim.payload["x"])            # strength = functional connectivity
    assert np.allclose(conn.payload["x"][..., -1], node_strength(d3_store.arrays["plv"]))


def test_connected_components():
    a = np.zeros((4, 4))
    a[0, 1] = a[1, 0] = a[2, 3] = a[3, 2] = 1
    assert n_components(a) == 2


def test_diagnostics_are_label_blind_and_development_only(d3_store):
    from adgraph.evaluation.splits import subject_table, xu_partition
    from adgraph.graphs.diagnostics import construction_diagnostics, summarize

    dev, test = xu_partition(subject_table(d3_store.windows))
    cons = {k: graph_constructions()[k] for k in ("dense", "knn_union_f20", "knn_mutual_f20",
                                                   "feature_similarity_knn_union_f20")}
    d = construction_diagnostics(d3_store, cons, dev, connectivity=("plv",), per_participant=3)
    assert set(d["n_participants"]) == {len(dev)}
    flipped = d3_store.subset(np.arange(d3_store.n_windows))
    flipped.windows = flipped.windows.assign(y=1 - flipped.windows["y"])
    d2 = construction_diagnostics(flipped, cons, dev, connectivity=("plv",), per_participant=3)
    pd.testing.assert_frame_equal(d, d2)                                   # the diagnosis is never read
    s = summarize(d).set_index("construction")
    assert s.loc["dense", "density"] == pytest.approx(1.0) and s.loc["knn_union_f20", "density"] < 0.5
    assert s.loc["knn_mutual_f20", "relative_edges"] <= s.loc["knn_union_f20", "relative_edges"]


def test_gcn_normalization_is_symmetric_with_one_self_loop_and_no_nan_for_isolated_nodes():
    torch = pytest.importorskip("torch")
    from adgraph.models.gcn import DenseGCNConv
    from adgraph.models.xu_dual import normalize

    a = adj(n=2, k=1)[:, 0]
    a[0, 5, :] = a[0, :, 5] = 0                                              # node 5 isolated in graph 0
    t = torch.from_numpy(a)
    n = normalize(t).numpy()
    m = a + np.eye(68)
    d = m.sum(-1)
    assert np.allclose(n, m / np.sqrt(d[:, :, None] * d[:, None, :]), atol=1e-6)   # D^-1/2 (A + I) D^-1/2
    assert np.allclose(n, np.swapaxes(n, -1, -2)) and np.isfinite(n).all()
    assert n[0, 5, 5] == pytest.approx(1.0) and np.allclose(np.delete(n[0, 5], 5), 0)  # isolated: keeps itself
    conv = DenseGCNConv(2, 3)
    x = torch.randn(2, 68, 2)
    with torch.no_grad():
        out = conv(x, t)
        want = torch.from_numpy(n.astype(np.float32)) @ conv.lin(x) if hasattr(conv, "lin") else None
    assert torch.isfinite(out).all() and out.shape == (2, 68, 3)
    if want is not None:
        assert torch.allclose(out, want, atol=1e-5)


def test_sparser_graphs_keep_more_of_each_rois_own_spectral_feature():
    from adgraph.graphs.diagnostics import propagation

    a = adj(n=20, k=1)[:, 0] * 0.6
    f = np.random.default_rng(0).standard_normal((20, 68))
    dense = propagation(a, f)
    knn = propagation(GraphConstruction("knn", k=7).transform(a[:, None])[:, 0], f)
    assert dense["gcn_self_weight"] == pytest.approx(np.mean(1 / (1 + a.sum(-1))), rel=1e-6)
    assert knn["gcn_self_weight"] > 2 * dense["gcn_self_weight"]
    assert knn["feature_variance_retained_1"] > dense["feature_variance_retained_1"]
    assert propagation(np.zeros((1, 68, 68)), f[:1])["feature_variance_retained_1"] == pytest.approx(1.0)
