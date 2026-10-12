import numpy as np
import pytest

from adgraph.graphs.dataset import build_classical, build_graph, node_strength
from adgraph.graphs.store import GraphStore

ALL = ["theta", "alpha1", "alpha2", "beta"]


def g(conn="plv", bands=ALL, mode="multi"):
    return {"connectivity": conn, "bands": bands, "band_mode": mode}


def nf(components, kinds=("log", "rel"), psd_bands="matched"):
    return {"components": list(components), "psd_kinds": list(kinds), "psd_bands": psd_bands}


def test_store_roundtrip(store, tmp_path):
    store.save(tmp_path)
    loaded = GraphStore.load(tmp_path)
    assert loaded.windows.equals(store.windows) or (loaded.windows["window_id"] == store.windows["window_id"]).all()
    for k in store.arrays:
        assert np.array_equal(loaded.arrays[k], store.arrays[k])
    assert GraphStore.load(tmp_path, keys=("plv",)).arrays.keys() == {"plv"}


def test_store_rejects_bad_shapes(store):
    bad = dict(store.arrays, plv=store.arrays["plv"][:, :3])
    with pytest.raises(ValueError):
        GraphStore(bad, store.windows, store.bands, store.band_edges, store.roi_names, store.channels)


@pytest.mark.parametrize("source,n_nodes", [("sensor", 30), ("source", 68)])
def test_classical_shapes(store, source, n_nodes):
    d = build_classical(store, {"source": source, "psd_kinds": ["log", "rel"], "bands": ALL})
    assert d.payload["X"].shape == (store.n_windows, 2 * 4 * n_nodes)
    assert len(d.info["feature_names"]) == d.payload["X"].shape[1]
    d1 = build_classical(store, {"source": source, "psd_kinds": ["rel"], "bands": ["alpha1"]})
    assert d1.payload["X"].shape == (store.n_windows, n_nodes)


@pytest.mark.parametrize("conn", ["plv", "ciplv"])
def test_classical_connectivity_features(store, conn):
    d = build_classical(store, {"source": conn, "bands": ["theta", "beta"]})
    assert d.payload["X"].shape == (store.n_windows, 2 * 68 * 67 // 2)
    assert d.payload["X"][0, 0] == store.arrays[conn][0, 0, 0, 1]


def test_classical_age_and_meta(store):
    d = build_classical(store, {"source": "age"})
    assert d.payload["X"].shape == (store.n_windows, 1)
    for col in ("window_id", "key", "dataset", "subject_id", "y"):
        assert col in d.meta


@pytest.mark.parametrize("conn", ["plv", "ciplv"])
def test_connectivity_switch_selects_right_array(store, conn):
    d = build_graph(store, g(conn), nf(["constant"]))
    assert np.array_equal(d.payload["adj"], store.arrays[conn])
    assert d.info["connectivity"] == conn


def test_plv_and_ciplv_differ(store):
    a = build_graph(store, g("plv"), nf(["constant"])).payload["adj"]
    b = build_graph(store, g("ciplv"), nf(["constant"])).payload["adj"]
    assert not np.allclose(a, b)


def test_band_modes(store):
    n = store.n_windows
    single = build_graph(store, g(bands=["alpha2"], mode="single"), nf(["constant"]))
    assert single.payload["adj"].shape == (n, 1, 68, 68)
    assert np.array_equal(single.payload["adj"][:, 0], store.arrays["plv"][:, 2])
    mean = build_graph(store, g(mode="mean"), nf(["constant"]))
    assert np.allclose(mean.payload["adj"][:, 0], store.arrays["plv"].mean(1), atol=1e-6)
    multi = build_graph(store, g(bands=["beta", "theta"]), nf(["constant"]))
    assert np.array_equal(multi.payload["adj"][:, 0], store.arrays["plv"][:, 3])     # order follows config
    assert multi.info["graph_bands"] == ["beta", "theta"]
    with pytest.raises(ValueError):
        build_graph(store, g(bands=["theta", "beta"], mode="single"), nf(["constant"]))


def test_psd_node_features_matched_and_all(store):
    n = store.n_windows
    m = build_graph(store, g(), nf(["psd"], kinds=["log", "rel"], psd_bands="matched"))
    assert m.payload["x"].shape == (n, 4, 68, 2)
    assert np.allclose(m.payload["x"][:, 1, :, 0], store.arrays["psd_log"][:, 1])     # alpha1 graph <- alpha1 PSD
    assert np.allclose(m.payload["x"][:, 3, :, 1], store.arrays["psd_rel"][:, 3])
    a = build_graph(store, g(mode="mean"), nf(["psd"], kinds=["rel"], psd_bands="all"))
    assert a.payload["x"].shape == (n, 1, 68, 4)
    assert np.allclose(a.payload["x"][:, 0, :, 2], store.arrays["psd_rel"][:, 2])
    s = build_graph(store, g(bands=["beta"], mode="single"), nf(["psd"], kinds=["log", "rel"], psd_bands="all"))
    assert s.payload["x"].shape == (n, 1, 68, 8)                  # Exp 3: all four bands on a single-band graph
    assert np.allclose(s.payload["x"][:, 0, :, 0], store.arrays["psd_log"][:, 0])
    sm = build_graph(store, g(bands=["beta"], mode="single"), nf(["psd"], kinds=["log"], psd_bands="matched"))
    assert sm.payload["x"].shape == (n, 1, 68, 1)
    assert np.allclose(sm.payload["x"][:, 0, :, 0], store.arrays["psd_log"][:, 3])
    md = build_graph(store, g(bands=["alpha1", "beta"]), nf(["psd"], kinds=["rel"], psd_bands="all"))
    assert md.payload["x"].shape == (n, 2, 68, 4)


def test_strength_and_combined_features(store):
    d = build_graph(store, g(), nf(["psd", "strength"], kinds=["log"]))
    x, adj = d.payload["x"], d.payload["adj"]
    assert x.shape == (store.n_windows, 4, 68, 2)                 # X_i,b = [PSD_i,b, strength_i,b]
    expected = adj.sum(-1) / 67                                   # diagonal is zero
    assert np.allclose(x[..., 1], expected, atol=1e-5)
    assert np.allclose(node_strength(adj), expected, atol=1e-5)
    assert d.info["feature_names"] == ["psd_log:band", "strength"]


def test_constant_and_identity(store):
    d = build_graph(store, g(bands=["theta"], mode="single"), nf(["constant", "identity"]))
    assert d.payload["x"].shape == (store.n_windows, 1, 68, 69)
    assert (d.payload["x"][..., 0] == 1).all()
    with pytest.raises(ValueError):
        build_graph(store, g(), nf([]))


def test_dataset_probe_target(store):
    d = build_classical(store, {"source": "source", "psd_kinds": ["log"]}, target="dataset")
    assert d.target == "y_dataset"
    assert (d.meta["y_dataset"] == (d.meta["dataset"] == "D2")).all()


@pytest.mark.parametrize("conn", ["plv", "ciplv"])
def test_strength_from_magnitude_adjacency(store, conn):
    d = build_graph(store, g(conn), nf(["strength"]))
    A = store.arrays[conn]
    assert (A >= 0).all()                                            # ciplv is the magnitude |ciPLV|
    off = A.sum(-1) / 67                                             # zero diagonal
    assert np.allclose(d.payload["x"][..., 0], off, atol=1e-5)
    assert (d.payload["x"] >= 0).all() and (d.payload["x"] <= 1).all()


def test_signed_ciplv_is_not_an_adjacency(store):
    assert np.allclose(np.abs(store.arrays["ciplv_signed"]), store.arrays["ciplv"])
    with pytest.raises(ValueError, match="signed"):
        build_graph(store, g("ciplv_signed"), nf(["constant"]))


def test_adjacency_contract():
    from adgraph.graphs.store import check_adjacency

    a = np.random.default_rng(0).uniform(size=(2, 5, 5))
    a = (a + np.swapaxes(a, -1, -2)) / 2
    a[:, range(5), range(5)] = 0
    check_adjacency(a)
    with pytest.raises(ValueError, match="symmetric"):
        check_adjacency(a + np.triu(np.ones((5, 5)), 1) * 0.1)
    with pytest.raises(ValueError, match="diagonal"):
        check_adjacency(a + np.eye(5) * 0.5)
    with pytest.raises(ValueError, match="outside"):
        check_adjacency(-a)


def test_store_validates_signed_ciplv(store):
    bad = dict(store.arrays, ciplv_signed=np.abs(store.arrays["ciplv_signed"]))   # symmetric -> invalid
    with pytest.raises(ValueError, match="antisymmetric"):
        GraphStore(bad, store.windows, store.bands, store.band_edges, store.roi_names, store.channels)


def test_experiment_node_feature_layouts(store):
    from pathlib import Path

    from adgraph.experiments.config import expand_variants, load_experiment
    from adgraph.graphs.dataset import build_dataset

    exp = Path("configs/experiments")
    n = store.n_windows
    for cfg, shape_x, names in [
        ("exp3_conn_psd.yaml", (n, 1, 68, 2), ["psd_log:theta", "psd_rel:theta"]),
        ("exp3b_conn_psd_allbands.yaml", (n, 1, 68, 8), None),
        ("exp4_band_mean.yaml", (n, 1, 68, 8), None),
        ("exp5_multiband.yaml", (n, 4, 68, 2), ["psd_log:band", "psd_rel:band"]),
        ("exp6_multiband_psd_strength.yaml", (n, 4, 68, 3), ["psd_log:band", "psd_rel:band", "strength"]),
    ]:
        v = expand_variants(load_experiment(exp / cfg))[0]
        d = build_dataset(store, v)
        assert d.payload["x"].shape == shape_x, cfg
        if names:
            assert d.info["feature_names"] == names
    # Exp 3 primary: theta graph gets theta PSD, beta graph gets beta PSD
    vs = {v["variant"]: v for v in expand_variants(load_experiment(exp / "exp3_conn_psd.yaml"))}
    for band, b in (("theta", 0), ("beta", 3)):
        d = build_dataset(store, vs[f"plv__{band}__gcn"])
        assert np.allclose(d.payload["x"][:, 0, :, 0], store.arrays["psd_log"][:, b])
        assert np.allclose(d.payload["x"][:, 0, :, 1], store.arrays["psd_rel"][:, b])


def test_covariates_payload(store):
    from adgraph.graphs.dataset import add_covariates

    d = add_covariates(build_graph(store, g(), nf(["constant"])), ["age"])
    assert d.payload["cov"].shape == (store.n_windows, 1)
    with pytest.raises(ValueError):
        add_covariates(build_classical(store, {"source": "age"}), ["age"])
    with pytest.raises(ValueError):
        add_covariates(d, ["mmse"])
