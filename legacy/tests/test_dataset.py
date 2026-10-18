"""Retired from tests/test_dataset.py by the final framework (docs/final/inventory.md): they test D1/D2 or
earlier-suite functionality that is no longer part of the code base. Kept verbatim; not collected."""

def test_classical_age_and_meta(store):
    d = build_classical(store, {"source": "age"})
    assert d.payload["X"].shape == (store.n_windows, 1)
    for col in ("window_id", "key", "dataset", "subject_id", "y"):
        assert col in d.meta


def test_dataset_probe_target(store):
    d = build_classical(store, {"source": "source", "psd_kinds": ["log"]}, target="dataset")
    assert d.target == "y_dataset"
    assert (d.meta["y_dataset"] == (d.meta["dataset"] == "D2")).all()


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


