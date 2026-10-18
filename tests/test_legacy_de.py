"""Differential entropy in the legacy graph builder (adgraph.legacy.pipeline; methodology log entry 58).

The store schema lists DE among the PSD kinds (graphs.store.PSD_KINDS). The builder computes it with the definition
of the LCMV / wMNE stores (sourcegraphs.features): DE = 0.5 ln(2 pi e var(x_band)), x_band the real part of the band
analytic signal of the window (the same band filter as PLV / |ciPLV|), for the source ROIs and the sensors. Stores
built before DE existed still load and serve every other feature."""

import numpy as np
import pytest

from adgraph.config import load_yaml
from adgraph.graphs.store import PSD_KINDS, GraphStore

CFG = load_yaml("configs/pipeline.yaml")
SF = CFG["preprocessing"]["sfreq"]


class Raw:
    info = {"sfreq": SF}

    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data


def _run(cfg=CFG):
    import adgraph.legacy.pipeline as pl

    rng = np.random.default_rng(0)
    data = rng.standard_normal((30, int(80 * SF))) * 1e-5
    W = rng.standard_normal((68, 30))
    res = pl.process_recording(Raw(data), W, cfg, offset=int(cfg["windowing"]["edge_trim_s"] * SF))
    return pl, data, W, res


def test_every_psd_kind_of_the_schema_is_produced():
    _, _, _, res = _run()
    n_w, n_b = res["plv"].shape[:2]
    for prefix, n in (("psd", 68), ("sensor_psd", 30)):
        for k in PSD_KINDS:
            arr = res[f"{prefix}_{k}"]
            assert arr.shape == (n_w, n_b, n) and np.isfinite(arr).all(), (prefix, k)


def test_de_is_the_gaussian_entropy_of_the_band_signal():
    pl, data, W, res = _run()
    from adgraph.connectivity.plv import band_analytic, band_analytic_fir

    fn = band_analytic_fir if CFG.get("boundary") else band_analytic
    tbw = CFG["connectivity"].get("transition_bandwidth_hz")
    win = int(round(CFG["windowing"]["window_s"] * SF))
    starts = res["start_sample"] - int(CFG["windowing"]["edge_trim_s"] * SF)
    for b, (lo, hi) in enumerate(CFG["connectivity"]["bands"].values()):
        a_roi = fn(W @ data, SF, lo, hi, chunk_rows=8, transition_bandwidth_hz=tbw)
        a_sen = fn(data, SF, lo, hi, chunk_rows=8, transition_bandwidth_hz=tbw)
        for w, s in enumerate(starts):
            want = 0.5 * np.log(2 * np.pi * np.e * np.var(a_roi[:, s:s + win].real, axis=1))
            assert np.allclose(res["psd_de"][w, b], want, rtol=0, atol=1e-5)
            want = 0.5 * np.log(2 * np.pi * np.e * np.var(a_sen[:, s:s + win].real, axis=1))
            assert np.allclose(res["sensor_psd_de"][w, b], want, rtol=0, atol=1e-5)


def test_de_agrees_with_band_power():
    """var(x_band) and the Welch band power estimate the same band variance: DE is close to
    0.5 ln(2 pi e P_band) (not equal: the 1 Hz filter transitions and the Welch estimate differ slightly)."""
    _, _, _, res = _run()
    for prefix in ("psd", "sensor_psd"):
        d = res[f"{prefix}_de"] - 0.5 * np.log(2 * np.pi * np.e * res[f"{prefix}_abs"])
        assert np.abs(np.median(d)) < 0.3


def test_sensor_de_follows_the_sensor_switch():
    cfg = {**CFG, "psd": {**CFG["psd"], "sensor": False}}
    _, _, _, res = _run(cfg)
    assert not any(k.startswith("sensor_psd") for k in res if isinstance(res[k], np.ndarray) and
                   k in ("sensor_psd_abs", "sensor_psd_de"))
    assert "psd_de" in res


def test_stores_built_before_de_still_load_and_serve_other_features(tmp_path):
    from adgraph.graphs.dataset import build_classical, build_graph
    from adgraph.legacy.synthetic import make_synthetic_store

    st = make_synthetic_store(psd_effect=0.5, seed=0)
    old = GraphStore({k: v for k, v in st.arrays.items() if not k.endswith("_de")}, st.windows, st.bands,
                     st.band_edges, st.roi_names, st.channels, st.subjects, st.manifest)
    d = old.save(tmp_path / "pre_de_store")
    loaded = GraphStore.load(d)
    assert "psd_de" not in loaded.arrays and "psd_log" in loaded.arrays
    assert build_classical(loaded, {"source": "source", "psd_kinds": ["log", "rel"],
                                    "bands": loaded.bands}).payload["X"].shape[0] == loaded.n_windows
    build_graph(loaded, {"connectivity": "plv", "bands": loaded.bands, "band_mode": "multi"},
                {"components": ["psd"], "psd_kinds": ["log"]})
    with pytest.raises(ValueError, match="built before"):
        build_classical(loaded, {"source": "source", "psd_kinds": ["de"], "bands": loaded.bands})
