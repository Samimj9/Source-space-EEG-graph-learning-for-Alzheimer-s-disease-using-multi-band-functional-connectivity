"""Connectivity band-pass design (methodology log entry 36): regression tests on synthetic signals.

The nominal bands are unchanged; the configured transition width (1.0 Hz on every edge)
must keep each band's passband on its nominal edges, keep cross-band leakage between alpha1 and
alpha2 at the noise floor, and keep filter start-up inside the edge trim.
"""

import numpy as np
import pytest
from scipy.signal import freqz

from adgraph.config import load_yaml
from adgraph.connectivity.plv import (band_analytic, band_filter_kernels, check_filters_fit_edge_trim, phase_sync,
                                      transition_kwargs)

CFG = load_yaml("configs/pipeline.yaml")
SF = float(CFG["preprocessing"]["sfreq"])
BANDS = CFG["connectivity"]["bands"]
F = np.arange(0.0, 45.0, 0.01)


def test_nominal_bands_and_transition_width_are_the_frozen_values():
    assert BANDS == {"theta": [4.0, 8.0], "alpha1": [8.0, 10.0], "alpha2": [10.0, 13.0], "beta": [13.0, 30.0]}
    assert CFG["connectivity"]["transition_bandwidth_hz"] == 1.0
    assert transition_kwargs(4.0, 8.0, 1.0) == {"l_trans_bandwidth": 1.0, "h_trans_bandwidth": 1.0}
    assert transition_kwargs(0.5, 8.0, 1.0)["l_trans_bandwidth"] == 0.5        # capped at l_freq
    assert transition_kwargs(4.0, 8.0, None) == {}                               # MNE 'auto' (before)


def _gain(h):
    return np.abs(freqz(h, worN=F, fs=SF)[1])


def test_passbands_sit_on_the_nominal_edges():
    for name, h in band_filter_kernels(CFG).items():
        lo, hi = BANDS[name]
        g = _gain(h)
        inside = (F >= lo + 0.15) & (F <= hi - 0.15)
        assert g[inside].min() > 10 ** (-0.1 / 20), name                          # flat (>= -0.1 dB) inside
        half = F[g >= 0.5]
        assert abs(half.min() - (lo - 0.5)) < 0.05 and abs(half.max() - (hi + 0.5)) < 0.05, name   # -6 dB at edge -/+ w/2
        assert g[(F < lo - 1.0) | (F > hi + 1.0)].max() < 10 ** (-50 / 20), name  # stopband beyond the transition


def test_alpha1_alpha2_overlap_is_limited_to_the_shared_edge():
    k = band_filter_kernels(CFG)
    both = F[(_gain(k["alpha1"]) >= 0.5) & (_gain(k["alpha2"]) >= 0.5)]
    assert both.min() >= 9.49 and both.max() <= 10.51                           # was 8.76-11.25 Hz with 'auto'
    p1 = _gain(k["alpha1"]) ** 2
    assert np.trapezoid(np.where((F >= 10) & (F < 13), p1, 0), F) / np.trapezoid(p1, F) < 0.15   # was 0.26


def test_filters_fit_inside_the_edge_trim():
    check_filters_fit_edge_trim(CFG)
    for h in band_filter_kernels(CFG).values():
        assert (len(h) // 2) / SF < CFG["windowing"]["edge_trim_s"]
    too_narrow = {**CFG, "connectivity": {**CFG["connectivity"], "transition_bandwidth_hz": 0.1}}
    with pytest.raises(ValueError, match="edge trim"):
        check_filters_fit_edge_trim(too_narrow)


def _coupled(freq, seconds=120, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SF)
    t = np.arange(n) / SF
    X = np.fft.rfft(rng.standard_normal((2, n)), axis=-1)
    X[:, 1:] /= np.sqrt(np.fft.rfftfreq(n, 1 / SF)[1:])                         # independent 1/f noise
    noise = np.fft.irfft(X, n, axis=-1)
    noise /= noise.std(axis=1, keepdims=True)
    if freq is None:
        return noise
    phase = 2 * np.pi * freq * t + np.cumsum(rng.standard_normal(n)) * 0.02
    return noise + 0.6 * np.vstack([np.sin(phase), np.sin(phase - np.pi / 3)])   # non-zero lag


def _mean_sync(x, band, w):
    lo, hi = BANDS[band]
    a = band_analytic(x, SF, lo, hi, transition_bandwidth_hz=w)
    win, trim = int(20 * SF), int(10 * SF)
    vals = np.array([phase_sync(a[:, s:s + win])[:2] for s in range(trim, x.shape[1] - trim - win + 1, win // 2)])
    return vals[:, 0, 0, 1].mean(), vals[:, 1, 0, 1].mean()                    # PLV, |ciPLV| of the pair


def test_band_specific_coupling_stays_in_its_band():
    w = CFG["connectivity"]["transition_bandwidth_hz"]
    null_plv = max(_mean_sync(_coupled(None), b, w)[0] for b in ("alpha1", "alpha2"))
    for freq, own, other in ((9.0, "alpha1", "alpha2"), (11.5, "alpha2", "alpha1")):
        x = _coupled(freq)
        plv_own, ci_own = _mean_sync(x, own, w)
        plv_other, ci_other = _mean_sync(x, other, w)
        assert plv_own > 0.8 and ci_own > 0.7, (freq, plv_own, ci_own)          # coupling recovered in its band
        assert plv_other < null_plv + 0.05 and ci_other < 0.15, (freq, plv_other, ci_other)   # at the noise floor
    # the previous MNE 'auto' design leaked 9 Hz coupling into alpha2 (documented, docs/band_filters.md)
    assert _mean_sync(_coupled(9.0), "alpha2", None)[0] > 0.5


def test_pipeline_applies_the_configured_transition_width(monkeypatch):
    import adgraph.pipeline as pl

    seen = []
    real = pl.band_analytic

    def spy(*a, **k):
        seen.append(k.get("transition_bandwidth_hz"))
        return real(*a, **k)

    monkeypatch.setattr(pl, "band_analytic", spy)

    class Raw:
        info = {"sfreq": SF}

        def get_data(self):
            return np.random.default_rng(0).standard_normal((30, int(80 * SF))) * 1e-5

    pl.process_recording(Raw(), np.random.default_rng(1).standard_normal((68, 30)), CFG)
    assert seen == [1.0] * len(BANDS)
