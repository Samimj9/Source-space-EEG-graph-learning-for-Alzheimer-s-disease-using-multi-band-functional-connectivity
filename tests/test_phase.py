import numpy as np
from scipy.signal import hilbert

from adgraph.connectivity.phase import band_analytic, log_power, phase_sync


def _analytic(sig):
    return hilbert(sig, axis=-1)


def test_identical_and_lagged_signals():
    t = np.arange(0, 20, 1 / 250)
    x = np.sin(2 * np.pi * 10 * t)
    rng = np.random.default_rng(1)
    sigs = np.stack([x, x, np.sin(2 * np.pi * 10 * t + np.pi / 2), rng.standard_normal(t.size)])
    plv, ciplv = phase_sync(_analytic(sigs))
    assert plv.shape == (4, 4) and np.allclose(np.diag(plv), 0)
    assert np.allclose(plv, plv.T)
    assert plv[0, 1] > 0.99 and ciplv[0, 1] < 0.05      # zero lag: PLV high, ciPLV ~0
    assert plv[0, 2] > 0.99 and ciplv[0, 2] > 0.95      # 90 deg lag: both high
    assert plv[0, 3] < 0.2
    assert (ciplv >= 0).all() and (ciplv <= 1).all()


def test_band_analytic_isolates_band():
    sf = 250
    t = np.arange(0, 30, 1 / sf)
    x = np.sin(2 * np.pi * 6 * t) + np.sin(2 * np.pi * 20 * t)
    theta = band_analytic(x[None], sf, 4, 8)
    beta = band_analytic(x[None], sf, 13, 30)
    mid = slice(5 * sf, 25 * sf)
    assert np.allclose(np.abs(theta[0, mid]), 1, atol=0.05)
    assert np.allclose(np.abs(beta[0, mid]), 1, atol=0.05)
    assert log_power(theta[:, mid]).shape == (1,)
