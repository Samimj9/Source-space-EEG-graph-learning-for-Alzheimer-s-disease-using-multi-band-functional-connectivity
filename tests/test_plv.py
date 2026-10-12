import numpy as np
import pytest
from scipy.signal import hilbert

from adgraph.connectivity.plv import band_analytic, phase_sync


def _analytic(sig):
    return hilbert(sig, axis=-1)


def test_identical_and_lagged_signals():
    t = np.arange(0, 20, 1 / 250)
    x = np.sin(2 * np.pi * 10 * t)
    rng = np.random.default_rng(1)
    sigs = np.stack([x, x, np.sin(2 * np.pi * 10 * t + np.pi / 2), rng.standard_normal(t.size)])
    plv, ciplv, signed = phase_sync(_analytic(sigs))
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


def test_matrix_properties_plv_ciplv_signed():
    rng = np.random.default_rng(3)
    t = np.arange(0, 20, 1 / 250)
    base = np.sin(2 * np.pi * 10 * t)
    sigs = np.stack([np.sin(2 * np.pi * 10 * t + ph) + 0.5 * rng.standard_normal(t.size)
                     for ph in rng.uniform(0, 2 * np.pi, 12)] + [base])
    plv, ciplv, signed = phase_sync(_analytic(sigs))
    for m in (plv, ciplv):
        assert np.allclose(m, m.T) and np.allclose(np.diag(m), 0)
        assert (m >= 0).all() and (m <= 1).all()
    assert np.allclose(signed, -signed.T) and np.allclose(np.diag(signed), 0)     # antisymmetric
    assert (signed >= -1).all() and (signed <= 1).all() and (signed < 0).any()
    assert np.allclose(ciplv, np.abs(signed))                                       # magnitude adjacency


def test_signed_ciplv_sign_follows_phase_offset_and_roi_order():
    """Sign = sign of sin(mean phase difference phi_j - phi_k); swapping the ROI order flips it."""
    t = np.arange(0, 20, 1 / 250)
    a = np.sin(2 * np.pi * 10 * t)
    b = np.sin(2 * np.pi * 10 * t - np.pi / 4)        # phi_a - phi_b = +pi/4
    _, _, signed = phase_sync(_analytic(np.stack([a, b])))
    assert signed[0, 1] > 0.5 and signed[1, 0] == pytest.approx(-signed[0, 1])
    _, _, swapped = phase_sync(_analytic(np.stack([b, a])))
    assert swapped[0, 1] == pytest.approx(-signed[0, 1])
    c = np.sin(2 * np.pi * 10 * t + 7 * np.pi / 4)     # same circular offset as -pi/4: sign is circular
    _, _, wrapped = phase_sync(_analytic(np.stack([a, c])))
    assert wrapped[0, 1] == pytest.approx(signed[0, 1], abs=1e-3)
