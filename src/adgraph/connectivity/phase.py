"""Band-limited phase synchrony: PLV and corrected imaginary PLV."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def band_analytic(x: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    """Zero-phase FIR band-pass along the last axis, then the analytic signal."""
    from mne.filter import filter_data

    xf = filter_data(x, sfreq, l_freq, h_freq, method="fir", phase="zero", fir_design="firwin", verbose="error")
    return hilbert(xf, axis=-1)


def phase_sync(analytic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PLV and |ciPLV| between all rows of an (n_nodes, n_times) analytic signal.

    PLV = |<e^{i(phi_j - phi_k)}>_t|, computed as Z Z^H / T with Z = a / |a|.
    ciPLV = |Im C| / sqrt(1 - (Re C)^2) (Bruna et al., 2018), insensitive to
    zero-lag coupling such as field spread / source leakage.
    Diagonals are set to 0 (no self-loops).
    """
    mag = np.abs(analytic)
    z = analytic / np.where(mag == 0, 1, mag)
    c = z @ z.conj().T / analytic.shape[-1]
    plv = np.abs(c)
    ciplv = np.abs(c.imag) / np.sqrt(np.clip(1 - c.real ** 2, 1e-12, None))
    np.fill_diagonal(plv, 0.0)
    np.fill_diagonal(ciplv, 0.0)
    return plv, np.clip(ciplv, 0.0, 1.0)


def log_power(analytic: np.ndarray) -> np.ndarray:
    """log10 of mean instantaneous power per row."""
    return np.log10(np.mean(np.abs(analytic) ** 2, axis=-1))
