"""Band-limited phase synchrony: PLV and corrected imaginary PLV."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def band_analytic(x: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    """Zero-phase FIR band-pass along the last axis, then the analytic signal."""
    from mne.filter import filter_data

    xf = filter_data(x, sfreq, l_freq, h_freq, method="fir", phase="zero", fir_design="firwin", verbose="error")
    return hilbert(xf, axis=-1)


def phase_sync(analytic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PLV, |ciPLV| and signed ciPLV between all rows of an (n_nodes, n_times) analytic signal.

    C_jk = <e^{i(phi_j - phi_k)}>_t, computed as Z Z^H / T with Z = a / |a|.

    * PLV_jk = |C_jk|: symmetric, in [0, 1].
    * signed ciPLV_jk = Im C_jk / sqrt(1 - (Re C_jk)^2) (Bruna et al., 2018): in [-1, 1],
      **antisymmetric** (ciPLV_kj = -ciPLV_jk); the sign says which node leads in phase.
      Insensitive to zero-lag coupling such as field spread / source leakage.
    * |ciPLV|: magnitude of the signed value; symmetric, in [0, 1].

    Handling (methods log #33): GNN adjacencies and connectivity strength use the
    **magnitude** |ciPLV| (stored as ``ciplv``), because none of the models is a signed or
    directed graph model. The signed matrix is kept separately (``ciplv_signed``) for
    analyses that need the lead/lag direction; it is never used as an adjacency.
    Diagonals are 0 (no self-loops).
    """
    mag = np.abs(analytic)
    z = analytic / np.where(mag == 0, 1, mag)
    c = z @ z.conj().T / analytic.shape[-1]
    plv = np.clip(np.abs(c), 0.0, 1.0)
    signed = np.clip(c.imag / np.sqrt(np.clip(1 - c.real ** 2, 1e-12, None)), -1.0, 1.0)
    signed = (signed - signed.T) / 2          # exact antisymmetry despite rounding
    np.fill_diagonal(plv, 0.0)
    np.fill_diagonal(signed, 0.0)
    plv = (plv + plv.T) / 2
    return plv, np.abs(signed), signed
