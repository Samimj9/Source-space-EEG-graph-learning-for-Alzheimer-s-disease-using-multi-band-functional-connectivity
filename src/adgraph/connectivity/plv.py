"""Band-limited phase synchrony: PLV and corrected imaginary PLV."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def transition_kwargs(l_freq: float, h_freq: float, transition_bandwidth_hz: float | None) -> dict:
    """MNE transition-band arguments: one width on both edges (lower capped at ``l_freq``),
    or MNE's 'auto' widths when ``transition_bandwidth_hz`` is None."""
    if transition_bandwidth_hz is None:
        return {}
    w = float(transition_bandwidth_hz)
    return {"l_trans_bandwidth": min(w, l_freq), "h_trans_bandwidth": w}


def band_analytic(x: np.ndarray, sfreq: float, l_freq: float, h_freq: float,
                  chunk_rows: int | None = None, transition_bandwidth_hz: float | None = None) -> np.ndarray:
    """Zero-phase FIR band-pass along the last axis, then the analytic signal.

    ``transition_bandwidth_hz``: width of both transition bands (configs/pipeline.yaml,
    ``connectivity.transition_bandwidth_hz``; None = MNE 'auto'). The nominal band edges are
    the passband edges in either case. Rows (ROIs) are filtered and transformed independently,
    so with ``chunk_rows`` they are processed a few at a time into one preallocated output:
    identical values, but the filter and FFT temporaries cover ``chunk_rows`` rows only."""
    from mne.filter import filter_data

    trans = transition_kwargs(l_freq, h_freq, transition_bandwidth_hz)

    def one(block):
        xf = filter_data(block, sfreq, l_freq, h_freq, method="fir", phase="zero", fir_design="firwin",
                         verbose="error", **trans)
        return hilbert(xf, axis=-1)

    if not chunk_rows or chunk_rows >= x.shape[0]:
        return one(x)
    out = np.empty(x.shape, dtype=np.complex128)
    for i in range(0, x.shape[0], chunk_rows):
        out[i:i + chunk_rows] = one(np.ascontiguousarray(x[i:i + chunk_rows], dtype=np.float64))
    return out


def phase_sync(analytic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PLV, |ciPLV| and signed ciPLV between all rows of an (n_nodes, n_times) analytic signal.

    C_jk = <e^{i(phi_j - phi_k)}>_t, computed as Z Z^H / T with Z = a / |a|.

    * PLV_jk = |C_jk|: symmetric, in [0, 1].
    * signed ciPLV_jk = Im C_jk / sqrt(1 - (Re C_jk)^2) (Bruna et al., 2018): in [-1, 1],
      **antisymmetric** (ciPLV_kj = -ciPLV_jk). Its sign is the sign of the non-zero-lag
      (imaginary) part of the mean phase relationship phi_j - phi_k, i.e. it is defined
      relative to the ROI ordering (j, k) and flips when the order is swapped. It does
      **not** establish directed connectivity or causal direction: phase differences are
      only defined modulo 2*pi, and common inputs, volume conduction residue and
      differing ROI time constants can all produce a consistent non-zero phase offset.
      The magnitude is insensitive to purely zero-lag coupling (field spread / leakage).
    * |ciPLV|: magnitude of the signed value; symmetric, in [0, 1].

    Handling (methods log #30): GNN adjacencies and connectivity strength use the
    symmetric, non-negative **magnitude** |ciPLV| (stored as ``ciplv``). The signed matrix
    is kept separately (``ciplv_signed``) for descriptive analyses of the sign of the
    non-zero-lag phase relationship; it is never used as an adjacency.
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


def band_filter_kernels(cfg: dict) -> dict[str, np.ndarray]:
    """The FIR kernels ``band_analytic`` applies for every configured band (for checks and docs)."""
    import mne

    sfreq = float(cfg["preprocessing"]["sfreq"])
    w = cfg["connectivity"].get("transition_bandwidth_hz")
    return {name: mne.filter.create_filter(None, sfreq, lo, hi, method="fir", phase="zero", fir_design="firwin",
                                           verbose="error", **transition_kwargs(lo, hi, w))
            for name, (lo, hi) in cfg["connectivity"]["bands"].items()}


def check_filters_fit_edge_trim(cfg: dict) -> None:
    """Every band filter's half-length must lie inside the edge trim, so filter start-up at the
    recording ends (continuous filtering) never reaches a kept window."""
    sfreq, trim = float(cfg["preprocessing"]["sfreq"]), float(cfg["windowing"]["edge_trim_s"])
    for name, h in band_filter_kernels(cfg).items():
        half = (len(h) // 2) / sfreq
        if half >= trim:
            raise ValueError(f"{name} band filter half-length {half:.2f} s >= edge trim {trim} s; "
                             "use a wider connectivity.transition_bandwidth_hz or a longer edge trim")
