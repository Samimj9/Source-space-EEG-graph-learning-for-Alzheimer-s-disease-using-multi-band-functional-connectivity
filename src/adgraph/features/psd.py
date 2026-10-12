"""Spectral power features per window (source ROIs or sensors).

Implementation
--------------
1. A power spectral density is estimated for every row of an (n_signals, n_times)
   window with one of the configurable estimators:

   * ``welch``      scipy.signal.welch: ``segment_s``-long tapered segments with
                    fractional ``overlap``, averaged by ``mean`` or ``median``.
                    Default: 2 s Hann segments, 50 % overlap, 0.5 Hz resolution.
   * ``multitaper`` mne.time_frequency.psd_array_multitaper with half-bandwidth
                    ``bandwidth_hz / 2`` (DPSS tapers, adaptive weighting off).
   * ``periodogram`` scipy.signal.periodogram with a Hann taper over the whole window.

   All estimators return a one-sided density (units^2 / Hz) with linear detrending
   of each segment ('constant' for multitaper, which demeans).

2. Absolute band power = sum of PSD bins with ``lo <= f < hi`` times the bin width
   (rectangle rule on half-open bands, so adjacent bands never share a bin).
3. Log band power = log10(absolute band power).
4. Relative band power = absolute band power / total power in ``relative_range``
   (same half-open rule, so bins 1.0 ... 44.5 Hz at 0.5 Hz resolution).

Relative-power denominator vs. preprocessing passband
-----------------------------------------------------
The denominator must only contain frequencies that preprocessing preserves.
Preprocessing is a zero-phase FIR band-pass 0.1–45 Hz (MNE ``firwin``), where 0.1
and 45 Hz are the *passband edges*. Its response, measured at 250 Hz
(tests/test_psd.py::test_relative_range_inside_preserved_passband):

    0.1 Hz -0.03 dB | 1 Hz -0.00 dB | 44.5 Hz -0.01 dB | 45 Hz -0.04 dB
    47 Hz -0.6 dB   | 50.6 Hz -6 dB (cutoff) | 56.25 Hz -48 dB (stopband)

So 1–45 Hz lies entirely inside the preserved passband and is kept unchanged. The lower
limit of 1 Hz (rather than 0.1 Hz) excludes the slow-drift region that the 0.1 Hz
high-pass only partly removes; the reason for choosing exactly 1 Hz requires author
confirmation. ``check_relative_range`` makes the pipeline fail, rather than silently
adapt, if the range is ever set outside the configured passband.

The features are stateless per window (no fitting), so they can be computed for
all windows before any train/test split without leakage.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

ESTIMATORS = ("welch", "multitaper", "periodogram")


def compute_psd(x: np.ndarray, sfreq: float, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """PSD of each row of ``x`` (…, n_times). Returns ``(freqs, psd)`` with psd shape (…, n_freqs)."""
    method = cfg["method"]
    if method == "welch":
        p = cfg.get("welch", {})
        nperseg = int(round(p.get("segment_s", 2.0) * sfreq))
        nperseg = min(nperseg, x.shape[-1])
        noverlap = int(round(nperseg * p.get("overlap", 0.5)))
        freqs, psd = signal.welch(x, fs=sfreq, window=p.get("window", "hann"), nperseg=nperseg,
                                  noverlap=noverlap, detrend="linear", average=p.get("average", "mean"), axis=-1)
    elif method == "periodogram":
        freqs, psd = signal.periodogram(x, fs=sfreq, window=cfg.get("periodogram", {}).get("window", "hann"),
                                        detrend="linear", axis=-1)
    elif method == "multitaper":
        from mne.time_frequency import psd_array_multitaper

        p = cfg.get("multitaper", {})
        psd, freqs = psd_array_multitaper(x, sfreq, fmin=0.0, fmax=sfreq / 2, bandwidth=p.get("bandwidth_hz", 1.0),
                                          adaptive=False, normalization="full", verbose="error")
    else:
        raise ValueError(f"unknown PSD method {method!r}; choose from {ESTIMATORS}")
    return freqs, psd


def check_relative_range(relative_range: list[float], l_freq: float, h_freq: float) -> None:
    """Raise if the relative-power denominator extends outside the preprocessing passband."""
    lo, hi = relative_range
    if not (l_freq <= lo < hi <= h_freq):
        raise ValueError(f"psd.relative_range {relative_range} Hz is not inside the preprocessing passband "
                         f"[{l_freq}, {h_freq}] Hz; choose a denominator range that preprocessing preserves")


def _band_sum(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        raise ValueError(f"no PSD bins in [{lo}, {hi}) Hz; frequency resolution too coarse")
    df = freqs[1] - freqs[0]
    return psd[..., mask].sum(axis=-1) * df


def band_powers(freqs: np.ndarray, psd: np.ndarray, bands: dict[str, list[float]],
                relative_range: list[float]) -> dict[str, np.ndarray]:
    """Absolute, log10 and relative power per band.

    Returns arrays of shape (…, n_bands) under keys ``abs``, ``log``, ``rel``, with
    the band axis ordered as ``bands``.
    """
    absolute = np.stack([_band_sum(freqs, psd, lo, hi) for lo, hi in bands.values()], axis=-1)
    total = _band_sum(freqs, psd, *relative_range)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        log = np.log10(absolute)
        rel = absolute / total
    return {"abs": absolute, "log": log, "rel": rel}


def window_band_powers(x: np.ndarray, sfreq: float, bands: dict[str, list[float]], cfg: dict) -> dict[str, np.ndarray]:
    """Convenience: PSD + band powers for one (n_signals, n_times) window -> (n_bands, n_signals) arrays."""
    freqs, psd = compute_psd(x, sfreq, cfg)
    out = band_powers(freqs, psd, bands, cfg["relative_range"])
    return {k: np.moveaxis(v, -1, 0) for k, v in out.items()}
