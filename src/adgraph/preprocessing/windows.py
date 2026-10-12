"""Fixed-length windowing and per-subject adaptive artifact rejection."""

from __future__ import annotations

import numpy as np


def window_starts(n_samples: int, sfreq: float, window_s: float, step_s: float, edge_trim_s: float) -> np.ndarray:
    """Start samples of all windows lying fully inside the edge-trimmed recording."""
    win, step, trim = (int(round(v * sfreq)) for v in (window_s, step_s, edge_trim_s))
    last_start = n_samples - trim - win
    if last_start < trim:
        return np.empty(0, dtype=int)
    return np.arange(trim, last_start + 1, step, dtype=int)


def reject_windows(data: np.ndarray, starts: np.ndarray, win_len: int, mad_k: float,
                   max_bad_channel_fraction: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Per-subject adaptive peak-to-peak rejection.

    threshold = median + mad_k * MAD over all (window, channel) peak-to-peak values
    of this subject; a window is rejected when more than ``max_bad_channel_fraction``
    of its channels exceed the threshold.

    Returns ``(keep_mask, threshold, ptp)`` with ``ptp`` of shape (n_windows, n_channels).
    """
    if len(starts) == 0:
        return np.zeros(0, dtype=bool), float("nan"), np.empty((0, data.shape[0]))
    ptp = np.stack([np.ptp(data[:, s:s + win_len], axis=1) for s in starts])
    med = np.median(ptp)
    thr = med + mad_k * np.median(np.abs(ptp - med))
    frac = (ptp > thr).mean(axis=1)
    return frac <= max_bad_channel_fraction, float(thr), ptp
