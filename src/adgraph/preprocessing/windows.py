"""Fixed-length windowing and per-subject adaptive artifact rejection."""

from __future__ import annotations

import numpy as np


def window_samples(sfreq: float, window_s: float, step_s: float) -> tuple[int, int]:
    """Window length and step in samples; refuses non-integer sample counts."""
    win, step = window_s * sfreq, step_s * sfreq
    if abs(win - round(win)) > 1e-9 or abs(step - round(step)) > 1e-9:
        raise ValueError(f"window {window_s}s / step {step_s}s are not whole samples at {sfreq} Hz")
    return int(round(win)), int(round(step))


def expected_n_windows(duration_s: float, window_s: float, step_s: float, edge_trim_s: float) -> int:
    """Number of candidate windows (before rejection) for a recording of ``duration_s``.

    n = floor((duration - 2 * trim - window) / step) + 1, or 0 if the trimmed
    recording is shorter than one window. Primary setting (20 s, 10 s step,
    10 s trim): a 300 s recording gives floor((300 - 20 - 20) / 10) + 1 = 27.
    """
    usable = duration_s - 2 * edge_trim_s - window_s
    return 0 if usable < 0 else int(np.floor(usable / step_s + 1e-9)) + 1


def window_starts(n_samples: int, sfreq: float, window_s: float, step_s: float, edge_trim_s: float) -> np.ndarray:
    """Start samples of all windows lying fully inside the edge-trimmed recording."""
    win, step = window_samples(sfreq, window_s, step_s)
    trim = int(round(edge_trim_s * sfreq))
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
