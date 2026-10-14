"""Boundary-aware processing (pipeline v4; methodology log entry 41).

Boundary-Aware Filtering and Hilbert Phase Estimation for Source-Space EEG Functional Connectivity:

1. **Crop first.** The ``windowing.edge_trim_s`` at each end of the recording (10 s) is discarded at the
   original sampling rate, *before* resampling and any filtering. Discarded samples (which carry the
   large start/end transients) therefore never enter any filter, the interpolation, the reference, the
   source operator or the phase estimation.
2. **Explicit padding.** Every FIR filter applied to the cropped continuous signal uses MNE's
   ``reflect_limited`` padding (``boundary.filter_pad``): the signal is extended by its mirror image
   about the end sample (limited to the signal length), followed by zeros.
3. **Local analytic signal.** Band-pass + Hilbert transform is computed as one finite complex FIR
   (``connectivity.plv.band_analytic_fir``), so each analytic sample depends on a known, finite stretch
   of input; no FFT wrap-around.
4. **Boundary margin.** Windows must lie at least ``margin`` samples inside the cropped signal, where
   the margin is computed from the actual filters (``margins``). With ``boundary.margin_policy:
   full_support`` it is the half-length of the 0.1-45 Hz sensor band-pass plus the half-length of the
   analytic kernel, so no sample of an accepted window depends on padded data through any FIR in the
   chain. ``feature_band`` instead uses the analytic half-length plus the half-length of the 45 Hz
   low-pass part only (padding then reaches retained windows only through components below ~0.2 Hz).
5. **Window grid.** 20 s windows with a 10 s step tile the valid interval [margin, n - margin) starting
   at its first sample; window start samples are stored relative to the original recording.

The same rules apply to every recording of both datasets.
"""

from __future__ import annotations

import numpy as np

POLICIES = ("full_support", "feature_band")


def boundary_aware(cfg: dict) -> bool:
    """True for configurations with a ``boundary`` section (pipeline v4)."""
    return bool(cfg.get("boundary"))


def filter_pad(cfg: dict) -> dict:
    """Keyword arguments making the padding of MNE filters explicit (empty for v3 configurations)."""
    return {"pad": cfg["boundary"]["filter_pad"]} if boundary_aware(cfg) else {}


def crop_edges(raw, cfg: dict) -> dict:
    """Discard ``edge_trim_s`` at both ends of a loaded recording (in place, original sampling rate).
    Returns where the kept part starts and stops in the original recording (seconds)."""
    trim = float(cfg["windowing"]["edge_trim_s"])
    sf = raw.info["sfreq"]
    dur = raw.n_times / sf
    if dur <= 2 * trim:
        raise ValueError(f"recording of {dur:.1f} s is not longer than twice the edge trim ({trim} s)")
    n_orig = raw.n_times
    first, stop = int(round(trim * sf)), n_orig - int(round(trim * sf))
    raw.crop(tmin=first / sf, tmax=(stop - 1) / sf, include_tmax=True, verbose="error")
    if raw.n_times != stop - first:
        raise RuntimeError(f"crop kept {raw.n_times} samples, expected {stop - first}")
    return {"crop_start_s": first / sf, "crop_stop_s": stop / sf, "orig_n_times": int(n_orig),
            "orig_duration_s": dur}


def sensor_bandpass_applied(cfg: dict) -> bool:
    """False for provider-preprocessed data (Dataset 3: ``preprocessing.bandpass: false``)."""
    return bool(cfg["preprocessing"].get("bandpass", True))


def sensor_filter_half(cfg: dict) -> int:
    """Half-length (samples at preprocessing.sfreq) of the 0.1-45 Hz sensor band-pass (0 if none is applied)."""
    import mne

    pp = cfg["preprocessing"]
    if not sensor_bandpass_applied(cfg):
        return 0
    return len(mne.filter.create_filter(None, pp["sfreq"], pp["l_freq"], pp["h_freq"], fir_design="firwin",
                                        verbose="error")) // 2


def sensor_lowpass_half(cfg: dict) -> int:
    """Half-length of the low-pass (h_freq) part of the sensor band-pass alone (0 if none is applied)."""
    import mne

    pp = cfg["preprocessing"]
    if not sensor_bandpass_applied(cfg):
        return 0
    return len(mne.filter.create_filter(None, pp["sfreq"], None, pp["h_freq"], fir_design="firwin",
                                        verbose="error")) // 2


def margins(cfg: dict) -> dict:
    """Boundary margin (samples at preprocessing.sfreq) and its components, from the actual filters."""
    from ..connectivity.plv import analytic_kernel

    sf = float(cfg["preprocessing"]["sfreq"])
    tbw = cfg["connectivity"].get("transition_bandwidth_hz")
    analytic = max(len(analytic_kernel(sf, float(lo), float(hi), tbw)) // 2
                   for lo, hi in cfg["connectivity"]["bands"].values())
    policy = cfg["boundary"]["margin_policy"]
    if policy not in POLICIES:
        raise ValueError(f"boundary.margin_policy must be one of {POLICIES}, got {policy!r}")
    sensor = sensor_filter_half(cfg) if policy == "full_support" else sensor_lowpass_half(cfg)
    total = sensor + analytic
    return {"policy": policy, "sensor_filter_half_samples": sensor, "analytic_half_samples": analytic,
            "margin_samples": total, "margin_s": total / sf, "sfreq": sf}


def valid_window_starts(n_samples: int, cfg: dict, margin: int | None = None) -> np.ndarray:
    """Window start samples (relative to the cropped signal) of all windows lying inside
    [margin, n_samples - margin), on a grid starting at ``margin`` with the configured step."""
    from .windows import window_samples

    sf = float(cfg["preprocessing"]["sfreq"])
    win, step = window_samples(sf, cfg["windowing"]["window_s"], cfg["windowing"]["step_s"])
    m = margins(cfg)["margin_samples"] if margin is None else int(margin)
    last = n_samples - m - win
    return np.arange(m, last + 1, step, dtype=int) if last >= m else np.empty(0, dtype=int)


def expected_valid_windows(duration_s: float, cfg: dict) -> int:
    """Candidate windows of a recording of ``duration_s`` under the v4 rules (before rejection)."""
    sf = float(cfg["preprocessing"]["sfreq"])
    n = int(round((duration_s - 2 * float(cfg["windowing"]["edge_trim_s"])) * sf))
    return len(valid_window_starts(n, cfg))


def report(cfg: dict) -> dict:
    """Everything about the boundary handling that a manifest must record."""
    from ..connectivity.plv import analytic_kernel_report

    sf = float(cfg["preprocessing"]["sfreq"])
    tbw = cfg["connectivity"].get("transition_bandwidth_hz")
    m = margins(cfg)
    return {**cfg["boundary"], **m,
            "edge_trim_s": cfg["windowing"]["edge_trim_s"],
            "sensor_bandpass_applied": sensor_bandpass_applied(cfg),
            "sensor_bandpass_taps": 2 * sensor_filter_half(cfg) + 1 if sensor_bandpass_applied(cfg) else 0,
            "sensor_lowpass_taps": 2 * sensor_lowpass_half(cfg) + 1 if sensor_bandpass_applied(cfg) else 0,
            "analytic_kernels": {b: analytic_kernel_report(sf, float(lo), float(hi), tbw)
                                 for b, (lo, hi) in cfg["connectivity"]["bands"].items()},
            "window_grid": "start at the first sample of [margin, n - margin) of the cropped signal; "
                           "start_sample stored relative to the original recording"}
