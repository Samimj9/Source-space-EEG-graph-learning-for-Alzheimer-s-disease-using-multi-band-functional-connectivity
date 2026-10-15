"""Discontinuity-aware processing of recordings with EEGLAB ``boundary`` events (Dataset 3; methodology log
entry 45).

How MNE represents them (``mne.io.eeglab._read_annotations_eeglab``, MNE 1.13.2): every EEGLAB event becomes an
annotation with ``onset = (latency - 1) / srate`` (seconds from the first sample; EEGLAB latencies are 1-based
and a boundary sits at ``k + 0.5``, i.e. between two samples) and ``duration = event.duration / srate``. For a
boundary event, EEGLAB's ``duration`` is the number of samples that were REMOVED there; those samples are not in
the file, so the duration is irrelevant to the file's own time line. Empty durations are read as NaN, many are 0;
events with a NaN latency are dropped by MNE. The description stays ``boundary``; MNE does not treat it as a bad
segment, so nothing downstream skips it automatically.

Rule implemented here: a discontinuity at fractional sample position ``p = onset * sfreq`` splits the recording
before sample ``ceil(p)`` (between ``k - 1`` and ``k`` for ``p = k - 0.5``). The recording is split into continuous
segments; each segment is resampled (and filtered, if the configuration filters) ON ITS OWN, so no operation that
assumes continuity spreads information across a discontinuity; windows are placed inside each segment only, at
least the boundary margin (the same ``preprocessing.boundary.margins``) from both segment ends, on the 20 s /
10 s grid starting at the segment's first valid sample. Segments too short for one window are not processed.
Instantaneous operations (average reference, the ROI operator) are applied to the concatenated segments; the
analytic FIR is applied to the concatenation as well, which is exact for every accepted window because its
support (half-length <= margin) never leaves the window's own segment.
"""

from __future__ import annotations

import numpy as np


def enabled(cfg: dict) -> bool:
    """True if the configuration splits recordings at EEGLAB boundary events (``discontinuities.handling``)."""
    return (cfg.get("discontinuities") or {}).get("handling") == "split"


def boundary_cuts(raw, description: str = "boundary") -> tuple[np.ndarray, int]:
    """(sorted unique cut positions = index of the first sample after each discontinuity, strictly inside the
    recording; number of boundary events in the file, including any at its very start or end)."""
    ann = raw.annotations
    mask = np.array([str(d).strip().lower() == description for d in ann.description], dtype=bool)
    p = (np.asarray(ann.onset, dtype=float)[mask] - raw.first_time) * raw.info["sfreq"]
    cuts = np.unique(np.ceil(p - 1e-6).astype(int))
    return cuts[(cuts > 0) & (cuts < raw.n_times)], int(mask.sum())


def split(n_samples: int, cuts) -> list[tuple[int, int]]:
    """Continuous [start, stop) segments of a signal of ``n_samples`` cut before each position in ``cuts``."""
    edges = [0, *sorted(int(c) for c in cuts if 0 < c < n_samples), n_samples]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def min_segment_samples(cfg: dict) -> int:
    """Shortest segment (samples at ``preprocessing.sfreq``) that holds one window plus both margins."""
    from .boundary import margins
    from .windows import window_samples

    win, _ = window_samples(float(cfg["preprocessing"]["sfreq"]), cfg["windowing"]["window_s"],
                            cfg["windowing"]["step_s"])
    return win + 2 * margins(cfg)["margin_samples"]


def exact_length(n: int, sf_in: float, sf_out: float) -> int:
    """Largest length <= ``n`` that resamples to a whole number of samples (500 -> 250 Hz: even lengths).

    MNE's FFT resampling of other lengths spreads the output over the segment's exact duration, a sub-sample
    time shift common to all channels; trimming at most ``sf_in / gcd`` - 1 samples at the segment end avoids it."""
    import math

    step = int(round(sf_in)) // math.gcd(int(round(sf_in)), int(round(sf_out)))
    return (n // step) * step


def resample_npad(sf_in: float, seconds: float = 4.0) -> int:
    """Padding (samples per side, even) used when resampling each segment."""
    return int(round(seconds * sf_in)) // 2 * 2


def resampled_length(n: int, sf_in: float, sf_out: float) -> int:
    """Number of samples MNE's resampling produces for ``n`` input samples."""
    return int(round(n * sf_out / sf_in))


def plan(raw, crop_first: int, cuts: np.ndarray, cfg: dict) -> dict:
    """Segments of the cropped recording (original rate) and which of them can hold windows.

    ``crop_first``: samples (original rate) removed at the start by ``boundary.crop_edges``; ``cuts`` are positions
    in the UNCROPPED recording. Returns the cuts inside the analysed interval, every segment, the usable ones, and
    the counts that the report needs."""
    from .boundary import valid_window_starts

    sf, sf_new = raw.info["sfreq"], float(cfg["preprocessing"]["sfreq"])
    n = raw.n_times                                            # cropped length, original rate
    inside = np.asarray([c - crop_first for c in cuts if 0 < c - crop_first < n], dtype=int)
    segs = split(n, inside)
    need = min_segment_samples(cfg)
    segs = [(a, a + exact_length(b - a, sf, sf_new)) if exact_length(b - a, sf, sf_new) else (a, b) for a, b in segs]
    usable = [s for s in segs if resampled_length(s[1] - s[0], sf, sf_new) >= need]
    # what the continuous v4 grid would have placed, and how many of those windows would cross a cut
    n_new = resampled_length(n, sf, sf_new)
    cont = valid_window_starts(n_new, cfg)
    win = int(round(float(cfg["windowing"]["window_s"]) * sf_new))
    spanning = int(windows_crossing(cont, cont + win, inside * sf_new / sf, 0, tol=0.0).sum())
    return {"cuts_cropped": inside, "segments": segs, "usable": usable, "continuous_candidates": int(len(cont)),
            "continuous_windows_spanning_cut": spanning, "sfreq_in": sf}


def process_segments(raw, usable: list[tuple[int, int]], cfg: dict, offset_new: int):
    """Resample (and band-pass, if configured) every usable segment independently and concatenate them.

    Returns ``(raw_new, segments)`` where ``segments`` is a list of dicts with ``proc_start``/``proc_stop`` (samples
    of the concatenated signal) and ``orig_start``/``orig_stop`` (samples at the new rate, relative to the ORIGINAL
    recording: ``offset_new`` = where the cropped signal starts)."""
    import mne

    from .boundary import filter_pad

    pp = cfg["preprocessing"]
    sf = raw.info["sfreq"]
    parts, segments, pos = [], [], 0
    for a, b in usable:
        seg = _slice(raw, a, b)
        # fixed, even padding (4 s, mirrored by filter_pad) instead of npad='auto': MNE's automatic FFT-length
        # padding shifts some segment lengths by one input sample; with a fixed pad an exact-length segment
        # (exact_length) resamples onto (a + k * sf / sfreq) exactly (methodology log entry 45)
        seg.resample(pp["sfreq"], npad=resample_npad(sf), verbose="error", **filter_pad(cfg))
        if pp.get("bandpass", True):
            seg.filter(pp["l_freq"], pp["h_freq"], fir_design="firwin", verbose="error", **filter_pad(cfg))
        x = seg.get_data()
        o = offset_new + int(round(a * pp["sfreq"] / sf))
        segments.append({"proc_start": pos, "proc_stop": pos + x.shape[1], "orig_start": o, "orig_stop": o + x.shape[1]})
        parts.append(x)
        pos += x.shape[1]
    info = raw.info.copy()
    with info._unlock():
        info["sfreq"] = float(pp["sfreq"])
        info["lowpass"] = min(info["lowpass"], float(pp["sfreq"]) / 2)
    out = mne.io.RawArray(np.concatenate(parts, axis=1), info, verbose="error")
    return out, segments


def _slice(raw, start: int, stop: int):
    """Samples [start, stop) as a new Raw with the same info (index-based: MNE's time-based crop can land one
    sample early through floating-point truncation)."""
    import mne

    return mne.io.RawArray(raw.get_data(start=start, stop=stop), raw.info.copy(), verbose="error")


def per_segment(raw, segments: list[dict], fn) -> np.ndarray:
    """Apply ``fn(raw_segment) -> raw_segment`` (e.g. a QC high-pass) to every segment separately; concatenated data."""
    return np.concatenate([fn(_slice(raw, s["proc_start"], s["proc_stop"])).get_data() for s in segments], axis=1)


def window_starts(segments: list[dict], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(window starts in the concatenated signal, segment index of each): the v4 grid inside every segment."""
    from .boundary import valid_window_starts

    starts, idx = [], []
    for j, s in enumerate(segments):
        local = valid_window_starts(s["proc_stop"] - s["proc_start"], cfg)
        starts.append(s["proc_start"] + local)
        idx.append(np.full(len(local), j, dtype=int))
    return (np.concatenate(starts) if starts else np.empty(0, int),
            np.concatenate(idx) if idx else np.empty(0, int))


def to_original(starts: np.ndarray, seg_idx: np.ndarray, segments: list[dict]) -> np.ndarray:
    """Concatenated-signal sample -> sample at the new rate relative to the original recording."""
    if not len(starts):
        return np.empty(0, int)
    ps = np.array([segments[j]["proc_start"] for j in seg_idx])
    os_ = np.array([segments[j]["orig_start"] for j in seg_idx])
    return starts - ps + os_


def cuts_original(cuts: np.ndarray, sf_in: float, sf_new: float) -> list[float]:
    """Cut positions (uncropped original-rate samples) at the new rate, relative to the original recording."""
    return [float(c) * sf_new / sf_in for c in cuts]


def windows_crossing(starts, stops, cuts_new, margin: int, tol: float = 1.0) -> np.ndarray:
    """Mask of windows [start, stop) that contain a cut or have one closer than ``margin`` samples to either end.

    ``tol`` (samples) absorbs the rounding of stored positions to the integer grid of the new rate (a segment
    starts at a fractional position when the original rate is not a multiple of the new one)."""
    starts, stops = np.asarray(starts, float), np.asarray(stops, float)
    if not len(starts) or not len(cuts_new):
        return np.zeros(len(starts), dtype=bool)
    c = np.asarray(cuts_new, float)[None, :]
    return ((c > starts[:, None] - margin + tol) & (c < stops[:, None] + margin - tol)).any(axis=1)
