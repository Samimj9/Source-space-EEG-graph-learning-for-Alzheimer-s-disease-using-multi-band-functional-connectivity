"""Boundary-safe recording handling for the D3 LCMV / Xu-protocol suite (methodology log entry 50).

1. **Segments.** Every EEGLAB ``boundary`` event cuts the recording (``preprocessing.discontinuities``, entry 45:
   MNE onset = (latency - 1) / srate, cut before sample ceil(onset * srate)). Each continuous segment is trimmed
   to an even length and resampled to 250 Hz ON ITS OWN (fixed 4 s padding); segments are never concatenated.
2. **Margin.** ``margin`` = the largest half-length of the band analytic kernels (delta: 0.5 Hz lower transition,
   825 samples = 3.3 s at 250 Hz). A window lies at least ``margin`` from both ends of its own segment, so every
   filtered value inside it is computed only from samples of that segment (no filtering across a boundary or a
   recording end). This is the only trimming applied (no fixed edge crop).
3. **Five-minute selection (deterministic).** The *interior* of a segment is [start + margin, stop - margin).
   * If an interior holds 300 s, the first 300 s of the FIRST such segment (in time) are selected: one
     uninterrupted interval, 291 windows of 10 s with a 1 s step.
   * Otherwise the interiors are taken in time order, each as one separate piece (never joined), until 300 s are
     selected; the last piece is shortened to what is still needed (but never below one window). Pieces shorter
     than one window are skipped. Fewer than 300 s are selected only when the recording does not hold them; the
     shortfall is reported, the participant is not excluded for it.
4. **Windows.** 10 s windows, 1 s step, inside each piece only (a piece of length L gives floor((L - 10) / 1) + 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..preprocessing import discontinuities as disc


@dataclass
class Segment:
    index: int
    orig_start: int                 # first sample at the NEW rate, relative to the recording start
    data: np.ndarray                # (n_channels, n_times) volts, resampled, average-referenced
    sfreq: float

    @property
    def n(self) -> int:
        return self.data.shape[1]


@dataclass
class Piece:
    segment: int
    start: int                      # segment-local sample (new rate): first sample a window may start at
    stop: int                       # segment-local: last window ends at or before this sample
    windows: list = field(default_factory=list)   # segment-local window starts

    @property
    def length(self) -> int:
        return self.stop - self.start


def margin_samples(cfg: dict) -> int:
    from ..connectivity.plv import analytic_kernel

    sf, w = float(cfg["preprocessing"]["sfreq"]), cfg["filters"]["transition_bandwidth_hz"]
    return max(len(analytic_kernel(sf, float(lo), float(hi), w)) // 2 for lo, hi in cfg["bands"].values())


def window_len(cfg: dict) -> tuple[int, int]:
    sf = float(cfg["preprocessing"]["sfreq"])
    return int(round(cfg["windowing"]["window_s"] * sf)), int(round(cfg["windowing"]["step_s"] * sf))


def segments(raw, cfg: dict) -> tuple[list[Segment], dict]:
    """Continuous segments of a loaded recording, each resampled separately and average-referenced."""
    import mne

    sf_in, sf = raw.info["sfreq"], float(cfg["preprocessing"]["sfreq"])
    cuts, n_events = disc.boundary_cuts(raw)
    parts = disc.split(raw.n_times, cuts)
    out, removed = [], 0
    for j, (a, b) in enumerate(parts):
        n = disc.exact_length(b - a, sf_in, sf)
        removed += (b - a) - n
        if disc.resampled_length(n, sf_in, sf) < 2:
            continue
        seg = disc._slice(raw, a, a + n)
        seg.resample(sf, npad=disc.resample_npad(sf_in), pad="reflect_limited", verbose="error")
        x = seg.get_data()
        x = x - x.mean(axis=0, keepdims=True)                   # common average (instantaneous)
        out.append(Segment(j, int(round(a * sf / sf_in)), x, sf))
    info = {"n_boundary_events": int(n_events), "n_discontinuities": int(len(cuts)),
            "boundary_positions_s": [round(float(c) / sf_in, 3) for c in cuts],
            "segment_durations_s": [round((b - a) / sf_in, 3) for a, b in parts],
            "samples_trimmed_for_exact_resampling": int(removed),
            "original_duration_s": round(raw.n_times / sf_in, 3)}
    return out, info


def select(segs: list[Segment], cfg: dict, margin: int) -> tuple[list[Piece], dict]:
    """Deterministic five-minute selection (module doc, step 3) and the window grid of every piece."""
    win, step = window_len(cfg)
    need = int(round(cfg["windowing"]["selection_s"] * segs[0].sfreq)) if segs else 0
    interiors = [(i, margin, s.n - margin) for i, s in enumerate(segs) if s.n - 2 * margin >= win]
    pieces, mode = [], "none"
    long_ = [t for t in interiors if t[2] - t[1] >= need]
    if long_:
        i, a, _ = long_[0]
        pieces, mode = [Piece(i, a, a + need)], "contiguous"
    else:
        got = 0
        for i, a, b in interiors:
            if got >= need:
                break
            length = min(b - a, max(need - got, win))
            pieces.append(Piece(i, a, a + length))
            got += length
        mode = "multi_segment" if len(pieces) > 1 else ("single_short_segment" if pieces else "none")
    for p in pieces:
        p.windows = list(range(p.start, p.stop - win + 1, step))
    sf = segs[0].sfreq if segs else 1.0
    info = {"selection_mode": mode, "n_pieces": len(pieces),
            "selected_duration_s": round(sum(p.length for p in pieces) / sf, 3),
            "selection_target_s": cfg["windowing"]["selection_s"],
            "usable_interior_s": round(sum(b - a for _, a, b in interiors) / sf, 3),
            "n_usable_segments": len(interiors), "n_segments": len(segs), "margin_samples": int(margin),
            "n_candidate_windows": int(sum(len(p.windows) for p in pieces))}
    return pieces, info


def crossing_check(segs: list[Segment], pieces: list[Piece], cfg: dict, margin: int) -> None:
    """Raise if any window is closer than ``margin`` to its segment ends (i.e. could see a discontinuity)."""
    win, _ = window_len(cfg)
    for p in pieces:
        n = segs[p.segment].n
        for s in p.windows:
            if s < margin or s + win > n - margin:
                raise RuntimeError(f"window at {s} of segment {segs[p.segment].index} violates the boundary margin")
