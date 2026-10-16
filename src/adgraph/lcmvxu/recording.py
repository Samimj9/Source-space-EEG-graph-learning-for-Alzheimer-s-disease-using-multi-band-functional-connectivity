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
     selected; the last piece is shortened to what is still needed. Pieces shorter than one window are skipped,
     and no piece is added once less than one window is still needed (it could only hold a window by exceeding
     300 s; methodology log entry 53). The selection therefore never exceeds 300 s; fewer than 300 s are selected
     when the recording does not hold them as whole windows; the shortfall is reported, the participant is not
     excluded for it.
4. **Windows.** 10 s windows, 1 s step, inside each piece only (a piece of length L gives floor((L - 10) / 1) + 1).
5. **Exact positions.** Segment ``j`` starts at input sample ``start_in`` (exact integer); its resampled sample
   ``k`` lies at input position ``start_in + k * sf_in / sf`` exactly. ``segment_table`` records these integers
   (and the integer cut samples), so boundary validation never goes through rounded seconds.
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
class SegmentShape:
    """Length-only stand-in for ``Segment`` (selection from the boundary geometry alone, without the signal)."""
    index: int
    orig_start: int
    n: int
    sfreq: float


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
    table, info = segment_table(raw.n_times, cuts, n_events, sf_in, sf)
    out = []
    for t in table:
        if not t["kept"]:
            continue
        seg = disc._slice(raw, t["start_in"], t["start_in"] + t["n_in"])
        seg.resample(sf, npad=disc.resample_npad(sf_in), pad="reflect_limited", verbose="error")
        x = seg.get_data()
        if x.shape[1] != t["n"]:
            raise RuntimeError(f"segment {t['index']}: resampled to {x.shape[1]} samples, expected {t['n']}")
        x = x - x.mean(axis=0, keepdims=True)                   # common average (instantaneous)
        out.append(Segment(t["index"], t["orig_start"], x, sf))
    return out, info


def segment_table(n_times: int, cuts, n_events: int, sf_in: float, sf: float) -> tuple[list[dict], dict]:
    """Segment geometry from the boundary cuts alone (no signal): per segment its index, exact input-rate start
    ``start_in`` and trimmed length ``n_in``, resampled length ``n`` and ``orig_start`` (= round(start_in * sf /
    sf_in), the 250 Hz position used for ``start_sample``); plus the recording-level info of ``segments``."""
    parts = disc.split(n_times, cuts)
    table, removed = [], 0
    for j, (a, b) in enumerate(parts):
        n_in = disc.exact_length(b - a, sf_in, sf)
        removed += (b - a) - n_in
        n = disc.resampled_length(n_in, sf_in, sf)
        table.append({"index": j, "start_in": int(a), "stop_in": int(b), "n_in": int(n_in), "n": int(n),
                      "orig_start": int(round(a * sf / sf_in)), "kept": n >= 2})
    info = {"n_boundary_events": int(n_events), "n_discontinuities": int(len(cuts)),
            "boundary_positions_s": [round(float(c) / sf_in, 3) for c in cuts],
            "boundary_cuts_in": [int(c) for c in cuts],
            "segments_in": [[t["index"], t["start_in"], t["n_in"], t["orig_start"]] for t in table if t["kept"]],
            "segment_durations_s": [round((b - a) / sf_in, 3) for a, b in parts],
            "samples_trimmed_for_exact_resampling": int(removed),
            "original_duration_s": round(n_times / sf_in, 3)}
    return table, info


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
            if need - got < win:              # no further window fits within the 300 s target (entry 53)
                break
            length = min(b - a, need - got)   # >= win: every interior holds one window
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
