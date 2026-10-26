"""Label-blind signal audits of the D2 harmonization choices (decisions D3, D4, D5; design section 4).

* ``filter_response``: the magnitude of the D2 band-pass as applied (zero phase = squared Butterworth magnitude) at
  the band edges and inside every analysis band, and its impulse-response length (the post-filter crop must cover it);
* ``line_noise``: 50 Hz contamination of one recording's analysed data (the 250 Hz, average-referenced segments that
  enter the store): the 49-51 Hz power relative to 35-45 Hz, and the fraction of broadband variance at 47-53 Hz;
* ``notch_sensitivity``: what a 50 Hz notch WOULD change, without applying it: the relative change of the broadband
  data covariance (the LCMV input) and the fraction of candidate windows whose peak-to-peak acceptance flips. The
  store features themselves (Welch band powers to 45 Hz, band filters ending at 45 Hz with 1 Hz transitions) exclude
  50 Hz by construction, so line noise reaches them only through the LCMV weights and the window rejection.
Nothing here uses a label or changes a store; the decision on a notch is the author's, before any classification.
"""

from __future__ import annotations

import numpy as np


def filter_response(cfg: dict, sf: float = 1024.0) -> dict:
    """Zero-phase magnitude (dB) of the configured D2 band-pass and its impulse-response length."""
    import mne

    bp = cfg["preprocessing"]["bandpass"]
    sos = mne.filter.construct_iir_filter({"order": int(bp["order"]), "ftype": bp["ftype"], "output": "sos"},
                                          [bp["l_freq"], bp["h_freq"]], None, sf, "bandpass", return_copy=False,
                                          verbose="error")["sos"]
    from scipy.signal import sosfreqz

    freqs = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 13.0, 20.0, 25.0, 30.0, 40.0, 45.0, 50.0, 60.0])
    _, h = sosfreqz(sos, worN=freqs, fs=sf)
    db = 20 * np.log10(np.abs(h) ** 2 + 1e-300)               # forward-backward: squared magnitude
    x = np.zeros(int(sf * 120))
    c = len(x) // 2
    x[c] = 1.0
    y = mne.filter.filter_data(x[None], sf, bp["l_freq"], bp["h_freq"], method="iir",
                               iir_params={"order": int(bp["order"]), "ftype": "butter", "output": "sos"},
                               phase="zero", verbose="error")[0]
    a = np.abs(y)
    idx = np.flatnonzero(a > 1e-4 * a.max())
    settle = float(max(idx.max() - c, c - idx.min()) / sf)
    bands = {name: [float(lo), float(hi)] for name, (lo, hi) in cfg["bands"].items()}
    _, hb = sosfreqz(sos, worN=np.linspace(0.5, 45, 400), fs=sf)
    return {"filter": bp, "sfreq": sf, "magnitude_db": dict(zip(freqs.tolist(), db.round(3).tolist())),
            "impulse_length_s_1e-4": settle, "settle_crop_s": float(cfg["preprocessing"]["settle_s"]),
            "settle_covers_impulse": settle <= float(cfg["preprocessing"]["settle_s"]),
            "min_gain_db_0.5_45": float(20 * np.log10(np.abs(hb).min() ** 2)), "bands": bands}


def _psd(x: np.ndarray, sf: float):
    from scipy.signal import welch

    f, p = welch(x, fs=sf, nperseg=int(2 * sf), noverlap=int(sf), axis=-1)
    return f, p


def line_noise(x: np.ndarray, sf: float, f0: float = 50.0) -> dict:
    """(channels, samples) volts -> 50 Hz contamination (median over channels)."""
    f, p = _psd(x, sf)
    if f0 + 1 >= sf / 2:
        return {"line_ratio_db": float("nan"), "line_variance_fraction": float("nan")}
    line = p[:, (f >= f0 - 1) & (f <= f0 + 1)].mean(axis=1)
    ref = p[:, (f >= 35) & (f <= 45)].mean(axis=1)
    band = p[:, (f >= f0 - 3) & (f <= f0 + 3)].sum(axis=1)
    total = p[:, (f >= 0.5)].sum(axis=1)
    return {"line_ratio_db": float(np.median(10 * np.log10(line / np.maximum(ref, 1e-40)))),
            "line_variance_fraction": float(np.median(band / np.maximum(total, 1e-40)))}


def notch(x: np.ndarray, sf: float, f0: float = 50.0) -> np.ndarray:
    import mne

    return mne.filter.notch_filter(x, sf, [f0], method="iir", phase="zero", verbose="error")


def notch_sensitivity(segs, pieces, cfg: dict) -> dict:
    """Effect a 50 Hz notch would have on the LCMV covariance input and on the window rejection (diagnostic only)."""
    from ..sourcegraphs.build import reject
    from ..sourcegraphs.recording import Segment

    if not pieces:
        return {"covariance_rel_change": float("nan"), "rejection_flip_fraction": float("nan")}
    sf = segs[0].sfreq
    nsegs = [Segment(s.index, s.orig_start, notch(s.data, sf), s.sfreq) for s in segs]
    sel = np.concatenate([segs[p.segment].data[:, p.start:p.stop] for p in pieces], axis=1)
    nsel = np.concatenate([nsegs[p.segment].data[:, p.start:p.stop] for p in pieces], axis=1)
    c0, c1 = np.cov(sel), np.cov(nsel)
    _, k0, _, _ = reject(segs, pieces, cfg)
    _, k1, _, _ = reject(nsegs, pieces, cfg)
    return {"covariance_rel_change": float(np.linalg.norm(c0 - c1) / np.linalg.norm(c0)),
            "rejection_flip_fraction": float(np.mean(k0 != k1)) if len(k0) else float("nan"),
            "accepted_without_notch": int(k0.sum()), "accepted_with_notch": int(k1.sum())}


def recording_audit(segs, pieces, cfg: dict) -> dict:
    """Line noise + notch sensitivity of one recording's analysed data (the selected pieces)."""
    if not pieces:
        return {}
    sel = np.concatenate([segs[p.segment].data[:, p.start:p.stop] for p in pieces], axis=1)
    return {**line_noise(sel, segs[0].sfreq), **notch_sensitivity(segs, pieces, cfg)}
