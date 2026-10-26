"""Label-blind signal audits of the D2 harmonization choices (decisions D3, D4, D5; design section 4).

* ``filter_response``: the magnitude of the D2 band-pass as applied (zero phase = squared Butterworth magnitude) at
  the band edges and inside every analysis band, and its impulse-response length (the post-filter crop must cover it);
* ``line_noise``: 50 Hz contamination of one recording's analysed data (the 250 Hz, average-referenced segments that
  enter the store): the 49-51 Hz power relative to 35-45 Hz, and the fraction of broadband variance at 47-53 Hz;
* ``notch_sensitivity``: the DIAGNOSTIC notch of the first audit (an IIR notch on the already resampled 250 Hz
  segments): what such a notch would change in the broadband covariance (the LCMV input) and the window acceptance.
  It is not the preprocessing notch and is kept only to compare with the first audit;
* ``notch_response`` / ``combined_support``: the ACTUAL preprocessing notch of the D2 stores (``d2store.
  remove_line_noise``: zero-phase FIR at the input rate, after the band-pass, before the settle crop and resampling),
  its magnitude in every analysis band and its edge support together with the band-pass;
* ``compare``: one recording processed without and with the actual notch (same crops, same pieces): 50 Hz power,
  harmonics and other narrowband peaks, covariance change, accepted windows, and the power change in every band.
The store features (Welch band powers to 45 Hz; band filters ending at 45 Hz with 1 Hz transitions) exclude 50 Hz by
construction, so line noise reaches them through the LCMV weights and the window rejection. Nothing here uses a label.
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


def impulse_support(y: np.ndarray, c: int, sf: float, rel: float = 1e-4) -> float:
    a = np.abs(y)
    idx = np.flatnonzero(a > rel * a.max())
    return float(max(idx.max() - c, c - idx.min()) / sf)


def notch_response(cfg: dict, sf: float = 1024.0) -> dict:
    """Magnitude (dB) of the configured preprocessing notch alone and of band-pass + notch, and the edge support of
    the cascade (must be covered by ``settle_s``)."""
    from .d2store import remove_line_noise
    import mne

    ln = cfg["preprocessing"].get("line_noise")
    bp = cfg["preprocessing"]["bandpass"]
    x = np.zeros(int(sf * 120))
    c = len(x) // 2
    x[c] = 1.0
    notch_only = remove_line_noise(x[None], sf, ln)[0]
    band = mne.filter.filter_data(x[None], sf, bp["l_freq"], bp["h_freq"], method="iir",
                                  iir_params={"order": int(bp["order"]), "ftype": "butter", "output": "sos"},
                                  phase="zero", verbose="error")
    cascade = remove_line_noise(band, sf, ln)[0]
    f = np.fft.rfftfreq(len(x), 1 / sf)

    def db(y, q):
        return float(20 * np.log10(abs(np.fft.rfft(y)[np.argmin(abs(f - q))]) + 1e-300))
    freqs = [0.5, 1.0, 4.0, 8.0, 13.0, 25.0, 40.0, 44.0, 45.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 100.0]
    inband = (f >= 0.5) & (f <= 45.0)
    dev = np.abs(20 * np.log10(np.abs(np.fft.rfft(notch_only))[inband] + 1e-300))
    support_notch = impulse_support(notch_only - (np.arange(len(x)) == c), c, sf) if ln else 0.0
    out = {"line_noise": ln, "notch_db": {q: round(db(notch_only, q), 3) for q in freqs},
           "cascade_db": {q: round(db(cascade, q), 3) for q in freqs},
           "notch_max_abs_db_0.5_45": float(dev.max()), "notch_support_s": support_notch,
           "bandpass_support_s": impulse_support(band[0], c, sf),
           "settle_crop_s": float(cfg["preprocessing"]["settle_s"])}
    out["combined_support_s"] = out["bandpass_support_s"] + support_notch
    out["settle_covers_cascade"] = out["combined_support_s"] <= out["settle_crop_s"]
    return out


def narrowband_peaks(f: np.ndarray, p: np.ndarray, lo: float = 1.0, hi: float | None = None,
                     min_db: float = 10.0) -> list[tuple[float, float]]:
    """(frequency, prominence dB) of PSD bins exceeding the median of their +-5 Hz neighbourhood by ``min_db``
    (median over channels first); local maxima only."""
    m = np.median(p, axis=0) if p.ndim > 1 else p
    hi = hi or f[-1] - 5
    out = []
    for i in np.flatnonzero((f >= lo) & (f <= hi)):
        nb = (np.abs(f - f[i]) <= 5) & (np.abs(f - f[i]) > 1)
        prom = 10 * np.log10(m[i] / max(np.median(m[nb]), 1e-40))
        if prom >= min_db and m[i] == m[max(i - 1, 0):i + 2].max():
            out.append((round(float(f[i]), 2), round(float(prom), 1)))
    return out


def band_powers(x: np.ndarray, sf: float, bands: dict) -> dict:
    f, p = _psd(x, sf)
    return {b: p[:, (f >= lo) & (f < hi)].sum(axis=1) for b, (lo, hi) in bands.items()}


def compare(before: np.ndarray, after: np.ndarray, sf: float, cfg: dict) -> dict:
    """One recording without / with the preprocessing notch (same samples): line noise, harmonics, residual
    narrowband peaks, covariance change and the power change per analysis band (median over channels, dB)."""
    out = {}
    for tag, x in (("before", before), ("after", after)):
        ln = line_noise(x, sf)
        out.update({f"{k}_{tag}": v for k, v in ln.items()})
        f, p = _psd(x, sf)
        if 101 < sf / 2:
            h = p[:, (f >= 99) & (f <= 101)].mean(axis=1) / np.maximum(p[:, (f >= 35) & (f <= 45)].mean(axis=1), 1e-40)
            out[f"harmonic100_ratio_db_{tag}"] = float(np.median(10 * np.log10(h)))
        out[f"narrowband_peaks_{tag}"] = ";".join(f"{a}Hz:{b}dB" for a, b in narrowband_peaks(f, p))
    c0, c1 = np.cov(before), np.cov(after)
    out["covariance_rel_change_actual_notch"] = float(np.linalg.norm(c0 - c1) / np.linalg.norm(c0))
    pb, pa = band_powers(before, sf, cfg["bands"]), band_powers(after, sf, cfg["bands"])
    for b in cfg["bands"]:
        out[f"band_power_change_db_{b}"] = float(np.median(10 * np.log10(pa[b] / np.maximum(pb[b], 1e-40))))
    return out


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
    """DIAGNOSTIC (first audit): effect of an IIR 50 Hz notch on the already resampled segments on the LCMV covariance
    input and on the window rejection. Not the preprocessing notch (see ``compare``)."""
    from ..sourcegraphs.build import reject
    from ..sourcegraphs.recording import Segment

    if not pieces:
        return {"diag_covariance_rel_change": float("nan"), "diag_rejection_flip_fraction": float("nan")}
    sf = segs[0].sfreq
    nsegs = [Segment(s.index, s.orig_start, notch(s.data, sf), s.sfreq) for s in segs]
    sel = np.concatenate([segs[p.segment].data[:, p.start:p.stop] for p in pieces], axis=1)
    nsel = np.concatenate([nsegs[p.segment].data[:, p.start:p.stop] for p in pieces], axis=1)
    c0, c1 = np.cov(sel), np.cov(nsel)
    _, k0, _, _ = reject(segs, pieces, cfg)
    _, k1, _, _ = reject(nsegs, pieces, cfg)
    return {"diag_covariance_rel_change": float(np.linalg.norm(c0 - c1) / np.linalg.norm(c0)),
            "diag_rejection_flip_fraction": float(np.mean(k0 != k1)) if len(k0) else float("nan"),
            "diag_accepted_without_notch": int(k0.sum()), "diag_accepted_with_notch": int(k1.sum())}


def recording_audit(segs, pieces, cfg: dict) -> dict:
    """Line noise + notch sensitivity of one recording's analysed data (the selected pieces)."""
    if not pieces:
        return {}
    sel = np.concatenate([segs[p.segment].data[:, p.start:p.stop] for p in pieces], axis=1)
    return {**line_noise(sel, segs[0].sfreq), **notch_sensitivity(segs, pieces, cfg)}
