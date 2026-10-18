"""Per-window features of the LCMV / Xu suite (stateless; computed before any split, no leakage).

For one selected piece of a continuous segment, ``x`` holds the signal from ``margin`` samples before the first
window to ``margin`` samples after the last one (all inside the same segment). Band analytic signals are computed
on that whole stretch with the finite analytic FIR (``connectivity.plv.band_analytic_fir``), so every sample of
every window is exact (the kernel half-length is at most ``margin``).

Per window and band:
* PSD: Welch (2 s Hann, 50 %) of the window's broadband signal; band power abs / log10 / relative to 0.5-45 Hz;
* DE (Xu et al.): 0.5 * ln(2 * pi * e * var(x_band)), x_band = real part of the band analytic signal;
* PLV and |ciPLV| (ROIs only): ``connectivity.plv.phase_sync`` of the band analytic ROI signals.
"""

from __future__ import annotations

import numpy as np

from ..connectivity.plv import band_analytic_fir, phase_sync
from ..features.psd import window_band_powers


def window_features(roi: np.ndarray, sensor: np.ndarray, starts: list[int], win: int, cfg: dict,
                    chunk_rows: int = 17) -> dict[str, np.ndarray]:
    """``roi`` (68, T), ``sensor`` (19, T): one piece with margins; ``starts``: window starts in this array."""
    sf = float(cfg["preprocessing"]["sfreq"])
    bands = cfg["bands"]
    w = cfg["filters"]["transition_bandwidth_hz"]
    n, k, r, c = len(starts), len(bands), roi.shape[0], sensor.shape[0]
    out = {name: np.zeros(shape, np.float32) for name, shape in (
        ("plv", (n, k, r, r)), ("ciplv", (n, k, r, r)), ("psd_de", (n, k, r)), ("sensor_psd_de", (n, k, c)))}
    if cfg["features"].get("save_signed_ciplv"):
        out["ciplv_signed"] = np.zeros((n, k, r, r), np.float32)
    for b, (lo, hi) in enumerate(bands.values()):
        a_roi = band_analytic_fir(roi, sf, float(lo), float(hi), chunk_rows, w)
        a_sen = band_analytic_fir(sensor, sf, float(lo), float(hi), None, w)
        for i, s in enumerate(starts):
            seg = a_roi[:, s:s + win]
            plv, ciplv, signed = phase_sync(seg)
            out["plv"][i, b], out["ciplv"][i, b] = plv, ciplv
            if "ciplv_signed" in out:
                out["ciplv_signed"][i, b] = signed
            out["psd_de"][i, b] = 0.5 * np.log(2 * np.pi * np.e * np.var(seg.real, axis=1))
            out["sensor_psd_de"][i, b] = 0.5 * np.log(2 * np.pi * np.e * np.var(a_sen[:, s:s + win].real, axis=1))
        del a_roi, a_sen
    pcfg = cfg["psd"]
    for prefix, x in (("psd", roi), ("sensor_psd", sensor)):
        bp = [window_band_powers(x[:, s:s + win], sf, bands, pcfg) for s in starts]
        for kind in ("abs", "log", "rel"):
            out[f"{prefix}_{kind}"] = (np.stack([p[kind] for p in bp]).astype(np.float32) if bp
                                       else np.zeros((0, k, x.shape[0]), np.float32))
    return out
