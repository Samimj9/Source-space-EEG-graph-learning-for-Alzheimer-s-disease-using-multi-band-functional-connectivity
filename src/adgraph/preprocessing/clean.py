"""Continuous-signal cleaning: filtering, bad channels, interpolation, reference."""

from __future__ import annotations

import numpy as np

from ..io.eeg import robust_channel_std


def detect_bad_channels(data: np.ndarray, ch_names: list[str], rules: dict) -> dict[str, str]:
    """Flag flat, outlying-amplitude and uncorrelated channels. Returns ``{channel: reason}``."""
    std = np.std(data, axis=1)
    bads: dict[str, str] = {}

    for i in np.flatnonzero(std < rules["flat_std_v"]):
        bads[ch_names[i]] = "flat"

    ok = std >= rules["flat_std_v"]
    log_std = np.log(std[ok])
    mad = np.median(np.abs(log_std - np.median(log_std))) * 1.4826
    if mad > 0:
        z = (log_std - np.median(log_std)) / mad
        for i, zi in zip(np.flatnonzero(ok), z):
            if abs(zi) > rules["robust_z_log_std"]:
                bads.setdefault(ch_names[i], f"amplitude z={zi:.1f}")

    if ok.sum() > 2:
        idx = np.flatnonzero(ok)
        corr = np.abs(np.corrcoef(data[idx]))
        np.fill_diagonal(corr, np.nan)
        neighbor = np.nanpercentile(corr, 90, axis=1)
        for i, c in zip(idx, neighbor):
            if c < rules["min_neighbor_corr"]:
                bads.setdefault(ch_names[i], f"low correlation {c:.2f}")
    return bads


def preprocess(raw, cfg: dict) -> tuple[object, dict]:
    """Resample, band-pass, check units, interpolate bad channels, average reference.

    The average reference is added as a projection and applied, so the data and
    ``raw.info`` both carry it (required by the inverse operator).
    """
    pp = cfg["preprocessing"]
    raw.resample(pp["sfreq"], verbose="error")
    raw.filter(pp["l_freq"], pp["h_freq"], fir_design="firwin", verbose="error")

    med_std = robust_channel_std(raw.get_data())
    lo, hi = cfg["units"]["plausible_std_v"]
    if not lo <= med_std <= hi:
        raise ValueError(f"median channel std {med_std:.3g} V outside plausible range [{lo}, {hi}]: check unit scale")

    detect_on = raw.copy().filter(1.0, None, fir_design="firwin", verbose="error").get_data()
    bads = detect_bad_channels(detect_on, raw.ch_names, pp["bad_channels"])
    frac = len(bads) / len(raw.ch_names)
    raw.info["bads"] = list(bads)
    if bads and pp["interpolate_bads"]:
        raw.interpolate_bads(reset_bads=True, verbose="error")

    raw.set_eeg_reference(pp["reference"], projection=True, verbose="error")
    raw.apply_proj(verbose="error")

    qc = {
        "median_std_uv": med_std * 1e6,
        "bad_channels": bads,
        "n_bad_channels": len(bads),
        "bad_fraction_warning": frac > pp["bad_channels"]["max_fraction_warn"],
    }
    return raw, qc
