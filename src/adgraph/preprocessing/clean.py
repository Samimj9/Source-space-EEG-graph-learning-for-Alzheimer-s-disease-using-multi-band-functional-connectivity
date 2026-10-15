"""Continuous-signal cleaning: filtering, bad channels, interpolation, reference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.eeg import robust_channel_std


class FlatChannelError(ValueError):
    """A confirmed flat channel in a recording whose bad channels are not interpolated (Dataset 3): it must
    never enter the average reference, the source model or connectivity as a valid measurement."""


def channel_metrics(data: np.ndarray, rules: dict) -> pd.DataFrame:
    """Per-channel quantities behind the bad-channel rules, and the rule each channel fails.

    ``std`` (V); ``robust_std`` = 1.4826 MAD (V); ``log_std_z`` = robust z of log(std) among the
    non-flat channels; ``neighbor_corr_p90`` = 90th percentile |correlation| with the other
    non-flat channels; ``bad_reason`` = first failed rule (flat, amplitude, low correlation) or ''."""
    std = np.std(data, axis=1)
    flat = std < rules["flat_std_v"]
    ok = ~flat
    z = np.full(len(std), np.nan)
    neighbor = np.full(len(std), np.nan)
    log_std = np.log(std[ok])
    mad = np.median(np.abs(log_std - np.median(log_std))) * 1.4826 if ok.any() else 0.0
    if mad > 0:
        z[ok] = (log_std - np.median(log_std)) / mad
    if ok.sum() > 2:
        corr = np.abs(np.corrcoef(data[ok]))
        np.fill_diagonal(corr, np.nan)
        neighbor[ok] = np.nanpercentile(corr, 90, axis=1)
    reason = np.where(flat, "flat", "").astype(object)
    for i in np.flatnonzero(ok):
        if abs(z[i]) > rules["robust_z_log_std"]:
            reason[i] = f"amplitude z={z[i]:.1f}"
        elif neighbor[i] < rules["min_neighbor_corr"]:
            reason[i] = f"low correlation {neighbor[i]:.2f}"
    med = np.median(data, axis=1, keepdims=True)
    return pd.DataFrame({"std": std, "robust_std": 1.4826 * np.median(np.abs(data - med), axis=1),
                         "log_std_z": z, "neighbor_corr_p90": neighbor, "bad_reason": reason})


def detect_bad_channels(data: np.ndarray, ch_names: list[str], rules: dict) -> dict[str, str]:
    """Flag flat, outlying-amplitude and uncorrelated channels. Returns ``{channel: reason}``
    (flat channels first, then amplitude, then correlation, each in channel order)."""
    reasons = channel_metrics(data, rules)["bad_reason"].tolist()
    bads: dict[str, str] = {}
    for kind in ("flat", "amplitude", "low correlation"):
        bads.update({ch: r for ch, r in zip(ch_names, reasons) if r.startswith(kind)})
    return bads


def resample_and_bandpass(raw, cfg: dict):
    """First preprocessing step (in place): resample to ``sfreq``, band-pass ``l_freq``-``h_freq``.

    Pipeline v4 (``boundary`` in the config) makes the padding of both operations explicit
    (``boundary.filter_pad``, MNE's default value, so the numbers are unchanged); the recording has
    then already been cropped to the analysed interval (``preprocessing.boundary.crop_edges``)."""
    from .boundary import filter_pad

    pp = cfg["preprocessing"]
    raw.resample(pp["sfreq"], verbose="error", **filter_pad(cfg))
    if pp.get("bandpass", True):          # false: provider-filtered data (Dataset 3), never filtered twice
        raw.filter(pp["l_freq"], pp["h_freq"], fir_design="firwin", verbose="error", **filter_pad(cfg))
    return raw


def bandpass_half_length(cfg: dict) -> int:
    """Half length (samples at ``preprocessing.sfreq``) of the 0.1-45 Hz FIR used by
    ``resample_and_bandpass``: how far a transient can spread through that filter."""
    import mne

    pp = cfg["preprocessing"]
    if not pp.get("bandpass", True):
        return 0
    h = mne.filter.create_filter(None, pp["sfreq"], pp["l_freq"], pp["h_freq"], fir_design="firwin", verbose="error")
    return len(h) // 2


def analysed_interval(n_samples: int, sfreq: float, cfg: dict) -> slice:
    """Samples that can enter windows: the recording without ``windowing.edge_trim_s`` at each
    end (the same rounding as ``window_starts``)."""
    from .boundary import boundary_aware

    if boundary_aware(cfg):                       # v4: the edges were cropped before any filtering
        return slice(0, n_samples)
    trim = int(round(cfg["windowing"]["edge_trim_s"] * sfreq))
    if n_samples - 2 * trim < 1:
        raise ValueError(f"recording of {n_samples / sfreq:.1f} s is not longer than twice the edge trim "
                         f"({cfg['windowing']['edge_trim_s']} s)")
    return slice(trim, n_samples - trim)


def robust_std(data: np.ndarray) -> np.ndarray:
    """Per-channel 1.4826 x median absolute deviation (V); insensitive to brief large transients."""
    med = np.median(data, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(data - med), axis=1)


def preprocess(raw, cfg: dict) -> tuple[object, dict]:
    """Resample, band-pass, unit safeguard, bad channels, interpolation, average reference,
    amplitude plausibility (methodology log entry 39).

    1. Resample to 250 Hz and band-pass 0.1-45 Hz (the continuous signal that is analysed).
    2. **Unit safeguard** (``units.gross_robust_std_v``): median over channels of the robust std
       (1.4826 MAD) of the whole 0.1-45 Hz recording. MAD ignores brief edge transients; the
       range only separates physiological EEG from 10^3 / 10^6 unit-scale errors.
    3. **Bad channels** on the analysed interval (edge trim removed) of the 1 Hz high-passed
       signal: the rules see only data that can enter windows. Interpolated (spherical spline).
    4. Average reference (projection, applied; required by the inverse operator).
    5. **Amplitude plausibility** (``units.plausible_std_v``): median over channels of the std of
       the analysed interval of the final signal (0.1-45 Hz, interpolated, average reference),
       i.e. the signal the windows are cut from.

    Dataset 3 (OpenNeuro ds004504 derivatives, methodology log entry 44): ``preprocessing.bandpass: false``
    (the provider's 0.5-45 Hz filter, ASR and ICA cleaning are used as given; nothing is filtered twice) and
    ``interpolate_bads: false`` (the bad-channel rules are evaluated and reported, nothing is interpolated; a
    confirmed *flat* channel raises ``FlatChannelError`` before referencing instead of entering the analysis).

    Window rejection (``process_recording``) is unchanged. ``qc["channels"]`` holds the
    per-channel quantities; ``bad_channels_whole_recording`` (QC only) what step 3 would flag
    on the whole recording.

    Pipeline v4 (methodology log entry 41): the recording reaching this function has already been
    cropped to the analysed interval (``boundary.crop_edges``), so every step, including the unit
    safeguard, sees only the analysed interval, and every filter pads explicitly
    (``boundary.filter_pad``). ``bad_channels_whole_recording`` then equals ``bad_channels``.
    """
    from .boundary import filter_pad

    pp = cfg["preprocessing"]
    resample_and_bandpass(raw, cfg)
    data = raw.get_data()
    iv = analysed_interval(data.shape[1], raw.info["sfreq"], cfg)

    gross = float(np.median(robust_std(data)))
    glo, ghi = cfg["units"]["gross_robust_std_v"]
    if not glo <= gross <= ghi:
        raise ValueError(f"unit safeguard failed: robust median channel std {gross:.3g} V outside [{glo}, {ghi}] V "
                         "(whole 0.1-45 Hz recording): the EDF unit scale is wrong by orders of magnitude; "
                         "run scripts/diagnose_amplitude.py")

    hp = raw.copy().filter(1.0, None, fir_design="firwin", verbose="error", **filter_pad(cfg)).get_data()
    rules = pp["bad_channels"]
    metrics = channel_metrics(hp[:, iv], rules)
    bads = detect_bad_channels(hp[:, iv], raw.ch_names, rules)
    bads_whole = detect_bad_channels(hp, raw.ch_names, rules)          # QC only: edges included
    del hp
    raw.info["bads"] = list(bads)
    if bads and pp["interpolate_bads"]:
        raw.interpolate_bads(reset_bads=True, verbose="error")
    elif not pp["interpolate_bads"]:
        # Dataset 3 (provider-cleaned, entry 44): a confirmed flat channel (std over the whole analysed interval
        # of the 1 Hz high-passed signal below rules.flat_std_v = 0.1 uV, two orders of magnitude below
        # physiological EEG) is not a measurement: stop before referencing; the author decides how to handle it.
        flat = [ch for ch, why in bads.items() if why == "flat"]
        if flat:
            stds = metrics.set_index(pd.Index(raw.ch_names))["std"]
            raise FlatChannelError(
                f"flat channel(s) {flat} (std {', '.join(f'{stds[c] * 1e6:.3g} uV' for c in flat)} over the analysed "
                f"interval, < {rules['flat_std_v'] * 1e6:.3g} uV): not interpolated by configuration, so this "
                "recording is not processed; decide how to handle it (methodology log entry 44)")
        # other flagged channels (amplitude, correlation) are reported only and stay in the signal
        raw.info["bads"] = []

    raw.set_eeg_reference(pp["reference"], projection=True, verbose="error")
    raw.apply_proj(verbose="error")

    final = raw.get_data()[:, iv]
    final_std, final_robust = final.std(axis=1), robust_std(final)
    med_std = float(np.median(final_std))
    lo, hi = cfg["units"]["plausible_std_v"]
    if not lo <= med_std <= hi:
        raise ValueError(f"median channel std {med_std:.3g} V outside plausible range [{lo}, {hi}] (analysed "
                         "interval, 0.1-45 Hz, average reference): run scripts/diagnose_amplitude.py")

    channels = pd.DataFrame({
        "channel": raw.ch_names,
        "std_uv": np.std(data[:, iv], axis=1) * 1e6,          # analysed interval, 0.1-45 Hz, recording reference
        "robust_std_uv": robust_std(data[:, iv]) * 1e6,
        "whole_recording_std_uv": np.std(data, axis=1) * 1e6,
        "hp1_std_uv": metrics["std"].to_numpy() * 1e6,         # what the bad-channel rules see
        "log_std_z": metrics["log_std_z"].to_numpy(),
        "neighbor_corr_p90": metrics["neighbor_corr_p90"].to_numpy(),
        "bad_reason": [bads.get(c, "") for c in raw.ch_names],
        "bad_reason_whole_recording": [bads_whole.get(c, "") for c in raw.ch_names],
        "interpolated": [c in bads and bool(pp["interpolate_bads"]) for c in raw.ch_names],
        "final_std_uv": final_std * 1e6,                       # after interpolation + average reference
        "final_robust_std_uv": final_robust * 1e6,
    })
    qc = {
        "median_std_uv": med_std * 1e6,                        # the plausibility statistic (step 5)
        "median_robust_std_uv": float(np.median(final_robust)) * 1e6,
        "gross_robust_std_uv": gross * 1e6,                    # the unit safeguard statistic (step 2)
        "whole_recording_std_uv": robust_channel_std(data) * 1e6,   # former check statistic (QC only)
        "analysed_interval_s": [iv.start / raw.info["sfreq"], iv.stop / raw.info["sfreq"]],
        "bad_channels": bads,
        "n_bad_channels": len(bads),
        "bad_fraction_warning": len(bads) / len(raw.ch_names) > rules["max_fraction_warn"],
        "bad_channels_whole_recording": bads_whole,
        "channels": channels,
    }
    return raw, qc
