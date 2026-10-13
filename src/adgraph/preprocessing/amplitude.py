"""Read-only amplitude diagnostic of one recording (methodology log entry 38).

Breaks the preprocessing amplitude check (median over channels of the per-channel std) down
by stage, so a failure can be attributed to a unit-conversion error, to the stage at which
the check is applied, or to genuinely noisy EEG:

==========  ======================================================================
stage       signal
==========  ======================================================================
``mne``     as read by MNE (header physical units converted to volts), no dataset scale
``scaled``  after the dataset scale (``units.scale``): what ``load_raw`` returns
``qc``      resampled + 0.1-45 Hz band-pass, recording reference: what the check sees
``hp1``     ``qc`` + 1 Hz high-pass (the signal used for bad-channel detection)
``car``     ``qc`` + average reference over the 30 channels (no interpolation)
``car_hp1`` ``hp1`` + average reference
==========  ======================================================================

Nothing here changes the pipeline or the files; the EDF is only read. All amplitudes are
reported in microvolts (µV) unless a column says otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.eeg import channel_mapping, load_raw
from .clean import resample_and_bandpass

STAGES = ("mne", "scaled", "qc", "hp1", "car", "car_hp1")
BANDS = {"0.1-1": (0.1, 1.0), "1-4": (1.0, 4.0), "4-8": (4.0, 8.0), "8-13": (8.0, 13.0),
         "13-30": (13.0, 30.0), "30-45": (30.0, 45.0)}
# Evidence thresholds used by ``interpret`` (stated, not tuned on the data):
TYPICAL_CAR_STD_UV = (2.0, 100.0)  # resting scalp EEG, 1-45 Hz, average reference: a few to tens of µV
LOW_FREQ_FRACTION = 0.5            # >= half of the 0.1-45 Hz variance below 1 Hz -> slow drift dominates
COMMON_MODE_FRACTION = 0.5         # >= half of the channel variance shared by all channels -> reference/common mode
HEAVY_TAIL_RATIO = 1.5             # std / robust std (1.4826 MAD) -> large transients (artifacts)
CLIP_FRACTION = 1e-3               # samples at the EDF physical limits -> clipping


def channel_stats(data_v: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Per-channel amplitude statistics (µV) of ``data_v`` (channels x samples, volts)."""
    x = np.asarray(data_v, float) * 1e6
    med = np.median(x, axis=1, keepdims=True)
    dev = np.abs(x - med)
    t_min = np.arange(x.shape[1]) / sfreq / 60.0
    slope = np.polyfit(t_min, x.T, 1)[0] if x.shape[1] > 1 else np.zeros(len(x))
    pct = np.percentile(dev, [50, 95, 99, 99.9], axis=1)
    return pd.DataFrame({
        "mean_uv": x.mean(1), "std_uv": x.std(1), "robust_std_uv": 1.4826 * np.median(dev, axis=1),
        "abs_dev_p50_uv": pct[0], "abs_dev_p95_uv": pct[1], "abs_dev_p99_uv": pct[2], "abs_dev_p999_uv": pct[3],
        "ptp_uv": np.ptp(x, axis=1), "trend_uv_per_min": slope,
    })


def band_fractions(data_v: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Per-channel fraction of 0.1-45 Hz power in each band, alpha peak and its spectral density."""
    from scipy.signal import welch

    nper = int(min(data_v.shape[1], 20 * sfreq))          # 0.05 Hz resolution resolves 0.1-1 Hz
    f, p = welch(np.asarray(data_v, float) * 1e6, fs=sfreq, nperseg=nper, axis=1)   # µV^2/Hz
    total = p[:, (f >= 0.1) & (f < 45)].sum(1)
    out = {f"frac_{name}": p[:, (f >= lo) & (f < hi)].sum(1) / total for name, (lo, hi) in BANDS.items()}
    a = (f >= 7) & (f <= 13)
    out["alpha_peak_hz"] = f[a][np.argmax(p[:, a], axis=1)]
    out["alpha_peak_asd_uv_per_rthz"] = np.sqrt(p[:, a].max(1))
    return pd.DataFrame(out)


def line_noise_ratio(data_v: np.ndarray, sfreq: float, line_hz: float) -> np.ndarray:
    """Per-channel spectral density at ``line_hz`` over the median density 35-45 Hz (NaN if above Nyquist)."""
    from scipy.signal import welch

    if line_hz + 2 >= sfreq / 2:
        return np.full(len(data_v), np.nan)
    f, p = welch(np.asarray(data_v, float), fs=sfreq, nperseg=int(min(data_v.shape[1], 4 * sfreq)), axis=1)
    peak = p[:, (f >= line_hz - 1) & (f <= line_hz + 1)].max(1)
    return peak / np.median(p[:, (f >= 35) & (f <= 45)], axis=1)


def block_std(data_v: np.ndarray, sfreq: float, block_s: float = 20.0) -> np.ndarray:
    """Median-over-channels std (µV) of consecutive non-overlapping ``block_s`` blocks."""
    n = int(block_s * sfreq)
    nb = data_v.shape[1] // n
    blocks = np.asarray(data_v[:, :nb * n], float).reshape(len(data_v), nb, n) * 1e6
    return np.median(blocks.std(axis=2), axis=0)


def common_mode_fraction(data_v: np.ndarray) -> float:
    """Variance of the across-channel mean / mean per-channel variance (1 = all channels identical)."""
    x = np.asarray(data_v, float)
    x = x - x.mean(1, keepdims=True)
    return float(x.mean(0).var() / x.var(1).mean())


def header_table(path, recorded: list[str]) -> pd.DataFrame:
    """EDF header of every channel: label, mapped electrode, physical dimension, MNE's conversion
    factor to volts, physical/digital limits, resolution and prefiltering (read via MNE)."""
    import mne

    raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
    mapping, _ = channel_mapping(raw.ch_names, recorded)
    ex = raw._raw_extras[0]                       # header fields MNE parsed (pinned MNE version)
    orig = getattr(raw, "_orig_units", None) or {}
    sel = list(ex.get("sel", range(len(raw.ch_names))))
    rows = []
    for i, ch in enumerate(raw.ch_names):
        j = sel[i] if i < len(sel) else i

        def get(key, default=np.nan, j=j):
            v = ex.get(key)
            return v[j] if v is not None and np.ndim(v) and j < len(v) else default

        pmin, pmax, dmin, dmax = (float(get(k)) for k in ("physical_min", "physical_max", "digital_min", "digital_max"))
        rows.append({"label": ch, "electrode": mapping.get(ch, ""), "header_dimension": orig.get(ch, ""),
                     "mne_to_volt": float(get("units")), "physical_min": pmin, "physical_max": pmax,
                     "digital_min": dmin, "digital_max": dmax,
                     "resolution_header_units": (pmax - pmin) / (dmax - dmin) if dmax != dmin else np.nan,
                     "highpass": str(get("highpass", "")), "lowpass": str(get("lowpass", "")),
                     "sfreq": raw.info["sfreq"]})
    return pd.DataFrame(rows)


def clipping_fraction(data_v: np.ndarray, header: pd.DataFrame, ch_names: list[str], scale: float) -> np.ndarray:
    """Per-channel fraction of samples at the EDF physical limits (within half a resolution step)."""
    h = header.set_index("electrode").loc[ch_names]
    k = (h["mne_to_volt"] * scale).to_numpy()[:, None]
    lo, hi = h["physical_min"].to_numpy()[:, None] * k, h["physical_max"].to_numpy()[:, None] * k
    tol = np.abs(h["resolution_header_units"].to_numpy()[:, None] * k) / 2
    x = np.asarray(data_v, float)
    return ((x <= lo + tol) | (x >= hi - tol)).mean(1)


def amplitude_report(path, dataset: str, cfg: dict) -> dict:
    """All diagnostics for one recording. Returns ``{"header", "channels", "spectral", "blocks",
    "summary"}`` (DataFrames and a dict). Reads the EDF; writes nothing."""
    recorded = cfg["channels"]["recorded"]
    header = header_table(path, recorded)
    raw, info = load_raw(path, dataset, cfg)                  # stage "scaled" (exactly as the pipeline)
    scale = float(info["unit_scale"])
    sf0, ch = raw.info["sfreq"], list(raw.ch_names)
    scaled = raw.get_data()
    stages = {"mne": (scaled / scale, sf0), "scaled": (scaled, sf0)}
    qc = resample_and_bandpass(raw.copy(), cfg)              # identical calls to preprocess()
    hp1 = qc.copy().filter(1.0, None, fir_design="firwin", verbose="error")
    stages["qc"] = (qc.get_data(), qc.info["sfreq"])
    stages["hp1"] = (hp1.get_data(), hp1.info["sfreq"])
    for name, r in (("car", qc), ("car_hp1", hp1)):
        c = r.copy().set_eeg_reference("average", projection=True, verbose="error").apply_proj(verbose="error")
        stages[name] = (c.get_data(), c.info["sfreq"])
    del raw, qc, hp1

    per_stage = []
    for name in STAGES:
        d, sf = stages[name]
        st = channel_stats(d, sf)
        st.insert(0, "channel", ch)
        st.insert(0, "stage", name)
        per_stage.append(st)
    channels = pd.concat(per_stage, ignore_index=True)

    d_qc, sf_qc = stages["qc"]
    bands = band_fractions(d_qc, sf_qc)
    bands.insert(0, "channel", ch)
    clip = clipping_fraction(scaled, header, ch, scale)
    bands["clip_fraction"] = clip
    bands["line50_ratio"] = line_noise_ratio(scaled, sf0, 50.0)
    bands["line60_ratio"] = line_noise_ratio(scaled, sf0, 60.0)
    bands["flat_fraction"] = (np.diff(scaled, axis=1) == 0).mean(1)

    nb = len(block_std(d_qc, sf_qc))
    blocks = pd.DataFrame({"block_start_s": np.arange(nb) * 20.0,
                           "scaled_median_std_uv": block_std(scaled, sf0)[:nb],    # unfiltered (edge check)
                           "qc_median_std_uv": block_std(d_qc, sf_qc),
                           "car_hp1_median_std_uv": block_std(stages["car_hp1"][0], sf_qc)})
    trim = int(cfg["windowing"]["edge_trim_s"] * sf_qc)
    med = channels.groupby("stage", sort=False)[["std_uv", "robust_std_uv"]].median()
    used = header[header["electrode"].isin(ch)]
    lo, hi = cfg["units"]["plausible_std_v"]
    summary = {
        "path": str(path), "dataset": dataset, "orig_sfreq": sf0, "duration_s": scaled.shape[1] / sf0,
        "header_dimensions": sorted(used["header_dimension"].astype(str).unique().tolist()),
        "mne_to_volt": sorted(used["mne_to_volt"].unique().tolist()), "dataset_scale": scale,
        "physical_range_header_units": [float(used["physical_min"].min()), float(used["physical_max"].max())],
        "resolution_header_units": float(used["resolution_header_units"].median()),
        "prefiltering": sorted({f"HP {a} / LP {b}" for a, b in zip(used["highpass"], used["lowpass"])}),
        "median_std_uv": {s: float(med.loc[s, "std_uv"]) for s in STAGES},
        "median_robust_std_uv": {s: float(med.loc[s, "robust_std_uv"]) for s in STAGES},
        "qc_trimmed_median_std_uv": float(np.median(d_qc[:, trim:d_qc.shape[1] - trim].std(1)) * 1e6),
        "edge_trim_s": cfg["windowing"]["edge_trim_s"],
        "qc_check": {"value_v": float(med.loc["qc", "std_uv"]) * 1e-6, "plausible_v": [lo, hi],
                     "passes": bool(lo <= med.loc["qc", "std_uv"] * 1e-6 <= hi)},
        "median_abs_dc_offset_uv": float(np.median(np.abs(channels.loc[channels.stage == "scaled", "mean_uv"]))),
        "median_abs_trend_uv_per_min": float(np.median(np.abs(
            channels.loc[channels.stage == "scaled", "trend_uv_per_min"]))),
        "median_band_fraction_qc": {b: float(bands[f"frac_{b}"].median()) for b in BANDS},
        "median_alpha_peak_hz": float(bands["alpha_peak_hz"].median()),
        "median_alpha_peak_asd_uv_per_rthz": float(bands["alpha_peak_asd_uv_per_rthz"].median()),
        "common_mode_fraction": {s: common_mode_fraction(stages[s][0]) for s in ("qc", "hp1")},
        "std_over_robust_std_qc": float(med.loc["qc", "std_uv"] / med.loc["qc", "robust_std_uv"]),
        "block_qc_std_uv_percentiles": {f"p{q}": float(np.percentile(blocks["qc_median_std_uv"], q))
                                        for q in (10, 50, 90, 100)} if len(blocks) else {},
        "max_clip_fraction": float(clip.max()),
        "channels_clipping": [c for c, f in zip(ch, clip) if f > CLIP_FRACTION],
        "median_line50_ratio": float(np.nanmedian(bands["line50_ratio"])) if np.isfinite(bands["line50_ratio"]).any()
        else None,
        "median_flat_fraction": float(bands["flat_fraction"].median()),
    }
    summary["evidence"] = interpret(summary)
    return {"header": header, "channels": channels, "spectral": bands, "blocks": blocks, "summary": summary}


def interpret(s: dict) -> list[str]:
    """Plain statements of what the numbers indicate, with the criterion used for each.
    They are evidence for the author's decision, not an automatic decision."""
    out = []
    m = s["median_std_uv"]
    car = m["car_hp1"]
    lo_t, hi_t = TYPICAL_CAR_STD_UV
    out.append(f"QC stage (0.1-45 Hz, recording reference): median channel std {m['qc']:.1f} µV; "
               f"after 1 Hz high-pass {m['hp1']:.1f} µV; average reference {m['car']:.1f} µV; "
               f"both {car:.1f} µV.")
    if lo_t <= car <= hi_t:
        out.append(f"UNITS: 1-45 Hz average-referenced std {car:.1f} µV lies in the typical resting-EEG range "
                   f"{lo_t:g}-{hi_t:g} µV -> no evidence of a unit-conversion error (a wrong factor would "
                   "shift every stage by 10^3 or 10^6).")
    else:
        out.append(f"UNITS: 1-45 Hz average-referenced std {car:.3g} µV is OUTSIDE the typical {lo_t:g}-{hi_t:g} µV "
                   f"-> check the unit chain (header {s['header_dimensions']}, MNE factor {s['mne_to_volt']}, "
                   f"dataset scale {s['dataset_scale']}).")
    lf = s["median_band_fraction_qc"]["0.1-1"]
    cm = s["common_mode_fraction"]
    if m["qc"] > 2 * car:
        causes = []
        if lf >= LOW_FREQ_FRACTION:
            causes.append(f"slow drift ({lf:.0%} of the 0.1-45 Hz power is below 1 Hz)")
        if cm["qc"] >= COMMON_MODE_FRACTION:
            causes.append(f"a common-mode/reference signal ({cm['qc']:.0%} of the channel variance is shared "
                          "by all channels and removed by the average reference)")
        out.append(f"STAGE: the QC value is {m['qc'] / car:.1f}x the 1-45 Hz average-referenced std; the excess is "
                   + (" and ".join(causes) if causes else "not explained by drift or common mode alone") + ".")
    else:
        out.append("STAGE: the QC value is within 2x the 1-45 Hz average-referenced std; the check stage does "
                   "not inflate the amplitude.")
    out.append(f"EDGES: QC-stage median std without the first/last {s['edge_trim_s']:g} s (the part that is "
               f"windowed) is {s['qc_trimmed_median_std_uv']:.1f} µV vs {m['qc']:.1f} µV on the whole recording"
               + (" -> transients at the recording edges inflate the check." if m["qc"] > 1.5 * s["qc_trimmed_median_std_uv"]
                  else "."))
    ratio = s["std_over_robust_std_qc"]
    blocks = s.get("block_qc_std_uv_percentiles") or {}
    tail = blocks and blocks["p50"] > 0 and blocks["p100"] / blocks["p50"]
    out.append(f"ARTIFACTS: std / robust std = {ratio:.2f} (> {HEAVY_TAIL_RATIO} indicates large transients); "
               + (f"20 s block std p50 {blocks['p50']:.1f} µV, p90 {blocks['p90']:.1f} µV, max {blocks['p100']:.1f} µV"
                  f" (max/median {tail:.1f})." if blocks else "too short for blocks."))
    out.append(f"CLIPPING: max fraction of samples at the EDF physical limits {s['max_clip_fraction']:.2g}"
               + (f" in {s['channels_clipping']}" if s["channels_clipping"] else " (none above 0.1 %)") + ".")
    out.append(f"DRIFT: median |DC offset| {s['median_abs_dc_offset_uv']:.0f} µV, median |linear trend| "
               f"{s['median_abs_trend_uv_per_min']:.1f} µV/min (unfiltered).")
    out.append(f"SPECTRUM: median alpha peak {s['median_alpha_peak_hz']:.1f} Hz, "
               f"{s['median_alpha_peak_asd_uv_per_rthz']:.1f} µV/sqrt(Hz); band fractions "
               + ", ".join(f"{b} Hz {v:.0%}" for b, v in s["median_band_fraction_qc"].items()) + ".")
    return out
