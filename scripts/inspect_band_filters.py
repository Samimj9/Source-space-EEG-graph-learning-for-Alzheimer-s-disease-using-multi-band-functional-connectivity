"""Actual frequency responses of the connectivity band-pass filters (no EEG involved).

    python scripts/inspect_band_filters.py [--out-table results/tables/band_filter_responses.csv]
                                           [--out-figure docs/img/band_filter_responses.png]

Builds exactly the filters applied by ``adgraph.connectivity.plv.band_analytic`` (MNE
``filter_data``: FIR, zero phase, firwin design, default 'auto' transition bandwidths and filter
length, at the pipeline sampling rate) for every band in configs/pipeline.yaml, and reports for
each band:

* nominal band (config) and MNE's automatic lower / upper transition bandwidths;
* transition bands (from the stopband edge to the nominal edge) and filter length;
* measured passband (gain >= -0.1 dB), -3 dB and -6 dB edges, and -40 dB stopband edges;
* for a flat (white) input spectrum: share of the band's output power that lies outside its
  nominal band, and inside each other band's nominal band;
* -6 dB overlap with the neighbouring band.

Changes nothing; documentation of the current (unchanged) method.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import freqz

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pipeline.yaml")
parser.add_argument("--design", choices=["config", "auto"], default="config",
                    help="config: connectivity.transition_bandwidth_hz as applied now; auto: MNE 'auto' (before)")
parser.add_argument("--out-table", help="default: results/tables/band_filter_responses_<design>.csv")
parser.add_argument("--out-figure", help="default: docs/img/band_filter_responses_<design>.png")
args = parser.parse_args()
args.out_table = args.out_table or f"results/tables/band_filter_responses_{args.design}.csv"
args.out_figure = args.out_figure or f"docs/img/band_filter_responses_{args.design}.png"

import mne  # noqa: E402

from adgraph.connectivity.plv import transition_kwargs  # noqa: E402

cfg = yaml.safe_load(Path(args.config).read_text())
sfreq = float(cfg["preprocessing"]["sfreq"])
bands = cfg["connectivity"]["bands"]
width = None if args.design == "auto" else cfg["connectivity"].get("transition_bandwidth_hz")
f = np.arange(0.0, 60.0 + 1e-9, 0.005)
resp, rows = {}, []


def edge(mask_from_center, freqs, center, side):
    """First frequency, moving out from ``center`` to ``side`` (-1 low, +1 high), where the mask fails."""
    idx_c = int(np.argmin(np.abs(freqs - center)))
    rng = range(idx_c, -1, -1) if side < 0 else range(idx_c, len(freqs))
    last = idx_c
    for i in rng:
        if not mask_from_center[i]:
            return float(freqs[last])
        last = i
    return float(freqs[last])


for name, (lo, hi) in bands.items():
    h = mne.filter.create_filter(None, sfreq, lo, hi, method="fir", phase="zero", fir_design="firwin",
                                 verbose="error", **transition_kwargs(lo, hi, width))
    # the kernel analysed here is the one the pipeline applies: impulse through filter_data
    imp = np.zeros(int(sfreq * 60))
    imp[len(imp) // 2] = 1.0
    out = mne.filter.filter_data(imp, sfreq, lo, hi, method="fir", phase="zero", fir_design="firwin",
                                 verbose="error", **transition_kwargs(lo, hi, width))
    half = len(h) // 2
    applied = out[len(imp) // 2 - half: len(imp) // 2 - half + len(h)]
    if not np.allclose(applied, h, atol=1e-10):
        raise RuntimeError(f"{name}: kernel from create_filter differs from what filter_data applies")
    _, H = freqz(h, worN=f, fs=sfreq)
    g = np.abs(H)                                        # phase="zero": applied once, magnitude |H|
    resp[name] = g
    if width is None:                                    # MNE's 'auto' transition widths (firwin rules)
        l_tb, h_tb = min(max(lo * 0.25, 2.0), lo), min(max(hi * 0.25, 2.0), sfreq / 2.0 - hi)
    else:
        l_tb, h_tb = min(width, lo), width
    c = (lo + hi) / 2
    db = 20 * np.log10(np.maximum(g, 1e-12))
    rows.append({"band": name, "nominal_low_hz": lo, "nominal_high_hz": hi,
                 "lower_transition_bw_hz": l_tb, "upper_transition_bw_hz": h_tb,
                 "lower_transition_hz": f"{lo - l_tb:g}-{lo:g}", "upper_transition_hz": f"{hi:g}-{hi + h_tb:g}",
                 "filter_length_samples": len(h), "filter_length_s": len(h) / sfreq,
                 "passband_-0.1dB_hz": f"{edge(db >= -0.1, f, c, -1):.2f}-{edge(db >= -0.1, f, c, 1):.2f}",
                 "-3dB_hz": f"{edge(db >= -3, f, c, -1):.2f}-{edge(db >= -3, f, c, 1):.2f}",
                 "-6dB_hz": f"{edge(db >= -6.02, f, c, -1):.2f}-{edge(db >= -6.02, f, c, 1):.2f}",
                 "-40dB_stopband_edges_hz": f"{edge(db >= -40, f, c, -1):.2f}-{edge(db >= -40, f, c, 1):.2f}"})

power = {k: v ** 2 for k, v in resp.items()}
names = list(bands)
for row in rows:
    p = power[row["band"]]
    total = np.trapezoid(p, f)
    lo, hi = bands[row["band"]]
    inside = (f >= lo) & (f < hi)
    row["power_outside_nominal_%"] = 100 * (1 - np.trapezoid(np.where(inside, p, 0), f) / total)
    for other in names:
        if other != row["band"]:
            olo, ohi = bands[other]
            m = (f >= olo) & (f < ohi)
            row[f"power_in_{other}_%"] = 100 * np.trapezoid(np.where(m, p, 0), f) / total
for a, b in zip(names, names[1:]):
    both = (resp[a] >= 0.5) & (resp[b] >= 0.5)
    ov = f[both]
    rows[names.index(a)][f"-6dB_overlap_with_{b}_hz"] = f"{ov.min():.2f}-{ov.max():.2f}" if len(ov) else "none"

table = pd.DataFrame(rows)
Path(args.out_table).parent.mkdir(parents=True, exist_ok=True)
table.to_csv(args.out_table, index=False)
pd.set_option("display.width", 250, "display.max_columns", 50)
print(table.T.to_string())

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
for (name, (lo, hi)), col in zip(bands.items(), colors):
    for ax in axes:
        ax.axvspan(lo, hi, color=col, alpha=0.12, lw=0)
    axes[0].plot(f, resp[name], color=col, lw=1.2, label=f"{name} (nominal {lo:g}-{hi:g} Hz)")
    axes[1].plot(f, 20 * np.log10(np.maximum(resp[name], 1e-6)), color=col, lw=1.2)
axes[0].axhline(0.5, color="k", lw=0.6, ls=":")
axes[0].text(44.5, 0.52, "-6 dB", ha="right", fontsize=7)
axes[0].set_ylabel("gain |H(f)|")
axes[0].legend(fontsize=7, frameon=False)
axes[1].set_ylim(-80, 5)
axes[1].set_ylabel("gain (dB)")
axes[1].set_xlabel("frequency (Hz)")
axes[1].set_xlim(0, 45)
label = "MNE 'auto' transitions (before)" if width is None else f"{width:g} Hz transitions (applied)"
fig.suptitle(f"Connectivity band-pass filters: MNE FIR, zero phase, firwin, {label}, {sfreq:g} Hz\n"
             "shaded: nominal bands", fontsize=8)
fig.tight_layout()
Path(args.out_figure).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out_figure, dpi=200)
print(f"\nwritten {args.out_table} and {args.out_figure}")
