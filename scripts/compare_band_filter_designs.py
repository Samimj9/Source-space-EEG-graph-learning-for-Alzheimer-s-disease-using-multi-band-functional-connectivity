"""Compare connectivity band-pass designs on synthetic signals (nominal bands unchanged; no EEG).

    python scripts/compare_band_filter_designs.py [--out results/tables/band_filter_design_comparison.csv]

Every design keeps the nominal bands of configs/pipeline.yaml, MNE FIR, zero phase, firwin
(Hamming), 'auto' filter length and filtering of the continuous signal before windowing.
Only the transition bandwidths differ:

* ``auto``: MNE's default (lower = min(max(0.25 l, 2), l), upper = min(max(0.25 h, 2), ...)),
  the design used up to methodology log entry 35;
* ``uniform_<w>``: the same transition width ``w`` Hz on every band edge.

Measured per design and band:
* frequency response: passband (>= -0.1 dB), -6 dB band, out-of-nominal-band power share
  (flat spectrum), -6 dB overlap with the neighbouring band;
* time domain: filter length (s), half-length vs the 10 s edge trim (continuous filtering:
  only the trimmed recording edges see filter start-up), 90 % energy duration of the impulse
  response (temporal smearing of the phase estimate);
* PLV / |ciPLV| on synthetic ROI pairs (20 s windows, 250 Hz, as in the pipeline):
  - band specificity: a coupled oscillation at 9 Hz (inside alpha1 only) or 11.5 Hz (inside
    alpha2 only) with a fixed non-zero phase lag, buried in independent 1/f noise; PLV and
    |ciPLV| are reported in alpha1 and alpha2 (ideal: high in the coupled band only);
  - null level: independent 1/f noise only (no coupling); mean PLV and |ciPLV| over windows
    (the estimator's floor, which rises as the effective bandwidth narrows).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import freqz, hilbert

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pipeline.yaml")
parser.add_argument("--widths", nargs="+", type=float, default=[1.5, 1.0, 0.5])
parser.add_argument("--out", default="results/tables/band_filter_design_comparison.csv")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import mne  # noqa: E402

cfg = yaml.safe_load(Path(args.config).read_text())
sf = float(cfg["preprocessing"]["sfreq"])
bands = cfg["connectivity"]["bands"]
win_s, trim_s = cfg["windowing"]["window_s"], cfg["windowing"]["edge_trim_s"]
f = np.arange(0.0, 60.0 + 1e-9, 0.005)
designs = {"auto": None, **{f"uniform_{w:g}": w for w in args.widths}}


def kernel(lo, hi, w):
    kw = {} if w is None else {"l_trans_bandwidth": min(w, lo), "h_trans_bandwidth": w}
    return mne.filter.create_filter(None, sf, lo, hi, method="fir", phase="zero", fir_design="firwin",
                                    verbose="error", **kw)


def edges(g, lo, hi, thr):
    c = int(np.argmin(np.abs(f - (lo + hi) / 2)))
    a = c
    while a > 0 and g[a - 1] >= thr:
        a -= 1
    b = c
    while b < len(f) - 1 and g[b + 1] >= thr:
        b += 1
    return f[a], f[b]


def pink(rng, n_sig, n):
    x = rng.standard_normal((n_sig, n))
    X = np.fft.rfft(x, axis=-1)
    fr = np.fft.rfftfreq(n, 1 / sf)
    X[:, 1:] /= np.sqrt(fr[1:])
    return np.fft.irfft(X, n, axis=-1)


def phase_sync(a):
    z = a / np.abs(a)
    c = (z @ z.conj().T) / a.shape[-1]
    plv = np.abs(c)
    ci = np.abs(c.imag / np.sqrt(np.clip(1 - c.real ** 2, 1e-12, None)))
    return plv[0, 1], ci[0, 1]


def analytic(x, lo, hi, w):
    kw = {} if w is None else {"l_trans_bandwidth": min(w, lo), "h_trans_bandwidth": w}
    return hilbert(mne.filter.filter_data(x, sf, lo, hi, method="fir", phase="zero", fir_design="firwin",
                                          verbose="error", **kw), axis=-1)


rng = np.random.default_rng(args.seed)
dur = 300.0                                                 # 5 min continuous, then 20 s windows
n = int(dur * sf)
t = np.arange(n) / sf
win, step, trim = int(win_s * sf), int(win_s * sf / 2), int(trim_s * sf)
starts = np.arange(trim, n - trim - win + 1, step)


def scenario(freq):
    noise = pink(rng, 2, n)
    noise /= noise.std(axis=1, keepdims=True)
    if freq is None:
        return noise
    lag = np.pi / 3                                        # non-zero lag: visible to |ciPLV| too
    amp_mod = 1 + 0.3 * np.sin(2 * np.pi * 0.05 * t)
    phase = 2 * np.pi * freq * t + np.cumsum(rng.standard_normal(n)) * 0.02   # slightly drifting oscillation
    sig = np.vstack([np.sin(phase), np.sin(phase - lag)]) * amp_mod
    return noise + 0.6 * sig


signals = {"coupled_9Hz": scenario(9.0), "coupled_11.5Hz": scenario(11.5), "null_noise": scenario(None)}
rows = []
for dname, w in designs.items():
    resp = {}
    for b, (lo, hi) in bands.items():
        h = kernel(lo, hi, w)
        g = np.abs(freqz(h, worN=f, fs=sf)[1])
        resp[b] = g
        p = g ** 2
        inside = (f >= lo) & (f < hi)
        energy = np.cumsum(np.sort(h ** 2)[::-1]) / np.sum(h ** 2)
        k90 = int(np.searchsorted(energy, 0.9)) + 1
        row = {"design": dname, "band": b, "nominal_hz": f"{lo:g}-{hi:g}",
               "transition_low_hz": (min(max(lo * 0.25, 2), lo) if w is None else min(w, lo)),
               "transition_high_hz": (min(max(hi * 0.25, 2), sf / 2 - hi) if w is None else w),
               "filter_length_s": len(h) / sf, "half_length_s": (len(h) // 2) / sf,
               "half_length_within_edge_trim": (len(h) // 2) / sf < trim_s,
               "impulse_90pct_energy_s": k90 / sf,
               "passband_0.1dB_hz": "%.2f-%.2f" % edges(g, lo, hi, 10 ** (-0.1 / 20)),
               "-6dB_hz": "%.2f-%.2f" % edges(g, lo, hi, 0.5),
               "power_outside_nominal_%": 100 * (1 - np.trapezoid(np.where(inside, p, 0), f) / np.trapezoid(p, f))}
        for sname, x in signals.items():
            a = analytic(x, lo, hi, w)
            vals = np.array([phase_sync(a[:, s:s + win]) for s in starts])
            row[f"PLV_{sname}"] = vals[:, 0].mean()
            row[f"ciPLV_{sname}"] = vals[:, 1].mean()
        rows.append(row)
    names = list(bands)
    for a_, b_ in zip(names, names[1:]):
        ov = f[(resp[a_] >= 0.5) & (resp[b_] >= 0.5)]
        for r in rows:
            if r["design"] == dname and r["band"] == a_:
                r[f"-6dB_overlap_next_hz"] = f"{ov.min():.2f}-{ov.max():.2f}" if len(ov) else "none"

table = pd.DataFrame(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
table.to_csv(args.out, index=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cols = ["design", "band", "-6dB_hz", "-6dB_overlap_next_hz", "power_outside_nominal_%", "filter_length_s",
        "half_length_within_edge_trim", "impulse_90pct_energy_s"]
print(table[cols].to_string(index=False))
print()
print(table[["design", "band"] + [c for c in table if c.startswith(("PLV_", "ciPLV_"))]].to_string(index=False))
print(f"\nwritten {args.out}")
