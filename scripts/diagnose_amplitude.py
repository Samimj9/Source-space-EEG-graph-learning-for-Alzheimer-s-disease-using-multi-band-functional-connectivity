"""Read-only amplitude diagnostic of real recordings (methodology log entry 38).

    python scripts/diagnose_amplitude.py --paths configs/paths.yaml --out <processed_dir>
    python scripts/diagnose_amplitude.py --paths configs/paths.yaml --out <processed_dir> --keys D1-1 D2-2

Default: the first recording of each dataset (the ones the --limit 1 smoke test processes).
For each recording it prints and saves the EDF header units, the per-stage amplitude
statistics behind the preprocessing amplitude check, spectra, drift, clipping and 20 s block
amplitudes, and a list of evidence statements (unit error vs check stage vs noisy EEG).

The EDFs are only read. Outputs go to a new directory <out>/diagnostics/amplitude_<UTC stamp>/
(never overwritten): header_<key>.csv, channels_<key>.csv, spectral_<key>.csv,
blocks_<key>.csv, summary.json, manifest.json, amplitude.png. Nothing in the pipeline changes.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adgraph.checkpoint import atomic_write_csv, atomic_write_json
from adgraph.config import load_paths, load_yaml
from adgraph.io.eeg import list_recordings
from adgraph.preprocessing.amplitude import STAGES, amplitude_report
from adgraph.provenance import git_commit, git_dirty, versions
from adgraph.storage import check_persistent

parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/pipeline.yaml")
parser.add_argument("--out", required=True, help="processed directory (in Drive); a new subfolder is created")
parser.add_argument("--keys", nargs="*", help="recording keys, e.g. D1-1 D2-2 (default: first of each dataset)")
parser.add_argument("--allow-ephemeral-storage", action="store_true")
parser.add_argument("--no-figure", action="store_true")
args = parser.parse_args()
check_persistent({"--out": args.out}, allow_ephemeral=args.allow_ephemeral_storage)

paths, cfg = load_paths(args.paths), load_yaml(args.config)
recs = pd.concat([list_recordings(paths["dataset1"]["eeg_dir"], "D1"),
                  list_recordings(paths["dataset2"]["eeg_dir"], "D2")], ignore_index=True)
if args.keys:
    unknown = sorted(set(args.keys) - set(recs["key"]))
    if unknown:
        raise SystemExit(f"unknown recording keys {unknown}; available e.g. {recs['key'].head(3).tolist()}")
    recs = recs[recs["key"].isin(args.keys)]
else:
    recs = recs.groupby("dataset", sort=True).head(1)

stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
out = Path(args.out) / "diagnostics" / f"amplitude_{stamp}"
out.mkdir(parents=True, exist_ok=False)
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 100)

summaries, reports = {}, {}
for rec in recs.itertuples():
    print(f"\n{'=' * 100}\n{rec.key}  ({Path(rec.path).name}, {rec.size_bytes / 1e6:.1f} MB)\n{'=' * 100}")
    r = amplitude_report(rec.path, rec.dataset, cfg)
    reports[rec.key] = r
    for name in ("header", "channels", "spectral", "blocks"):
        atomic_write_csv(out / f"{name}_{rec.key}.csv", r[name])
    s = r["summary"]
    summaries[rec.key] = s
    h = r["header"]
    print("EDF header (used channels):")
    print(h[h["electrode"] != ""][["label", "electrode", "header_dimension", "mne_to_volt", "physical_min",
                                   "physical_max", "digital_min", "digital_max", "resolution_header_units",
                                   "highpass", "lowpass"]].to_string(index=False))
    other = h[h["electrode"] == ""]
    if len(other):
        print(f"other channels (not used): {other['label'].tolist()}")
    print(f"\nsampling rate {s['orig_sfreq']} Hz, duration {s['duration_s']:.1f} s, dataset scale {s['dataset_scale']}")
    print("\nmedian over channels, per stage (µV):")
    print(pd.DataFrame({"std": s["median_std_uv"], "robust_std": s["median_robust_std_uv"]}).T.round(2)
          .to_string())
    fc, ck = s["former_check"], s["checks"]
    print(f"\nformer check (whole recording, recording reference): {fc['value_v'] * 1e6:.1f} µV -> "
          f"{'PASS' if fc['passes'] else 'FAIL'}")
    print(f"unit safeguard (pipeline): {ck['unit_safeguard']['value_uv']:.2f} µV in {ck['unit_safeguard']['range_uv']} "
          f"µV -> {'PASS' if ck['unit_safeguard']['passes'] else 'FAIL'}")
    pl = ck["plausibility"]
    print("plausibility (pipeline, analysed interval, average reference): "
          + (f"{pl['value_uv']:.2f} µV (robust {pl['robust_value_uv']:.2f}) in {pl['range_uv']} µV -> PASS"
             if pl["passes"] else f"FAIL: {pl.get('error')}"))
    if "channels" in ck:
        print("\nper-channel QC as the pipeline computes it (analysed interval):")
        print(pd.DataFrame(ck["channels"]).set_index("channel").round(2).to_string())
    ch = r["channels"].pivot(index="channel", columns="stage", values="std_uv")[list(STAGES)]
    print("\nper-channel std (µV) by stage:")
    print(ch.round(1).to_string())
    q = r["channels"][r["channels"]["stage"] == "qc"].set_index("channel")
    print("\nper-channel at the QC stage (µV): robust percentiles of |x - median|, drift (unfiltered):")
    sc = r["channels"][r["channels"]["stage"] == "scaled"].set_index("channel")
    print(pd.concat([q[["robust_std_uv", "abs_dev_p50_uv", "abs_dev_p95_uv", "abs_dev_p99_uv", "abs_dev_p999_uv",
                        "ptp_uv"]],
                     sc[["mean_uv", "trend_uv_per_min"]].add_prefix("raw_")], axis=1).round(1).to_string())
    print("\nspectral fractions (QC stage), clipping, line noise, flat samples:")
    print(r["spectral"].set_index("channel").round(3).to_string())
    print("\nevidence:")
    for line in s["evidence"]:
        print(f"  - {line}")

atomic_write_json(out / "summary.json", summaries)
atomic_write_json(out / "manifest.json", {
    "script": "scripts/diagnose_amplitude.py", "created_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": git_commit(), "git_dirty": git_dirty(), "versions": versions(), "config": cfg,
    "inputs": recs[["key", "path", "size_bytes"]].to_dict("records"), "read_only": True})

if not args.no_figure:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(reports), 2, figsize=(14, 4.2 * len(reports)), squeeze=False)
    for row, (key, r) in zip(axes, reports.items()):
        ch = r["channels"].pivot(index="channel", columns="stage", values="std_uv")
        ch = ch.loc[r["channels"]["channel"].unique()]
        for st in STAGES[1:]:
            row[0].plot(range(len(ch)), ch[st], marker="o", ms=3, label=st)
        lo, hi = r["summary"]["former_check"]["plausible_v"]
        row[0].axhline(hi * 1e6, color="k", ls="--", lw=0.8, label="plausibility upper limit")
        row[0].set_yscale("log")
        row[0].set_xticks(range(len(ch)), ch.index, rotation=90, fontsize=7)
        row[0].set_ylabel("channel std (µV)")
        row[0].set_title(f"{key}: per-channel std by stage")
        row[0].legend(fontsize=7, ncol=3)
        b = r["blocks"]
        row[1].plot(b["block_start_s"], b["qc_median_std_uv"], marker="o", ms=3, label="QC stage")
        row[1].plot(b["block_start_s"], b["car_hp1_median_std_uv"], marker="o", ms=3, label="1-45 Hz, avg ref")
        row[1].set_yscale("log")
        row[1].set_xlabel("time (s), 20 s blocks")
        row[1].set_ylabel("median channel std (µV)")
        row[1].set_title(f"{key}: amplitude over time")
        row[1].legend(fontsize=7)
    fig.suptitle("Amplitude diagnostic (read-only; not a paper figure)")
    fig.tight_layout()
    fig.savefig(out / "amplitude.png", dpi=120)
    print(f"\nfigure: {out / 'amplitude.png'}")
print(f"\nsaved to {out}\nShare summary.json (or the 'evidence' lines above) for the decision.")
