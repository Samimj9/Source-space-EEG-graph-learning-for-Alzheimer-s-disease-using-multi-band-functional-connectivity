"""Header-only check of every resting-state EDF: channel names, sampling rate, duration.

Run this before build_graphs.py; it reads no signal data. It first verifies the electrode
layout (``electrode_layout`` in paths.yaml, the AD-LEB ``.loc`` file) against
``channels.recorded``: the EDFs label channels by number only (D1 ``EEG E01``-``E32``, D2
``Misc ExG1``-``ExG32``), and channel *k* is electrode *k* of that layout.

    python scripts/legacy/inspect_raw.py --paths configs/paths.yaml --out /content/drive/MyDrive/AD-LEB/processed
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adgraph.config import common_channels, load_paths, load_yaml
from adgraph.io.eeg import inspect_headers, list_recordings
from adgraph.io.layout import check_layout
from adgraph.storage import check_persistent

parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/pipeline.yaml")
parser.add_argument("--out", required=True, help="directory for the header report (in Drive)")
parser.add_argument("--allow-ephemeral-storage", action="store_true", help="permit --out outside the mounted Drive")
parser.add_argument("--strict", action="store_true",
                    help="exit with an error if any recording lacks a canonical channel (automatic gate)")
args = parser.parse_args()
check_persistent({"--out": args.out}, allow_ephemeral=args.allow_ephemeral_storage)

paths, cfg = load_paths(args.paths), load_yaml(args.config)
try:
    layout = check_layout(paths, cfg["channels"]["recorded"])
except (OSError, ValueError) as e:
    if args.strict:
        raise SystemExit(f"header check failed: electrode layout cannot be verified: {e}")
    layout = None
    print(f"warning: electrode layout cannot be verified: {e}")
if layout:
    print(f"electrode layout: {layout['path']}\n  sha256 {layout['sha256']}\n  order (index 1..{len(layout['labels'])}): "
          + " ".join(f"{i}:{lab}" for i, lab in enumerate(layout["labels"], start=1))
          + "\n  matches channels.recorded in configs/pipeline.yaml")
recs = pd.concat([list_recordings(paths["dataset1"]["eeg_dir"], "D1"),
                  list_recordings(paths["dataset2"]["eeg_dir"], "D2")], ignore_index=True)
report = inspect_headers(recs, cfg["channels"]["recorded"])
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
out = Path(args.out) / f"header_report_{stamp}.csv"
report.to_csv(out, index=False)
pd.set_option("display.width", 200)
print(report[["key", "n_channels", "sfreq", "duration_s", "naming", "missing_canonical", "unrecognised"]].to_string())
print(f"\nwritten to {out}")
for (ds, naming), g in report.groupby(["dataset", "naming"]):
    first = str(g["mapping"].iloc[0]).split(",")
    print(f"{ds}: {len(g)} recording(s), {naming} labels, sfreq {sorted(set(g['sfreq']))} Hz; "
          f"mapping e.g. {', '.join(first[:3])} ... {first[-1]}")
used = set(common_channels(cfg))                        # the 30 channels entering source modelling
report["missing_used"] = [",".join(c for c in str(m).split(",") if c in used) for m in report["missing_canonical"]]
missing = report[report["missing_used"].str.len() > 0]
errors = report[report["mapping_error"].str.len() > 0]
if args.strict and len(errors):
    raise SystemExit(f"header check failed: {len(errors)} recording(s) have an ambiguous channel inventory:\n"
                     + errors[["key", "mapping_error"]].to_string(index=False))
if args.strict and len(missing):
    raise SystemExit(f"header check failed: {len(missing)} recording(s) lack channels used by the pipeline:\n"
                     + missing[["key", "missing_used"]].to_string(index=False))
print(f"header check passed: every recording has all {len(used)} channels used by the pipeline "
      f"({len(cfg['channels']['recorded'])} recorded, {', '.join(cfg['channels']['drop'])} dropped)")
