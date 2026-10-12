"""Header-only check of every resting-state EDF: channel names, sampling rate, duration.

Run this before build_graphs.py; it reads no signal data.

    python scripts/inspect_raw.py --paths configs/paths.yaml --out /content/drive/MyDrive/AD-LEB/processed
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adgraph.config import load_paths, load_yaml
from adgraph.io.eeg import inspect_headers, list_recordings

parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/pipeline.yaml")
parser.add_argument("--out", required=True, help="directory for the header report (in Drive)")
args = parser.parse_args()

paths, cfg = load_paths(args.paths), load_yaml(args.config)
recs = pd.concat([list_recordings(paths["dataset1"]["eeg_dir"], "D1"),
                  list_recordings(paths["dataset2"]["eeg_dir"], "D2")], ignore_index=True)
report = inspect_headers(recs, cfg["channels"]["recorded"])
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
out = Path(args.out) / f"header_report_{stamp}.csv"
report.to_csv(out, index=False)
pd.set_option("display.width", 200)
print(report[["key", "n_channels", "sfreq", "duration_s", "missing_canonical", "unrecognised"]].to_string())
print(f"\nwritten to {out}")
