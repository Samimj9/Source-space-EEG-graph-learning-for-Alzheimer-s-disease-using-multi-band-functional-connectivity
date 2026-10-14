"""Dataset 3 (OpenNeuro ds004504, AD and HC only) inputs: download and input check.

    python scripts/d3_data.py download   # metadata + 65 AD/HC preprocessed derivatives to paths.yaml dataset3.root
    python scripts/d3_data.py check      # participants (36 AD / 29 HC, FTD excluded) + header of every recording

Resumable and verified (size + MD5 against the OpenNeuro bucket). FTD recordings are never downloaded or used.
The check reads headers only (channel names, sampling rate, duration) and writes
``<processed>/d3_header_report_<time>.csv``; it exits 1 if a recording is missing or lacks a channel.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adgraph.config import load_paths, load_yaml
from adgraph.io import bids
from adgraph.io.eeg import channel_mapping, read_header

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["download", "check"])
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/pipeline_d3.yaml")
parser.add_argument("--base-url", default=None, help="OpenNeuro bucket URL (tests: file:// mirror)")
parser.add_argument("--allow-ephemeral-storage", action="store_true")
args = parser.parse_args()
paths, cfg = load_paths(args.paths), load_yaml(args.config)
root = Path(paths["dataset3"]["root"])
pd.set_option("display.width", 220, "display.max_rows", 200)

if not args.allow_ephemeral_storage:
    from adgraph.storage import check_persistent

    check_persistent({"Dataset 3 (ds004504)": root}, allow_ephemeral=False)

if args.action == "download":
    from adgraph.io.openneuro import BASE_URL, download_ds004504

    root.mkdir(parents=True, exist_ok=True)
    m = download_ds004504(root, args.base_url or BASE_URL,
                          progress=lambda i, n, sub: print(f"[{i + 1}/{n}] {sub}", flush=True))
    st = pd.Series([f["status"] for f in m["files"]]).value_counts().to_dict()
    print(f"\n{m['dataset']} {m['release']} ({m['doi']}, {m['license']}): cohort used {m['cohort']['used']}, "
          f"{len(m['cohort']['excluded'])} FTD participants excluded by design (not downloaded)")
    print(f"files: {st}; total {sum(f['size'] for f in m['files']) / 1e9:.2f} GB in {root}")
    print(f"provenance: {root / 'download_manifest.json'}")
    sys.exit(0)

kept, excluded = bids.load_participants_tsv(bids.participants_path(root))
bids.check_cohort(kept)
print(f"participants: {kept['label'].value_counts().to_dict()} used; {len(excluded)} excluded by design "
      f"({excluded['group'].value_counts().to_dict()})")
print(kept.groupby("label").agg(n=("key", "size"), age_median=("age", "median"), mmse_median=("mmse", "median")))
recs = bids.list_recordings(root)
recs = recs[recs["key"].isin(kept["key"])]
missing = sorted(set(kept["key"]) - set(recs["key"]))
rows = []
for r in recs.itertuples():
    raw = read_header(r.path, cfg["input"]["format"])
    mapping, unknown = channel_mapping(raw.ch_names, cfg["channels"]["recorded"])
    rows.append({"key": r.key, "n_channels": len(raw.ch_names), "sfreq": raw.info["sfreq"],
                 "duration_s": round(raw.n_times / raw.info["sfreq"], 1),
                 "missing_channels": ",".join(c for c in cfg["channels"]["recorded"] if c not in mapping.values()),
                 "unrecognised": ",".join(unknown), "size_mb": round(r.size_bytes / 1e6, 1)})
rep = pd.DataFrame(rows).merge(kept[["key", "label"]], on="key")
out = Path(paths["processed_dir"]) / f"d3_header_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
rep.to_csv(out, index=False)
print(rep.groupby("label").agg(recordings=("key", "size"), sfreq=("sfreq", lambda s: sorted(set(s))),
                               duration_min_s=("duration_s", "min"), duration_median_s=("duration_s", "median")))
bad = rep[rep["missing_channels"] != ""]
problems = ([f"missing recordings: {missing}"] if missing else []) + \
           ([f"recordings without all 19 channels: {bad['key'].tolist()}"] if len(bad) else [])
print(f"\nheader report: {out}")
if problems:
    print("FAILED: " + "; ".join(problems))
    sys.exit(1)
print(f"OK: {len(rep)} recordings (36 AD + 29 HC), all with the 19 configured channels")
