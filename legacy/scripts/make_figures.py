"""Generate publication figures from SAVED outputs only (no training, no EEG processing).

    python scripts/make_figures.py --config configs/figures/paper.yaml \
        --results /content/drive/MyDrive/AD-LEB/processed/experiments \
        --graphs  /content/drive/MyDrive/AD-LEB/processed/graphs_v3_<config hash> \
        [--stats stats.csv] --out /content/drive/MyDrive/AD-LEB/processed/figures

    # framework demo on synthetic inputs (watermarked "SYNTHETIC DATA – NOT RESULTS"):
    python scripts/make_figures.py --config configs/figures/paper.yaml --synthetic-demo --out /tmp/figdemo

Each run writes a new timestamped folder; existing figures are never overwritten.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from adgraph.graphs.store import GraphStore
from adgraph.progress import log_handler_context
from adgraph.storage import check_persistent
from adgraph.visualization.figures import make_figures

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--style", default="configs/figures/style.yaml")
parser.add_argument("--results", help="root containing saved experiment output directories")
parser.add_argument("--graphs", help="graph run directory (for PSD / connectivity / PSD-embedding figures)")
parser.add_argument("--stats", help="saved statistics CSV with corrected p-values (optional)")
parser.add_argument("--only", nargs="+", help="figure ids to generate")
parser.add_argument("--on-missing", choices=["skip", "error"], default="skip")
parser.add_argument("--synthetic-demo", action="store_true", help="build synthetic inputs and plot them (watermarked)")
parser.add_argument("--out", required=True)
parser.add_argument("--allow-ephemeral-storage", action="store_true", help="permit --out outside the mounted Drive")
args = parser.parse_args()
if not args.synthetic_demo:
    check_persistent({"--out": args.out}, allow_ephemeral=args.allow_ephemeral_storage)

config = yaml.safe_load(Path(args.config).read_text())
out = Path(args.out) / f"figures_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
store, results = None, args.results
if args.synthetic_demo:
    from adgraph.visualization.demo import build_synthetic_results

    store, results = build_synthetic_results(out / "_synthetic_inputs")
elif args.graphs:
    store = GraphStore.load(args.graphs)

with log_handler_context():
    writer = make_figures(config, out, results, store, args.stats, args.style, synthetic=args.synthetic_demo,
                          only=args.only, on_missing=args.on_missing)
print(f"{len(writer.figures)} figures written to {out}")
if writer.skipped:
    print("skipped (inputs not saved yet, or refused):\n" + json.dumps(writer.skipped, indent=1))
if writer.incomplete:
    print("drawn but INCOMPLETE (some requested results are not saved yet; listed in each figure's "
          "metadata as missing_inputs):")
    for f in writer.incomplete:
        print(f"  {f['id']} ({f['status']}): {len(f['missing_inputs'])} missing, e.g. {f['missing_inputs'][:3]}")
