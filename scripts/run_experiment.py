"""Run (or dry-run) an experiment config on a graph run.

    # inputs, shapes and splits only (no training):
    python scripts/run_experiment.py --config configs/experiments/exp5_multiband.yaml \
        --graphs /content/drive/MyDrive/AD-LEB/processed/graphs_v3_<stamp>_<hash> --dry-run
    # full run of selected settings:
    python scripts/run_experiment.py --config ... --graphs ... --settings within_D1 within_D2 \
        --out /content/drive/MyDrive/AD-LEB/processed/experiments
    # framework smoke test on synthetic data (no EEG):
    python scripts/run_experiment.py --config ... --synthetic --out /tmp/smoke \
        --override evaluation.n_repeats=1 --override training.epochs=2

Each run writes to a new timestamped directory under --out; nothing is overwritten.
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from adgraph.experiments.config import expand_variants, load_experiment, set_dotted
from adgraph.experiments.runner import dry_run, run_variant
from adgraph.graphs.dataset import build_dataset
from adgraph.graphs.store import GraphStore

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--graphs", help="graph run directory (graphs.npz, windows.csv, ...)")
src.add_argument("--synthetic", action="store_true", help="use a synthetic store (framework test only)")
parser.add_argument("--settings", nargs="+", help="default: the config's settings list")
parser.add_argument("--variants", nargs="+", help="only variants whose name contains one of these strings")
parser.add_argument("--override", action="append", default=[], help="dotted.key=yaml_value, applied before expansion")
parser.add_argument("--out", help="output root (required unless --dry-run)")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
cfg = load_experiment(args.config)
for item in args.override:
    key, value = item.split("=", 1)
    cfg = set_dotted(cfg, key, yaml.safe_load(value))
variants = expand_variants(cfg)
if args.variants:
    variants = [v for v in variants if any(s in v["name"] for s in args.variants)]
settings = args.settings or cfg["settings"]

if args.synthetic:
    from adgraph.synthetic import make_synthetic_store

    store = make_synthetic_store(psd_effect=0.5, conn_effect=0.5)
else:
    store = GraphStore.load(args.graphs)

if args.dry_run:
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(dry_run(store, variants, settings).to_string())
    raise SystemExit

if not args.out:
    parser.error("--out is required unless --dry-run")
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
out_root = Path(args.out) / f"{cfg['experiment']}_{stamp}"
for v in variants:
    data = build_dataset(store, v)
    for s in settings:
        run_variant(store, v, s, out_root, data=data)
print(f"outputs: {out_root}")
