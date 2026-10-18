"""Run (or dry-run) an experiment config on a graph run.

    # inputs, shapes and splits only (no training):
    python scripts/run_experiment.py --config configs/experiments/exp5_multiband.yaml \
        --graphs /content/drive/MyDrive/AD-LEB/processed/graphs_v3_<config hash> --dry-run
    # full run of selected settings (resumes automatically if interrupted):
    python scripts/run_experiment.py --config ... --graphs ... --settings within_D1 within_D2 \
        --out /content/drive/MyDrive/AD-LEB/processed/experiments
    # what is completed / partial / failed, without running anything:
    python scripts/run_experiment.py --config ... --graphs ... --out ... --status
    # framework smoke test on synthetic data (no EEG):
    python scripts/run_experiment.py --config ... --synthetic --out /tmp/smoke \
        --override evaluation.n_repeats=1 --override training.epochs=2

Results go to <out>/<experiment>/<variant>/<setting>/. Completed outer folds, inner-CV trials
and permutations are checkpointed there; rerunning the same command continues unfinished work
(--resume, default). Results from a different configuration or graph run are never reused:
the run stops with IncompatibleCheckpoint unless --force moves them to <out>/_superseded/.
Checkpoints are tied to the code (git commit + source hash): resume with the same commit.
For real graph runs --out must be on the mounted Google Drive (refused otherwise, unless
--allow-ephemeral-storage) and the pinned environment (requirements.txt) is verified.
"""

import argparse
import logging

import pandas as pd
import yaml

from adgraph.evaluation.splits import CROSS_SETTINGS
from adgraph.experiments.config import expand_variants, load_experiment, set_dotted
from adgraph.experiments.gate import gate_failures
from adgraph.experiments.plan import variant_matches
from adgraph.experiments.runner import dry_run, experiment_status, run_variant
from adgraph.graphs.dataset import build_dataset
from adgraph.graphs.store import GraphStore
from adgraph.provenance import verify_environment

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--graphs", help="graph run directory (graphs.npz, windows.csv, ...)")
src.add_argument("--synthetic", action="store_true", help="use a synthetic store (framework test only)")
parser.add_argument("--settings", nargs="+", help="default: the config's settings list")
parser.add_argument("--variants", nargs="+",
                    help="only variants containing one of these as whole name tokens ('plv' excludes 'ciplv')")
parser.add_argument("--override", action="append", default=[], help="dotted.key=yaml_value, applied before expansion")
parser.add_argument("--out", help="output root (required unless --dry-run)")
parser.add_argument("--dry-run", action="store_true")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--resume", dest="resume", action="store_true", default=True,
                  help="continue unfinished work, skip completed units (default)")
mode.add_argument("--no-resume", dest="resume", action="store_false",
                  help="refuse to touch an existing output directory")
mode.add_argument("--force", action="store_true",
                  help="recompute intentionally; existing outputs are moved to <out>/_superseded/")
parser.add_argument("--status", action="store_true", help="show completed/partial/failed work and exit")
parser.add_argument("--allow-ephemeral-storage", action="store_true",
                    help="permit --out outside the mounted Drive for a real run (outputs are lost on disconnect)")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
cfg = load_experiment(args.config)
for item in args.override:
    key, value = item.split("=", 1)
    cfg = set_dotted(cfg, key, yaml.safe_load(value))
variants = expand_variants(cfg)
if args.variants:
    variants = [v for v in variants if variant_matches(v, args.variants)]
settings = args.settings or cfg["settings"]
own = lambda v: [s for s in settings if s in v["settings"]]  # noqa: E731  (a variant may restrict its settings)

if args.synthetic:
    from adgraph.synthetic import make_synthetic_store

    store = make_synthetic_store(psd_effect=0.5, conn_effect=0.5)
elif args.status:
    store = GraphStore.load(args.graphs, keys=())            # status needs no feature arrays
else:
    store = GraphStore.load(args.graphs)

if args.dry_run:
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(dry_run(store, variants, settings).to_string())
    raise SystemExit

if not args.out:
    parser.error("--out is required unless --dry-run")
pd.set_option("display.width", 250, "display.max_colwidth", 80, "display.max_rows", 500)
if args.status:
    rows = [experiment_status(store, v, s, args.out) for v in variants for s in own(v)]
    print(pd.DataFrame(rows).to_string())
    raise SystemExit
if not args.synthetic:
    verify_environment()
gate = gate_failures(store)
for v in variants:
    data = None
    for s in own(v):
        if s in CROSS_SETTINGS and gate and not v["evaluation"].get("allow_ungated_cross_dataset", False):
            print(f"{v['name']} / {s}: not run, cross-dataset gate not passed: {'; '.join(gate)}")
            continue
        data = build_dataset(store, v) if data is None else data   # one dataset per variant, all settings
        summary = run_variant(store, v, s, args.out, data=data, resume=args.resume, force=args.force,
                              allow_ephemeral=args.allow_ephemeral_storage)
print(f"outputs under: {args.out}")
