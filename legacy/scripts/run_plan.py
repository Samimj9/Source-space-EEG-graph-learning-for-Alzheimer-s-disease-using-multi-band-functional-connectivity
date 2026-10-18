"""Run every experiment in the committed run plan (configs/run_plan.yaml), resumably.

    python scripts/run_plan.py --paths configs/paths.yaml            # run / resume everything
    python scripts/run_plan.py --paths configs/paths.yaml --status   # progress only, nothing computed

Graphs default to the most recently assembled full graph run in processed_dir; results go to
<processed_dir>/experiments/<experiment>/<variant>/<setting>/ (same layout as run_experiment.py).
Rerunning after a disconnect continues where it stopped. A failing unit is recorded in
<out>/_plan_status.csv and the plan continues with the next one. Cross-dataset settings are
skipped (reason recorded) when the graph run does not pass the DESIGN.md §7 gate.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from adgraph.config import load_paths
from adgraph.experiments.plan import load_plan, plan_status, run_plan
from adgraph.graphs.store import GraphStore
from adgraph.pipeline import latest_complete_graph_run
from adgraph.progress import log_handler_context
from adgraph.provenance import verify_environment

parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--plan", default="configs/run_plan.yaml")
parser.add_argument("--graphs", help="graph run directory (default: latest assembled full run in processed_dir)")
parser.add_argument("--out", help="default: <processed_dir>/experiments")
parser.add_argument("--status", action="store_true", help="show progress of every unit and exit")
parser.add_argument("--allow-ephemeral-storage", action="store_true")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
paths = load_paths(args.paths)
graphs = Path(args.graphs) if args.graphs else latest_complete_graph_run(paths["processed_dir"])
out = args.out or f"{paths['processed_dir']}/experiments"
entries = load_plan(args.plan)
pd.set_option("display.width", 250, "display.max_rows", 1000, "display.max_colwidth", 70)
print(f"graph run: {graphs}\nresults:   {out}\nplan:      {args.plan}\n", flush=True)

if args.status:
    store = GraphStore.load(graphs, keys=())                 # progress needs no feature arrays
    st = plan_status(entries, store, out)
    print(st.to_string(index=False))
    print("\n" + st["state"].value_counts().to_string())
    raise SystemExit

verify_environment()
store = GraphStore.load(graphs)
with log_handler_context():
    result = run_plan(entries, store, out, allow_ephemeral=args.allow_ephemeral_storage)
print(result.drop(columns=["updated"]).to_string(index=False))
print("\n" + result["state"].value_counts().to_string())
bad = result[result["state"].isin(["failed", "incompatible"])]
if len(bad):
    raise SystemExit(f"{len(bad)} unit(s) failed or incompatible; see {out}/_plan_status.csv")
