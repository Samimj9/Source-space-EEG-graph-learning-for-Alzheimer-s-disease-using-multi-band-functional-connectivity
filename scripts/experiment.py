"""Run ONE experiment config end to end (used by the per-experiment Colab notebooks).

    python scripts/experiment.py estimate --config configs/experiments/exp0_sensor_psd.yaml
    python scripts/experiment.py run      --config configs/experiments/exp0_sensor_psd.yaml
    python scripts/experiment.py status   --config configs/experiments/exp0_sensor_psd.yaml
    python scripts/experiment.py report   --config configs/experiments/exp0_sensor_psd.yaml

* estimate: exact number of model fits of this experiment's protocol, a real fit timed per
  architecture on this machine, the checkpoint write cost on the output folder, and the
  remaining time (completed folds subtracted). Saved under <results>/<experiment>/_estimates/.
* run: every variant and setting of this config only (within-dataset settings first, then
  pooled, then cross-dataset if the DESIGN.md §7 gate passes), with the config's full
  protocol (repeats, nested CV, grids). Resumable; a failing unit is recorded in
  <results>/<experiment>/_run_status.csv and the others continue. Never starts another experiment.
* status: progress per (variant, setting) from the checkpoints; computes nothing.
* report: results table + single-experiment diagnostic figures from saved outputs, in
  <results>/<experiment>/_reports/report_<time>/.

Paths come from configs/paths.yaml (graphs: latest assembled full graph run in processed_dir;
results: <processed_dir>/experiments). Nothing needs to be edited.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adgraph.checkpoint import atomic_write_csv, atomic_write_json
from adgraph.config import load_paths
from adgraph.experiments.config import load_experiment
from adgraph.experiments.plan import experiment_entries, plan_status, plan_units, run_plan
from adgraph.graphs.store import GraphStore
from adgraph.pipeline import latest_complete_graph_run
from adgraph.progress import log_handler_context
from adgraph.provenance import verify_environment
from adgraph.storage import check_persistent

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["estimate", "run", "status", "report"])
parser.add_argument("--config", required=True)
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--graphs", help="graph run (default: latest assembled full run in processed_dir)")
parser.add_argument("--out", help="results root (default: <processed_dir>/experiments, D3: experiments_D3)")
parser.add_argument("--allow-ephemeral-storage", action="store_true")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 60)
paths = load_paths(args.paths)
cfg = load_experiment(args.config)
# graph family: D1/D2 (default) or Dataset 3 (`graph_family: D3`: graph runs graphsD3_v*, results in
# experiments_D3/), never mixed: every results root holds the results of one graph run
from adgraph.experiments.config import FAMILIES

prefix, results_root, _ = FAMILIES[cfg.get("graph_family")]
graphs = Path(args.graphs) if args.graphs else latest_complete_graph_run(paths["processed_dir"], prefix=prefix)
out = Path(args.out or f"{paths['processed_dir']}/{results_root}")
if cfg.get("graph_family") in ("WX", "WX10", "D3H"):   # wMNE comparison (entry 55): the LCMV protocol, verified
    from adgraph.lcmvxu.protocol import reference_store, verify as verify_protocol

    tenfold = cfg["graph_family"] == "WX10"
    if cfg["graph_family"] in ("WX", "WX10"):
        from adgraph.lcmvxu.match import require_matched
        from adgraph.lcmvxu.validate import require_validated

        require_validated(graphs)
        require_matched(graphs, reference_store(paths["processed_dir"], tenfold))
    proto = verify_protocol(graphs, paths["processed_dir"], cfg["evaluation"], tenfold)
    print(f"frozen LCMV / Xu protocol verified against {Path(proto['reference']).name}: {proto['n_participants']} "
          f"participants, {proto['n_development']} development / {proto['n_holdout']} holdout, identical "
          f"{len(proto['dev_folds'])} development folds")
if cfg.get("graph_family") in ("LX", "LX10"):    # the frozen participant split must equal the one saved at build
    from adgraph.evaluation.splits import xu_partition
    from adgraph.lcmvxu.validate import require_validated

    require_validated(graphs)                    # only stores whose full validation passed (entry 53)

    saved = json.loads((graphs / "xu_split.json").read_text())
    subj = pd.read_csv(graphs / "subjects.csv")
    dev, test = xu_partition(subj.loc[subj["status"] == "ok", ["key", "dataset", "y"]])
    if (dev, test) != (saved["development"], saved["test"]):
        raise SystemExit(f"participant split differs from {graphs / 'xu_split.json'}; refusing to run")
    print(f"frozen participant split: {len(dev)} development, {len(test)} holdout test participants "
          f"(saved in {graphs / 'xu_split.json'})")
experiment = cfg["experiment"]
entries = experiment_entries(args.config)
exp_dir = out / experiment
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
print(f"experiment: {experiment}\nconfig:     {args.config}\ngraph run:  {graphs}\nresults:    {exp_dir}\n", flush=True)

if args.action == "status":
    st = plan_status(entries, GraphStore.load(graphs, keys=()), out)
    print(st.to_string(index=False))
    print("\n" + st["state"].value_counts().to_string())

elif args.action == "estimate":
    from adgraph.experiments.estimate import by_variant, estimate_experiment

    store_meta = GraphStore.load(graphs, keys=())
    if not store_meta.manifest.get("synthetic"):
        check_persistent({"results": out}, allow_ephemeral=args.allow_ephemeral_storage)
    exp_dir.mkdir(parents=True, exist_ok=True)
    units = plan_units(entries)
    variants = list({u[2]["name"]: u[2] for u in units}.values())
    settings = {}
    for _, _, v, s in units:
        settings.setdefault(v["name"], []).append(s)
    table, info = estimate_experiment(store_meta, variants, lambda v: settings[v["name"]], out,
                                      timing_store=GraphStore.load(graphs))
    est_dir = exp_dir / "_estimates"
    atomic_write_csv(est_dir / f"estimate_{stamp}.csv", table)
    atomic_write_json(est_dir / f"estimate_{stamp}.json", {**info, "graph_run": str(graphs), "config": args.config})
    print(f"measured on: {', '.join(info['devices'])}; checkpoint write: {info['checkpoint_seconds_per_unit']:.2f} s/unit")
    print("\nper variant (all settings):")
    print(by_variant(table).round(2).to_string(index=False))
    print("\nper variant and setting:")
    print(table.round(2).to_string(index=False))
    fmt = lambda h: f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"  # noqa: E731
    print(f"\nESTIMATED TOTAL: {fmt(info['total_hours'])}   REMAINING: {fmt(info['remaining_hours'])} "
          f"(approximate, for this runtime type: {', '.join(info['devices'])})")

elif args.action == "run":
    store = GraphStore.load(graphs)
    if not store.manifest.get("synthetic"):
        verify_environment()
    with log_handler_context():                    # log lines print above the progress bar
        result = run_plan(entries, store, out, allow_ephemeral=args.allow_ephemeral_storage,
                          status_path=exp_dir / "_run_status.csv")
    print(result.drop(columns=["updated", "order"]).to_string(index=False))
    print("\n" + result["state"].value_counts().to_string())
    bad = result[result["state"].isin(["failed", "incompatible"])]
    if len(bad):
        raise SystemExit(f"{len(bad)} unit(s) failed or incompatible; see {exp_dir}/_run_status.csv")
    print(f"\n{experiment}: all units completed or skipped. No other experiment is started.")

elif args.action == "report":
    from adgraph.experiments.report import experiment_report

    progress = plan_status(entries, GraphStore.load(graphs, keys=()), out)
    with log_handler_context():
        rep = experiment_report(out, experiment, progress)
    print(f"report: {rep['report_dir']}")
    if rep["table"].empty:
        print("no completed results yet")
    else:
        cols = [c for c in ["variant", "setting", "n_subjects", "n_repeats", "balanced_accuracy_mean",
                            "balanced_accuracy_sd", "roc_auc_mean", "roc_auc_sd", "roc_auc_ci95"] if c in rep["table"]]
        print(rep["table"][cols].round(3).to_string(index=False))
    print(f"figures: {', '.join(rep['figures']) or 'none'}")
    if rep["skipped"]:
        print(f"skipped figures: {rep['skipped']}")
