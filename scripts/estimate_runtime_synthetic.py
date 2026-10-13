"""Pre-run cost table for every experiment config, measured on SYNTHETIC data of real size.

    python scripts/estimate_runtime_synthetic.py [--rates audit|measure] [--threads 2] [--out ...]

Uses a synthetic graph run with the real cohort (D1 10 AD / 10 HC, D2 10 AD / 13 HC) and 27
windows per subject (300 s recordings), the committed configs (full protocol: repeats, nested
CV, grids, epochs) and the same estimator as the notebooks (adgraph.experiments.estimate):
exact fit counts x per-fit time per architecture. Per-fit times are either measured now
(``--rates measure``: one full fit per architecture; slow on CPU, mainly EdgeWeightedGAT) or
taken from the pre-real-data audit's synthetic CPU timings (``--rates audit``, default:
seconds per fit at 486 training windows, 60 epochs, 2 threads; methodology log entry 33),
scaled linearly with training windows. Checkpoint writes are not included here
(they depend on the output folder; the notebooks measure them on Drive).

The numbers describe the machine this script runs on (e.g. a 2-vCPU Colab CPU runtime with
--threads 2). Each experiment notebook re-measures on its own runtime before running.
No EEG is read.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from adgraph.experiments.estimate import cost_class, fit_plan, time_fit
from adgraph.experiments.plan import experiment_entries, plan_units
from adgraph.graphs.dataset import _meta
from adgraph.synthetic import make_synthetic_store

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=2, help="torch CPU threads (Colab CPU runtime: 2)")
parser.add_argument("--rates", choices=["audit", "measure"], default="audit")
parser.add_argument("--configs", nargs="+", help="default: every experiment config")
parser.add_argument("--out", default="results/tables/runtime_estimates_synthetic.csv")
args = parser.parse_args()

torch.set_num_threads(args.threads)
configs = args.configs or sorted(str(p) for p in Path("configs/experiments").rglob("*.yaml")
                                 if p.stem not in {"base", "classical_base", "gnn_base"})
store = make_synthetic_store(windows_per_subject=(27, 27), psd_effect=0.5, conn_effect=0.5)
# audit timings (synthetic, 2 CPU threads, 60 epochs): seconds per fit at AUDIT_N training windows
AUDIT_N = 486
AUDIT_SECONDS = {"gnn:gcn:single": 5.0, "gnn:weighted_graphsage:single": 5.0, "gnn:edge_weighted_gat:single": 45.0,
                 "gnn:gcn:mean": 5.0, "gnn:gcn:multi:shared": 14.0, "gnn:gcn:multi:per-band": 22.0,
                 "classical:conn:logreg": 0.5, "classical:conn:svm": 1.1, "classical": 0.1}


def audit_rate(v: dict) -> float:
    if v["input"] == "classical":
        conn = v.get("features", {}).get("source") in ("plv", "ciplv")
        key = f"classical:conn:{v['model']['name']}" if conn else "classical"
    else:
        _, enc, mode, shared, _ = cost_class(v).split(":")
        key = f"gnn:{enc}:{mode}" + (f":{shared}" if mode == "multi" else "")
    return AUDIT_SECONDS[key] / AUDIT_N


rate, rows, metas = {}, [], {}
for config in configs:
    for _, _, v, s in plan_units(experiment_entries(config)):
        cc = cost_class(v)
        key = cc if v["input"] == "graph" else f"{cc}:{config}"
        if key not in rate and args.rates == "audit":
            rate[key] = audit_rate(v)
        elif key not in rate:
            secs, n_win, dev = time_fit(store, v, "pooled" if "pooled" in v["settings"] else s)
            rate[key] = secs / n_win
            print(f"timed {key}: {secs:.1f} s for {n_win} training windows ({dev})", flush=True)
        tkey = v.get("target", "label")
        metas.setdefault(tkey, _meta(store, tkey))
        fits, units = fit_plan(metas[tkey], v, s)
        rows.append({"config": config, "experiment": v["experiment"], "variant": v["variant"], "setting": s,
                     "cost_class": cc, "fits": len(fits), "checkpoint_units": units,
                     "cpu_hours": rate[key] * sum(fits) / 3600})
table = pd.DataFrame(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
table.to_csv(args.out, index=False)
summary = (table.groupby("experiment", sort=False)
           .agg(variants=("variant", "nunique"), fits=("fits", "sum"), units=("checkpoint_units", "sum"),
                cpu_hours=("cpu_hours", "sum")).reset_index())
pd.set_option("display.width", 200)
print(summary.round(1).to_string(index=False))
print(f"\nwritten {args.out} (rates={args.rates}, threads={args.threads}, synthetic data, checkpoint writes excluded)")
