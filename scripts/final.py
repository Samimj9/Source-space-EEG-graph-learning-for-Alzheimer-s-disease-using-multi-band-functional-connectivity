"""The single command-line entry point of the final framework (what every final notebook calls).

    python scripts/final.py stores                                  # verify every store (read-only table)
    python scripts/final.py qc --inverse lcmv --mode B              # participant QC report (computed once)
    python scripts/final.py graphs --inverse lcmv --mode B          # graph-construction diagnostics (development
                                                                    # participants, label-blind)
    python scripts/final.py status --exp exp5 --inverse lcmv --mode B [--graph dense] [--atlas dk68]
    python scripts/final.py run    --exp exp5 --inverse lcmv --mode B [--graph dense] [--atlas dk68]
                                   [--variants plv__separate__mean] [--settings xu_dev_cv]
    python scripts/final.py report [--exp exp5 --inverse lcmv --mode B]   # tables (+ one experiment's diagnostics)
    python scripts/final.py select                                  # graph construction: development-only rule
    python scripts/final.py difficult                               # category C participants (development CV)
    python scripts/final.py parcellation [--bem <bem-sol.fif>]      # source-resolution analysis per atlas
    python scripts/final.py reliability --inverse lcmv --mode B     # split-half reliability of features (development)
    python scripts/final.py smoke [--quick]                         # CPU smoke run of Exp 0-8 on synthetic stores

``--exp`` takes exp0 ... exp8. Exp 0 always runs in sensor space (its results are stored under inverse ``sensor``).
One experiment per ``run`` call; nothing else is started. Outputs: ``<processed>/FINAL_v1/`` (``adgraph.release``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adgraph import release
from adgraph.config import load_paths

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["stores", "qc", "graphs", "status", "run", "report", "select", "difficult",
                                       "parcellation", "reliability", "smoke"])
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--exp", help="exp0 ... exp8")
parser.add_argument("--inverse", default="lcmv", choices=["lcmv", "wmne"])
parser.add_argument("--mode", default="B", choices=["B", "A"])
parser.add_argument("--atlas", default="dk68")
parser.add_argument("--graph", default=None, help="graph construction of configs/final/graphs.yaml (GNNs; default "
                                                  "dense); 'selected' = the recorded development-only decision")
parser.add_argument("--variants", nargs="*", default=None)
parser.add_argument("--settings", nargs="*", default=None, choices=["xu_dev_cv", "xu_holdout"])
parser.add_argument("--bem", default=None, help="parcellation: fsaverage BEM solution (default: the stores' BEM "
                                                "under <cache_dir>/subjects, else a sphere approximation)")
parser.add_argument("--quick", action="store_true", help="smoke: one variant per experiment, development CV only")
parser.add_argument("--allow-ephemeral-storage", action="store_true")
args = parser.parse_args()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 40)


def experiment_config(exp: str) -> Path:
    from adgraph.experiments.config import experiment_configs

    cfgs = experiment_configs()
    if exp not in cfgs:
        sys.exit(f"--exp must be one of {sorted(cfgs)}")
    return cfgs[exp]


def write_json(path: Path, obj) -> Path:
    from adgraph.checkpoint import atomic_write_json

    return atomic_write_json(path, obj)


def processed() -> Path:
    return Path(load_paths(args.paths)["processed_dir"])


if args.action == "smoke":
    # synthetic stores at the release paths of a temporary root; the real verification, lock and QC gate run
    from adgraph import storage
    from adgraph.experiments.config import experiment_configs
    from adgraph.experiments.plan import experiment_variants, run_experiment
    from adgraph.synthetic import write_release_stores

    storage.in_colab = lambda: False                       # the temporary root is deliberately local
    root = Path(tempfile.mkdtemp(prefix="final_smoke_"))
    write_release_stores(root, n_windows=3, psd_effect=0.8, conn_effect=0.8, band_direction=0.5)
    rows = []
    for exp, cfg in experiment_configs().items():
        for inv in ("lcmv", "wmne"):
            if exp == "exp0" and inv == "wmne":
                continue                                   # sensor space: once per mode
            vs = [v["variant"] for v in experiment_variants(cfg, inv, "B")]
            sel = vs[:1] if args.quick else None
            t = run_experiment(cfg, root, inv, "B", variants=sel, settings=["xu_dev_cv"] if args.quick else None)
            rows.append(t)
            print(t[["experiment", "variant", "setting", "inverse", "state", "participant_auc", "seconds",
                     "message"]].to_string(index=False), flush=True)
    res = pd.concat(rows, ignore_index=True)
    bad = res[res["state"] != "completed"]
    print(f"\nsmoke run: {len(res)} units, {len(bad)} not completed; synthetic results (never results) in {root}")
    sys.exit(1 if len(bad) else 0)

P = processed()


def resolve_graph(graph: str | None, exp: str | None) -> str | None:
    """``selected``: the graph construction recorded by ``select`` (FINAL_v1/decisions/graph_construction.json);
    classical experiments (Exp 0, 1) have none."""
    if graph != "selected":
        return graph
    if exp in ("exp0", "exp1"):
        return None
    f = release.root(P) / "decisions" / "graph_construction.json"
    if not f.exists():
        sys.exit(f"no graph-construction decision recorded ({f}); run notebook 03_graph_construction first, or pass "
                 "--graph dense explicitly")
    d = json.loads(f.read_text())
    if d.get("reason", "").startswith("comparison incomplete"):
        sys.exit(f"the graph-construction comparison is incomplete ({d.get('missing')}); finish notebook 03 first")
    print(f"graph construction: {d['selected']} (recorded decision: {d['reason']})")
    return d["selected"]


args.graph = resolve_graph(args.graph, args.exp)

if args.action == "stores":
    from adgraph.stores import overview

    t = overview(P, atlases=(args.atlas,))
    print(t.to_string(index=False))
    sys.exit(0 if t["passed"].all() else 1)

if args.action == "qc":
    from adgraph.qc.report import QCUnresolved, describe, require_qc
    from adgraph.stores import open_store

    store, rec = open_store(P, args.inverse, args.mode, args.atlas)
    try:
        rep = require_qc(store, P, args.inverse, args.mode, args.atlas)
        print(describe(rep))
    except QCUnresolved as exc:
        print(f"QC needs a decision before experiments can run:\n{exc}")
        sys.exit(2)

if args.action == "graphs":
    from adgraph.experiments.config import graph_constructions
    from adgraph.graphs.diagnostics import construction_diagnostics, summarize
    from adgraph.qc.report import development_keys
    from adgraph.stores import open_store

    store, rec = open_store(P, args.inverse, args.mode, args.atlas)
    diag = construction_diagnostics(store, graph_constructions(), development_keys(store))
    out = release.root(P) / "graphs" / f"{args.inverse}_{args.mode}_{args.atlas}" / \
        f"diagnostics_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    diag.to_csv(out / "diagnostics_by_band.csv", index=False)
    s = summarize(diag)
    s.to_csv(out / "diagnostics_summary.csv", index=False)
    write_json(out / "manifest.json", {"store": rec, "release": release.release_manifest(),
                                       "participants": "development only", "labels_used": False})
    print(s.round(3).to_string(index=False))
    print(f"\n-> {out}")

if args.action in ("status", "run"):
    from adgraph.experiments.plan import run_experiment, status_table

    if not args.exp:
        sys.exit("--exp is required")
    cfg = experiment_config(args.exp)
    if args.action == "status":
        print(status_table(cfg, P, args.inverse, args.mode, args.atlas, args.graph).to_string(index=False))
    else:
        from adgraph.provenance import verify_environment

        verify_environment()
        t = run_experiment(cfg, P, args.inverse, args.mode, args.atlas, args.graph, args.variants, args.settings,
                           allow_ephemeral=args.allow_ephemeral_storage)
        print(t.drop(columns=["updated"]).to_string(index=False))
        failed = t[t["state"] != "completed"]
        if len(failed):
            print(f"\n{len(failed)} unit(s) did not complete (recorded above; completed units are saved). Run again "
                  "to resume.")
            sys.exit(1)

if args.action == "report":
    from adgraph.experiments.report import experiment_report, write_tables

    res = write_tables(release.results_dir(P), release.root(P) / "reports")
    print(f"{res['n_results']} completed results -> {res['dir']}")
    if args.exp:
        from adgraph.experiments.config import load_experiment

        name = load_experiment(experiment_config(args.exp))["experiment"]
        inv = "sensor" if args.exp == "exp0" else args.inverse
        ctx = {"inverse": inv, "mode": args.mode, "atlas": args.atlas,
               "graph": "none" if args.exp in ("exp0", "exp1") else (args.graph or "dense")}
        rep = experiment_report(release.results_dir(P), name, ctx)
        print(rep["table"].drop(columns=["dir"], errors="ignore").round(3).to_string(index=False)
              if len(rep["table"]) else "no completed result of this experiment in this context")
        print(f"-> {rep['dir']}")

if args.action == "select":
    from adgraph.experiments.report import collect
    from adgraph.experiments.selection import select

    d = select(collect(release.results_dir(P)))
    out = release.root(P) / "decisions" / "graph_construction.json"
    if out.exists():
        prev = json.loads(out.read_text())
        if prev.get("selected") != d["selected"] and not prev.get("reason", "").startswith("comparison incomplete"):
            sys.exit(f"a different graph-construction decision is already recorded ({prev['selected']}); it is never "
                     f"overwritten (inspect {out})")
    write_json(out, {**d, "decided_at": datetime.now(timezone.utc).isoformat(),
                     "release": release.release_manifest()})
    print(pd.DataFrame(d["summary"]).round(3).to_string(index=False))
    print(f"\nselected: {d['selected']} ({d['reason']})\n-> {out}")

if args.action == "difficult":
    from adgraph.experiments.report import collect, development_only
    from adgraph.qc.difficult import difficult_participants

    t = development_only(collect(release.results_dir(P)))
    res = [(f"{r.inverse}/{r.mode}/{r.atlas}/{r.graph}/{r.experiment}/{r.variant}",
            pd.read_csv(Path(r.dir) / "predictions_participants.csv"), float(r.threshold)) for r in t.itertuples()]
    d = difficult_participants(res)
    out = release.root(P) / "qc" / f"difficult_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(d.head(30).round(3).to_string(index=False))
    print(f"\n{int(d['difficult'].sum())} difficult participants (kept; never excluded) -> {out}")

if args.action == "parcellation":
    import yaml

    from adgraph.source.inverse import fetch_fsaverage
    from adgraph.source.resolution import analyse, recommend_atlas
    from adgraph.sourcegraphs.config import resolve

    paths = load_paths(args.paths)
    subjects = Path(paths["cache_dir"]) / "subjects"
    fs = fetch_fsaverage(subjects)
    bem = args.bem or str(fs / "bem" / resolve("lcmv")["source"]["bem"])
    if not Path(bem).exists():
        bem = None
    table, rank = analyse(fs.parent, {"lcmv": resolve("lcmv"), "wmne": resolve("wmne")}, bem=bem)
    rule = yaml.safe_load(Path("configs/final/selection.yaml").read_text())["atlas_resolution"]
    rec = recommend_atlas(table, rule)
    out = release.root(P) / "parcellation" / f"resolution_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "resolution_by_atlas.csv", index=False)
    write_json(out / "spatial_rank.json", rank)
    write_json(out / "recommendation.json", {**rec, "forward_model": rank["forward_model"],
                                             "release": release.release_manifest()})
    print({k: v for k, v in rank.items() if k != "singular_values_rel"})
    print(table.round(3).to_string(index=False))
    print(f"\nrecommended atlas: {rec['recommended']} ({rank['forward_model']})\n-> {out}")

if args.action == "reliability":
    from adgraph.qc.report import development_keys
    from adgraph.source.resolution import operator_ranks, store_reliability
    from adgraph.stores import open_store

    store, rec = open_store(P, args.inverse, args.mode, args.atlas)
    dev = development_keys(store)
    rel = store_reliability(store, dev)
    ops = operator_ranks(store.root, dev) if args.inverse == "lcmv" else pd.DataFrame()
    out = release.root(P) / "parcellation" / f"reliability_{args.inverse}_{args.mode}_{args.atlas}_" \
                                             f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    rel.to_csv(out / "split_half_reliability.csv", index=False)
    if len(ops):
        ops.to_csv(out / "operator_ranks.csv", index=False)
        print(f"ROI operator rank: {ops['rank'].value_counts().to_dict()} (participants)")
    print(rel.round(3).to_string(index=False))
    print(f"-> {out}")
