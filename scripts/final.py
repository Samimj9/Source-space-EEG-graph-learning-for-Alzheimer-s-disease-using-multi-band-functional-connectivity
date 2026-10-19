"""The single command-line entry point of the final framework (what every final notebook calls).

    python scripts/final.py stores                                  # verify every store (read-only table)
    python scripts/final.py qc --inverse lcmv --mode B              # participant QC report (computed once)
    python scripts/final.py graphs --inverse lcmv --mode B          # graph-construction diagnostics (development
                                                                    # participants, label-blind)
    python scripts/final.py status --exp exp5 --inverse lcmv --mode B [--graph dense] [--atlas dk68]
    python scripts/final.py run    --exp exp5 --inverse lcmv --mode B [--graph dense] [--atlas dk68]
                                   [--variants plv__separate__mean]  # development CV only (the holdout: see below)
    python scripts/final.py report [--exp exp5 --inverse lcmv --mode B]   # tables (+ one experiment's diagnostics)
    python scripts/final.py select                                  # graph construction: development-only rule, on
                                                                    # the recorded atlas
    python scripts/final.py atlas-check --atlas r18                 # anatomical checks of an atlas (no EEG)
    python scripts/final.py store-resolution --inverse lcmv --atlas r18   # resolution of the stores' own operators
    python scripts/final.py roi-compare                             # dk68 vs r18 (development CV; notebook 04)
    python scripts/final.py choose-atlas --atlas r18 --reason "..." # the AUTHOR records the analysis atlas (once)
    python scripts/final.py freeze                                  # freeze the final analysis (configs/final/
                                                                    # holdout.yaml) once development CV is complete
    python scripts/final.py holdout --exp exp5                      # holdout of the frozen units only (descriptive)
    python scripts/final.py difficult                               # category C participants (development CV)
    python scripts/final.py parcellation [--bem <bem-sol.fif>]      # source-resolution analysis per atlas
    python scripts/final.py reliability --inverse lcmv --mode B     # split-half reliability of features (development)
    python scripts/final.py reconstruct-check --inverse lcmv --mode B   # can coarse ROI signals be rebuilt from the
                                                                    # store's saved operators + ROI signals? (read-only)
    python scripts/final.py smoke [--quick]                         # CPU smoke run of Exp 0-8 on synthetic stores

``--exp`` takes exp0 ... exp8. Exp 0 always runs in sensor space (its results are stored under inverse ``sensor``).
``--atlas selected`` / ``--graph selected`` read the recorded decisions (``experiments.decisions``). ``run`` and
``status`` cover development CV only; the holdout is evaluated only by ``holdout``, for the units of the frozen
final analysis. One experiment per ``run`` call; nothing else is started. Outputs: ``<processed>/FINAL_v1/``
(``adgraph.release``); coarser-atlas stores are built by ``scripts/build_stores.py`` next to the 68-ROI stores.
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
                                       "parcellation", "reliability", "reconstruct-check", "smoke", "atlas-check",
                                       "store-resolution", "roi-compare", "choose-atlas", "freeze", "holdout"])
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--exp", help="exp0 ... exp8")
parser.add_argument("--inverse", default="lcmv", choices=["lcmv", "wmne"])
parser.add_argument("--mode", default="B", choices=["B", "A"])
parser.add_argument("--atlas", default="dk68", help="dk68 | r18 | ... | selected (the recorded decision)")
parser.add_argument("--reason", default=None, help="choose-atlas: the scientific reason (recorded)")
parser.add_argument("--graph", default=None, help="graph construction of configs/final/graphs.yaml (GNNs; default "
                                                  "dense); 'selected' = the recorded development-only decision")
parser.add_argument("--variants", nargs="*", default=None)
parser.add_argument("--settings", nargs="*", default=None, choices=["xu_dev_cv"],
                    help="run / status: development CV only (the default); the holdout: action holdout")
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
    # r18 stores (matched to the dk68 ones) and the dk68 vs r18 comparison of notebook 04, development CV only
    from adgraph.experiments import roi_comparison as rc
    from adgraph.experiments.report import collect

    write_release_stores(root, n_windows=3, psd_effect=0.8, conn_effect=0.8, band_direction=0.5, atlas="r18")
    cfg = experiment_configs()
    for r in rc.runs():                                    # dk68 units run above are resumed, not recomputed
        t = run_experiment(cfg[r["exp"]], root, r["inverse"], "B", r["atlas"],
                           None if r["graph"] == "none" else r["graph"], [r["variant"]], ["xu_dev_cv"])
        rows.append(t)
    cls, missing = rc.classification(collect(release.results_dir(root)))
    summ = rc.summary(cls, missing)
    print(cls[["experiment", "variant", "inverse", "auc_dk68", "auc_r18", "auc_diff_r18_minus_dk68",
               "auc_diff_boot_low", "auc_diff_boot_high"]].round(3).to_string(index=False) if len(cls) else missing)
    print(f"ROI comparison (synthetic): {summ['statement']}")
    res = pd.concat(rows, ignore_index=True)
    bad = res[res["state"] != "completed"]
    print(f"\nsmoke run: {len(res)} units, {len(bad)} not completed; ROI comparison complete: {summ['complete']}; "
          f"synthetic results (never results) in {root}")
    sys.exit(1 if len(bad) or not summ["complete"] else 0)

P = processed()


from adgraph.experiments import decisions  # noqa: E402


def resolve_atlas(atlas: str, exp: str | None) -> str:
    """``selected``: the atlas the author recorded after notebook 04 (Exp 0 is sensor space: dk68)."""
    if atlas != "selected":
        return atlas
    if exp == "exp0":
        return "dk68"
    try:
        a = decisions.selected_atlas(P)
    except decisions.DecisionMissing as exc:
        sys.exit(str(exc))
    print(f"atlas: {a} (recorded decision)")
    return a


def resolve_graph(graph: str | None, exp: str | None, atlas: str) -> str | None:
    """``selected``: the graph construction recorded by ``select`` on this atlas; Exp 0, 1 have none."""
    if graph != "selected":
        return graph
    if exp in ("exp0", "exp1"):
        return None
    try:
        d = decisions.graph_decision(P, atlas)
    except decisions.DecisionMissing as exc:
        sys.exit(str(exc))
    print(f"graph construction: {d['selected']} (recorded decision on {atlas}: {d['reason']})")
    return d["selected"]


args.atlas = resolve_atlas(args.atlas, args.exp)
args.graph = resolve_graph(args.graph, args.exp, args.atlas)
dev_only = ["xu_dev_cv"]

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
        print(status_table(cfg, P, args.inverse, args.mode, args.atlas, args.graph, settings=dev_only)
              .to_string(index=False))
    else:
        from adgraph.provenance import verify_environment

        verify_environment()
        t = run_experiment(cfg, P, args.inverse, args.mode, args.atlas, args.graph, args.variants, dev_only,
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

    d = select(collect(release.results_dir(P)), atlas=args.atlas)
    out = decisions.graph_path(P)
    if out.exists():
        prev = json.loads(out.read_text())
        if prev.get("selected") != d["selected"] and not prev.get("reason", "").startswith("comparison incomplete"):
            sys.exit(f"a different graph-construction decision is already recorded ({prev['selected']}); it is never "
                     f"overwritten (inspect {out})")
    write_json(out, {**d, "decided_at": datetime.now(timezone.utc).isoformat(),
                     "release": release.release_manifest()})
    print(pd.DataFrame(d["summary"]).round(3).to_string(index=False))
    print(f"\nselected on atlas {args.atlas}: {d['selected']} ({d['reason']})\n-> {out}\n"
          "(a pre-specified exploratory decision rule on development CV, not a test of statistical superiority)")

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
    from adgraph.source.resolution import approximate_bem

    bem = args.bem or str(fs / "bem" / resolve("lcmv")["source"]["bem"])
    if not Path(bem).exists():                          # the stores' BEM missing: realistic-surface approximation
        bem = str(approximate_bem(Path(paths["cache_dir"]) / "fsaverage_approx3-bem-sol.fif"))
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
    ops, prov = operator_ranks(store.root, dev) if args.inverse == "lcmv" else (pd.DataFrame(), None)
    out = release.root(P) / "parcellation" / f"reliability_{args.inverse}_{args.mode}_{args.atlas}_" \
                                             f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    rel.to_csv(out / "split_half_reliability.csv", index=False)
    if len(ops):
        ops.to_csv(out / "operator_ranks.csv", index=False)
        write_json(out / "operator_checkpoints.json", prov)
        print(f"ROI operator rank: {ops['rank'].value_counts().to_dict()} ({len(ops)} participants; checkpoints: "
              f"{prov['relation']}, verified)")
    print(rel.round(3).to_string(index=False))
    print(f"-> {out}")

if args.action == "reconstruct-check":
    from adgraph.qc.report import development_keys
    from adgraph.source.resolution import reconstruction_check
    from adgraph.sourcegraphs import recording
    from adgraph.sourcegraphs.config import resolve
    from adgraph.stores import open_store

    store, rec = open_store(P, args.inverse, args.mode, args.atlas)
    cfg = resolve(args.inverse, args.atlas)
    win = int(round(cfg["windowing"]["window_s"] * cfg["preprocessing"]["sfreq"]))
    t, prov = reconstruction_check(store.root, development_keys(store), recording.margin_samples(cfg), win)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out = release.root(P) / "parcellation" / f"reconstruction_{args.inverse}_{args.mode}_{args.atlas}_{stamp}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    write_json(out.with_suffix(".json"), prov)
    print(t.round(6).to_string(index=False))
    print(f"\n{len(t)} participants (checkpoints: {prov['relation']}, verified): operator rank "
          f"{sorted(set(t['rank']))}, max constant response {t['constant_response'].max():.1e}, max relative residual "
          f"{t['rel_residual'].max():.1e}; store windows reconstructable exactly from the saved pieces: "
          f"{t['windows_reconstructable'].sum()} / {t['n_windows'].sum()} "
          f"({t['windows_reconstructable'].sum() / max(t['n_windows'].sum(), 1):.1%})")
    if prov["covariance_caveat"]:
        print(f"LCMV covariance caveat (methodology log entry 53): {sorted(prov['covariance_caveat'])}")
    print(f"-> {out}")
    if prov["missing"]:
        sys.exit(f"INCOMPLETE: no checkpoint for {prov['missing']}")

if args.action == "atlas-check":
    from adgraph.source.atlas import verify_atlas

    checks, rows = verify_atlas(args.atlas)
    t = pd.DataFrame(rows)
    out = release.root(P) / "parcellation" / f"atlas_{args.atlas}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "regions.csv", index=False)
    write_json(out / "checks.json", checks)
    print(t.to_string(index=False))
    for name, c in checks.items():
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name}" + (f": {c['detail']}" if c["detail"] else ""))
    print(f"-> {out}")
    sys.exit(0 if all(c["passed"] for c in checks.values()) else 1)

if args.action == "store-resolution":
    from adgraph.qc.report import development_keys
    from adgraph.source.inverse import fetch_fsaverage
    from adgraph.source.resolution import store_resolution
    from adgraph.stores import open_store

    store, rec = open_store(P, args.inverse, "B", args.atlas)
    fs = fetch_fsaverage(Path(load_paths(args.paths)["cache_dir"]) / "subjects")
    t, desc = store_resolution(store.root, fs.parent, args.atlas, development_keys(store))
    out = release.root(P) / "parcellation" / f"store_resolution_{args.inverse}_{args.atlas}_" \
                                             f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "per_participant.csv", index=False)
    write_json(out / "description.json", {**desc, "release": release.release_manifest(), "labels_used": False})
    cols = ["operator_rank", "redundant_signals", "crosstalk_fraction_median", "rois_dominated_by_own_region",
            "leakage_corr_abs_median", "participation_ratio"]
    cp = desc["checkpoints"]
    print({k: v for k, v in desc.items() if k != "checkpoints"})
    print(f"checkpoints: {cp['checkpoint_dir']} ({cp['relation']}); verified for {cp['n_verified']} participants")
    if cp["covariance_caveat"]:
        print(f"LCMV covariance caveat (methodology log entry 53): {sorted(cp['covariance_caveat'])}")
    print(t[cols].describe().loc[["min", "50%", "max"]].round(3).to_string())
    print(f"-> {out}")
    if cp["missing"]:
        sys.exit(f"INCOMPLETE: no checkpoint for {cp['missing']}")

if args.action == "roi-compare":
    from adgraph.checkpoint import atomic_write_csv, atomic_write_text
    from adgraph.experiments import roi_comparison as rc
    from adgraph.experiments.report import collect

    cfg = rc.config()
    cls, missing = rc.classification(collect(release.results_dir(P)), cfg)
    summ = rc.summary(cls, missing, cfg)
    ev = rc.source_evidence(release.root(P), cfg)
    out = release.root(P) / "roi_comparison" / f"compare_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    atomic_write_csv(out / "classification.csv", cls)
    for name, t in ev.items():
        atomic_write_csv(out / f"{name}.csv", t)
    write_json(out / "summary.json", {**summ, "config": cfg, "release": release.release_manifest(),
                                      "holdout": "not evaluated"})
    atomic_write_text(out / "report.md", rc.report_markdown(cls, summ, ev, cfg))
    for name, t in ev.items():
        print(f"==== {name}\n{t.round(3).to_string(index=False)}")
    if len(cls):
        print(cls.drop(columns=["graph"]).round(3).to_string(index=False))
    print(f"\n{summ['statement']}\n-> {out}")
    sys.exit(0 if summ["complete"] else 1)

if args.action == "choose-atlas":
    from adgraph.stores import verify_store

    if args.atlas == "selected" or not args.reason:
        sys.exit("pass --atlas <dk68|r18> and --reason '<the scientific reason>'")
    bad = [f"{inv}/{m}" for inv in ("lcmv", "wmne") for m in ("B", "A")
           if not verify_store(P, inv, m, args.atlas)["passed"]]
    if bad:
        sys.exit(f"the {args.atlas} stores do not all verify ({bad}); an atlas needs validated, matched stores")
    latest = sorted((release.root(P) / "roi_comparison").glob("compare_*"))
    d = decisions.record_atlas(P, args.atlas, args.reason, str(latest[-1]) if latest else None)
    print(f"recorded: atlas {d['atlas']} ({d['reason']}) -> {decisions.atlas_path(P)}")

if args.action == "freeze":
    from adgraph.experiments.report import collect
    from adgraph.provenance import git_commit

    try:
        d = decisions.freeze(P, collect(release.results_dir(P)), git_commit())
    except decisions.DecisionMissing as exc:
        sys.exit(str(exc))
    print(f"final analysis frozen: {len(d['units'])} units on atlas {d['atlas']}, graph {d['graph']}, commit "
          f"{d['git_commit']} -> {decisions.final_path(P)}")

if args.action == "holdout":
    from adgraph.experiments.plan import run_experiment
    from adgraph.provenance import verify_environment

    if not args.exp:
        sys.exit("--exp is required")
    verify_environment()
    f = decisions.final_path(P)
    if not f.exists():
        sys.exit("nothing is frozen: run `python scripts/final.py freeze` first (notebook 99)")
    units = [u for u in json.loads(f.read_text())["units"] if u["exp"] == args.exp]
    if not units:
        sys.exit(f"{args.exp} is not part of the frozen final analysis (configs/final/holdout.yaml)")
    cfg = experiment_config(args.exp)
    failed = 0
    for (inv, mode, atlas, graph), g in pd.DataFrame(units).groupby(["inverse", "mode", "atlas", "graph"], sort=False):
        t = run_experiment(cfg, P, "lcmv" if inv == "sensor" else inv, mode, atlas,
                           None if graph == "none" else graph, sorted(g["variant"]), ["xu_holdout"],
                           allow_ephemeral=args.allow_ephemeral_storage)
        print(t.drop(columns=["updated"]).to_string(index=False))
        failed += int((t["state"] != "completed").sum())
    print("holdout results are descriptive only (release.HOLDOUT_POLICY)")
    sys.exit(1 if failed else 0)
