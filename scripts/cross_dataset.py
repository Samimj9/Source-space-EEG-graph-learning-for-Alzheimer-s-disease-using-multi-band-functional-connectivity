"""Command-line entry point of the cross-dataset experiment D3 <-> D2 (docs/cross_dataset/run_guide.md).

    python scripts/cross_dataset.py audit                       # mapping, layout, geometry, EDF headers, D3 stores,
                                                                # inherited decisions (reads only; no graph is built)
    python scripts/cross_dataset.py inspect-d2                  # recording-level QC of D2 (no source model, no graph)
    python scripts/cross_dataset.py authorize --stage build_d2 --by "<name>" [--note ...]   # the AUTHOR, once
    python scripts/cross_dataset.py build-d2 --inverse lcmv [--limit 2]   # matched-channel D2 stores (authorized)
    python scripts/cross_dataset.py validate-d2 --inverse lcmv
    python scripts/cross_dataset.py compat --inverse lcmv --mode B        # compatibility report, shift diagnostics,
                                                                          # harmonization manifest, D2 QC (blind)
    python scripts/cross_dataset.py freeze                      # decision audit + frozen campaign (before training)
    python scripts/cross_dataset.py authorize --stage train --by "<name>"
    python scripts/cross_dataset.py status
    python scripts/cross_dataset.py run --direction D3_to_D2 --inverse lcmv --mode B [--experiments exp3 ...]
    python scripts/cross_dataset.py evaluate                    # target labels are read here only (frozen units)
    python scripts/cross_dataset.py report
    python scripts/cross_dataset.py smoke [--quick]             # synthetic end-to-end, both directions (temp dir)

Every output goes to ``<processed>/CROSS_D3_D2_19CH_v1/``; the release, amendment A2 and the D3 stores are read only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["audit", "inspect-d2", "authorize", "build-d2", "validate-d2", "compat",
                                       "freeze", "status", "run", "evaluate", "report", "smoke"])
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--inverse", default="lcmv", choices=["lcmv", "wmne"])
parser.add_argument("--mode", default="B", choices=["B", "A"])
parser.add_argument("--direction", nargs="*", default=None, choices=["D3_to_D2", "D2_to_D3"])
parser.add_argument("--inverses", nargs="*", default=None, choices=["lcmv", "wmne", "sensor"])
parser.add_argument("--modes", nargs="*", default=None, choices=["B", "A"])
parser.add_argument("--experiments", nargs="*", default=None)
parser.add_argument("--variants", nargs="*", default=None)
parser.add_argument("--stage", default="primary", choices=["primary", "stability"])
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--by", default="")
parser.add_argument("--note", default="")
parser.add_argument("--quick", action="store_true", help="smoke: 2 epochs, one variant per experiment")
parser.add_argument("--force", action="store_true")
parser.add_argument("--allow-ephemeral-storage", action="store_true")
args = parser.parse_args()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 40)

from adgraph.config import load_paths  # noqa: E402
from adgraph.crossdataset import channels, compat, d2store, namespace, plan, transfer  # noqa: E402
from adgraph.checkpoint import atomic_write_csv, atomic_write_json  # noqa: E402


def paths_and_processed():
    p = load_paths(args.paths)
    return p, Path(p["processed_dir"])


def d3_reference_geometry(paths: dict) -> dict:
    """The established D3 geometry (identical in every ds004504 recording, entry 46), read from the .set files."""
    from adgraph.io import bids, chanlocs
    from adgraph.sourcegraphs.config import resolve

    cfg = resolve("lcmv", "lobes12")
    recs = bids.list_recordings(paths["dataset3"]["root"])
    kept, _ = bids.load_participants_tsv(bids.participants_path(paths["dataset3"]["root"]))
    recs = recs[recs["key"].isin(kept["key"])]
    geoms = {chanlocs.geometry_of(p, cfg["channels"]["recorded"], cfg)["hash"]: chanlocs.geometry_of(
        p, cfg["channels"]["recorded"], cfg) for p in recs["path"]}
    if len(geoms) != 1:
        raise SystemExit(f"D3 recordings have {len(geoms)} different electrode geometries; expected one")
    return next(iter(geoms.values()))


def geometry(processed) -> dict:
    f = namespace.root(processed) / "audit" / "reference_geometry.json"
    if not f.exists():
        raise SystemExit("run `scripts/cross_dataset.py audit` first (it records the D3 reference geometry)")
    return json.loads(f.read_text())["d2_geometry"]


def stores_record(processed, c) -> dict:
    """{name: graph_run_id} of every store the campaign reads (verified)."""
    out = {}
    for inv in c["inverses"] + ["sensor"]:
        for mode in c["modes"]:
            from adgraph.graphs.store import graph_run_id

            out[f"D3/{inv}/{mode}"] = graph_run_id(transfer.open_d3(processed, inv, mode, c))
            out[f"D2/{inv}/{mode}"] = graph_run_id(transfer.open_d2(processed, inv, mode, c))
    return out


# ------------------------------------------------------------------------------------------------ smoke
if args.action == "smoke":
    from adgraph.crossdataset import evaluate, report
    from adgraph.crossdataset.synthetic import write_all

    tmp = Path(tempfile.mkdtemp(prefix="cross_smoke_"))
    write_all(tmp, n_windows=4, psd_effect=0.8, conn_effect=0.8)
    c = plan.load_campaign()
    if args.quick:
        from adgraph.models import trainer

        orig = trainer.GNNModel.__init__

        def init(self, model, training, *a, **k):
            orig(self, model, {**training, "epochs": 2}, *a, **k)
        trainer.GNNModel.__init__ = init
        c = {**c, "source_cv": {**c["source_cv"], "D2": {**c["source_cv"]["D2"], "repeats": 1}},
             "evaluation": {**c["evaluation"], "n_bootstrap": 200}}
    frozen = plan.freeze(tmp, c, stores_record(tmp, c), plan.decision_audit(tmp, c), synthetic=True)
    vs = None
    if args.quick:
        vs = [v[0]["variant"] for v in plan.variants(c).values()] + ["logreg__psd"]
    t = transfer.run(tmp, c, "primary", variants=vs, synthetic=True)
    ev = evaluate.evaluate(tmp, c)
    rep = report.report(tmp, c)
    print(t.groupby(["direction", "state"]).size().to_string())
    print(f"cross-dataset smoke run: {int((t['state'] != 'failed').sum())} / {len(t)} units completed; "
          f"{ev['summary']['n_units_evaluated']} evaluated; report {rep['dir']} (synthetic, never results)")
    failed = t[t["state"].str.startswith("failed")]
    if len(failed):
        print(failed.to_string(index=False))
    sys.exit(1 if len(failed) else 0)

P, processed = paths_and_processed()
c = plan.load_campaign()

if args.action == "audit":
    from adgraph.io.fingerprint import InputHashes
    from adgraph.io.layout import find_layout_file
    from adgraph.stores import verify_store

    table = channels.load_table()
    layout = Path(P["electrode_layout"]) if Path(P.get("electrode_layout", "")).is_file() else find_layout_file(
        P["drive_root"])
    reg = d2store.registry(P, hashes=InputHashes(namespace.sub(processed, "stores", "cache", "input_sha256.json")))
    headers = d2store.header_audit(reg)
    ref = d3_reference_geometry(P)
    d2_geom = channels.harmonized_geometry(ref, table)
    df, summary = channels.audit(table, layout, headers, ref)
    out = namespace.sub(processed, "audit")
    atomic_write_csv(out / "channel_mapping_audit.csv", df)
    atomic_write_csv(out / "d2_edf_headers.csv", headers.drop(columns=["raw_names"]))
    atomic_write_json(out / "reference_geometry.json", {"d3_geometry": ref, "d2_geometry": d2_geom})
    stores = []
    for inv in ("lcmv", "wmne"):
        for mode in ("B", "A"):
            for atlas in (["lobes12", "dk68"] if inv == "lcmv" else ["lobes12"]):
                r = verify_store(processed, inv, mode, atlas)
                stores.append({"inverse": inv, "mode": mode, "atlas": atlas, "passed": r["passed"], "dir": r["dir"],
                               "graph_run_id": r["graph_run_id"]})
    dec = plan.decision_audit(processed, c)
    atomic_write_json(out / "audit.json", {"channel_map": summary, "d3_stores": stores, "decision_audit": dec,
                                           "d2_registry": {"n": len(reg), "AD": int(reg["y"].sum()),
                                                           "HC": int((reg["y"] == 0).sum())}})
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=1, default=str))
    print(pd.DataFrame(stores).to_string(index=False))
    print(json.dumps(dec["checks"], indent=1))
    ok = all(s["passed"] for s in stores) and dec["passed"]
    print("AUDIT PASSED" if ok else "AUDIT FAILED")
    sys.exit(0 if ok else 1)

if args.action == "inspect-d2":
    from functools import partial

    from adgraph.io.fingerprint import InputHashes
    from adgraph.sourcegraphs.build import recording_qc

    cfg = d2store.resolve(args.inverse)
    table = channels.load_table()
    reg = d2store.registry(P, hashes=InputHashes(namespace.sub(processed, "stores", "cache", "input_sha256.json")))
    loader = partial(d2store.load_d2, table=table, geometry=geometry(processed))
    rows = []
    for rec in reg.itertuples():
        try:
            row = recording_qc(rec, cfg, loader)[0]
            row.pop("label", None), row.pop("y", None), row.pop("age", None), row.pop("mmse", None)
        except Exception as exc:
            row = {"key": rec.key, "status": f"error: {exc}"}
        rows.append(row)
    df = pd.DataFrame(rows)
    atomic_write_csv(namespace.sub(processed, "audit", "d2_inspection.csv"), df)
    cols = [x for x in ("key", "original_duration_s", "selection_mode", "selected_duration_s", "n_candidate_windows",
                        "status") if x in df]
    print(df[cols].to_string(index=False))
    sys.exit(0)

if args.action == "authorize":
    rec = namespace.authorize(processed, args.stage, args.by, args.note)
    print(f"authorized: {rec}")
    sys.exit(0)

if args.action == "build-d2":
    from adgraph.provenance import verify_environment

    verify_environment()
    try:
        b, a = d2store.run(P, processed, args.inverse, geometry(processed),
                           Path(P["cache_dir"]) / "subjects", args.limit, args.allow_ephemeral_storage)
    except namespace.NotAuthorized as exc:
        sys.exit(str(exc))
    print(f"D2 stores: {b}\n           {a}")
    sys.exit(0)

if args.action == "validate-d2":
    ok = True
    for mode in ("B", "A"):
        r = d2store.validate(d2store.store_dir(processed, args.inverse, mode, args.limit),
                             None if args.limit else d2store.EXPECTED_D2)
        ok &= r["passed"]
        print(mode, "PASSED" if r["passed"] else "FAILED", {k: v["passed"] for k, v in r["checks"].items()})
    sys.exit(0 if ok else 1)

if args.action == "compat":
    from adgraph.qc.report import compute as qc_compute
    from adgraph.sourcegraphs.config import resolve

    table = channels.load_table()
    d3 = transfer.open_d3(processed, args.inverse, args.mode, c)
    d2 = transfer.open_d2(processed, args.inverse, args.mode, c)
    t = compat.compatibility(d3, d2, resolve(args.inverse, c["atlas"]), d2store.resolve(args.inverse), table)
    shift = compat.shift_diagnostics({"D3": d3, "D2": d2})
    audit = json.loads((namespace.root(processed) / "audit" / "audit.json").read_text())
    man = compat.harmonization_manifest(d3, d2, resolve(args.inverse, c["atlas"]), d2store.resolve(args.inverse),
                                        table, t, audit.get("channel_map"))
    out = namespace.sub(processed, "audit", f"{args.inverse}_mode_{args.mode}")
    atomic_write_csv(out / "compatibility_report.csv", t)
    atomic_write_csv(out / "shift_diagnostics.csv", shift)
    atomic_write_json(out / "harmonization_manifest.json", man)
    qc = qc_compute(transfer.blind(d2), out / "d2_qc_blind", sorted(d2.subjects["key"]), "D2 (labels masked)")
    print(t.to_string(index=False))
    print(shift.round(3).to_string(index=False))
    print("D2 participant QC (labels masked):", json.dumps({k: qc["summary"][k] for k in (
        "category_A_participants", "category_B_participants", "multivariate_outliers")}))
    print("COMPATIBILITY PASSED" if compat.passed(t) else "COMPATIBILITY FAILED (unexpected difference)")
    sys.exit(0 if compat.passed(t) else 1)

if args.action == "freeze":
    dec = plan.decision_audit(processed, c)
    try:
        rec = plan.freeze(processed, c, stores_record(processed, c), dec)
    except plan.NotFrozen as exc:
        sys.exit(str(exc))
    print(f"frozen: {rec['n_units_primary']} primary + {rec['n_units_stability']} stability units, campaign "
          f"{rec['campaign_hash']}, commit {rec['git_commit']} -> {plan.frozen_path(processed)}")
    sys.exit(0)

if args.action == "status":
    t = transfer.status(processed, c, args.stage)
    print(t.groupby(["direction", "inverse", "mode", "state"]).size().unstack(fill_value=0).to_string())
    sys.exit(0)

if args.action == "run":
    try:
        t = transfer.run(processed, c, args.stage, args.experiments, args.variants, args.direction, args.inverses,
                         args.modes, force=args.force)
    except (plan.NotFrozen, namespace.NotAuthorized) as exc:
        sys.exit(str(exc))
    print(t.to_string(index=False))
    sys.exit(1 if t["state"].str.startswith("failed").any() else 0)

if args.action == "evaluate":
    from adgraph.crossdataset import evaluate

    try:
        r = evaluate.evaluate(processed, c, args.stage)
    except plan.NotFrozen as exc:
        sys.exit(str(exc))
    print(json.dumps(r["summary"], indent=1, default=str))
    sys.exit(0)

if args.action == "report":
    from adgraph.crossdataset import report

    r = report.report(processed, c, args.stage)
    print(f"report: {r['dir']}")
    sys.exit(0)
