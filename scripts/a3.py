"""Command-line entry point of amendment A3 (final-analysis integration; methodology log entry 61).

    python scripts/a3.py plan [--no-drive]     the units that would be frozen (dry run; --no-drive: from configs only)
    python scripts/a3.py audit                 read-only reproducibility audit of pins, results, environments
    python scripts/a3.py freeze                dry run of the dual freeze: checks + inventory, writes NOTHING
    python scripts/a3.py freeze --write        write the two frozen records and the master (needs approved A3 record)
    python scripts/a3.py verify                does the frozen analysis still verify? (read only)
    python scripts/a3.py holdout-status        completed / partial / pending holdout units (read only)
    python scripts/a3.py holdout --part release|A2 --exp expN --python <interpreter> [--repo . --base /content]
                                               holdout of one experiment, at that part's pin, in a worktree
    python scripts/a3.py report                consolidated development report (tables, figures, comparisons)
    python scripts/a3.py holdout-report        descriptive holdout report (needs a verifying freeze)
    python scripts/a3.py d2-manifest           D3 -> D2 model manifest and recommended source-only set (read only)
    python scripts/a3.py smoke [--no-link]     synthetic end-to-end freeze / holdout / report check (temporary root)
    python scripts/a3.py single-verify         the single selected model (a3_single_model.yaml): membership + checks, read only
    python scripts/a3.py single-holdout --python <interpreter> [--repo . --base /content]   holdout of ONLY that model
    python scripts/a3.py single-report         report of that one model (does not need the other frozen units)
    python scripts/a3.py probe                 which write-once method does this file system allow? (own temp folder)

Outputs: <processed>/FINAL_v1/amendments/A3_final/ (never overwritten). The freeze additionally creates the two frozen
records, once. Nothing here trains in the development setting or touches a result, pin, QC decision or store.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["plan", "audit", "freeze", "verify", "holdout-status", "holdout", "report",
                                       "holdout-report", "d2-manifest", "smoke", "probe", "single-verify", "single-holdout",
                                       "single-report"])
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--no-drive", action="store_true", help="plan: configuration only (atlas / graph from A3)")
parser.add_argument("--no-link", action="store_true", help="smoke: simulate a file system without hard links")
parser.add_argument("--write", action="store_true", help="freeze: actually write the records (default: dry run)")
parser.add_argument("--part", choices=["release", "A2"])
parser.add_argument("--exp")
parser.add_argument("--python", default=sys.executable, help="holdout: interpreter with the restored environment")
parser.add_argument("--repo", default=".", help="holdout: the A3 checkout (worktrees of the pins are made from it)")
parser.add_argument("--base", default="/content", help="holdout: where the pin worktrees are created")
parser.add_argument("--out", default=None, help="plan: also write the inventory CSV / Markdown here (new files)")
args = parser.parse_args()
pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 40)


def processed() -> Path:
    from adgraph.config import load_paths

    return Path(load_paths(args.paths)["processed_dir"])


if args.action == "smoke":
    from adgraph.a3 import smoke

    res = smoke.run(link=not args.no_link)
    print(res["summary"])
    sys.exit(0 if res["ok"] else 1)

if args.action == "plan" and args.no_drive:
    from adgraph.a3 import freeze
    from adgraph.a3 import namespace as ns

    rec = ns.record()
    units = freeze.planned_units_for(rec["recorded_atlas"], rec["recorded_graph"])
    inv = freeze.inventory_table(units, [])
    print(inv.drop(columns=["dev_auc", "dev_ci_low", "dev_ci_high", "dev_n_participants", "dev_result"]).to_string(
        index=False))
    print(inv.groupby(["part", "exp"]).size().to_string())
    print(f"total: {len(inv)} units (release part {len(units['release'])}, A2 part {len(units['A2'])})")
    if args.out:
        out = Path(args.out)
        ns.new_file(out.with_suffix(".csv"), inv[["part", "pin", "exp", "experiment", "variant", "inverse", "mode",
                                                    "atlas", "graph"]].to_csv(index=False))
    sys.exit(0)

P = processed()

if args.action in ("plan", "freeze"):
    from adgraph.a3 import freeze

    try:
        res = freeze.freeze(P, write=args.write and args.action == "freeze")
    except (freeze.FreezeRefused, freeze.FreezeConflict) as exc:
        sys.exit(f"REFUSED: {exc}")
    for c in res["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['check']}" + (f": {c['detail']}" if c["detail"] else ""))
    inv = res["inventory"]
    if len(inv):
        print(inv.groupby(["part", "exp"]).agg(units=("variant", "size"), dev_auc_min=("dev_auc", "min"),
                                               dev_auc_max=("dev_auc", "max")).round(3).to_string())
        print(f"{len(inv)} units would be frozen")
    if res["dry_run"]:
        print("DRY RUN: nothing was written." + ("" if res["ok"] else " Preconditions failed."))
        sys.exit(0 if res["ok"] else 1)
    print("written:", *[f"{w}  [{res['write_methods'].get(w)}]" for w in res["written"]], sep="\n  ")

if args.action == "probe":
    from adgraph.a3 import probe
    from adgraph.a3 import namespace as ns

    res = probe.probe(ns.root(P))
    for k, v in res.items():
        print(f"{k}: {v}")
    print("the freeze will use: " + ("hard link (atomic)" if res.get("method") == "link" else
                                     "claim file + rename (hard links unavailable here)"))
    sys.exit(0 if res["ok"] else 1)

if args.action in ("single-verify", "single-holdout", "single-report"):
    from adgraph.a3 import freeze, single

    sel = single.selection()
    u = sel["unit"]
    try:
        if args.action == "single-verify":
            ver = single.verify_light(P, sel)
            rank = single.gnn_ranking(P, sel)
            print(f"selected: {u['experiment']} / {u['variant']} / {u['inverse']} / mode {u['mode']} / {u['atlas']} / "
                  f"{u['graph']}\nfrozen A3 inventory: member of the A2 record ({ver['n_frozen_units']} frozen units); "
                  f"development AUC {ver['entry']['auc']:.4f} (reported {sel['development_auc_reported']}); rank "
                  f"{rank['rank_by_development_auc']} of {rank['n_frozen_gnn_units']} frozen GNN units"
                  + ("" if rank["is_highest"] else f"  !! NOT the highest: {rank['highest']}"))
            print(f"development result verified (marker, checksums, A2 pin); development environment: "
                  f"{ver['dev_libraries']}\nholdout state: {single.holdout_state(P, sel)}\n"
                  f"the other {ver['n_frozen_units'] - 1} frozen units would NOT be evaluated")
        elif args.action == "single-holdout":
            man = single.run_single(P, args.repo, Path(args.paths).resolve(), args.python, args.base)
            print(f"holdout {man['state_before']} -> {man['state_after']} (exit {man['exit_code']}) at pin "
                  f"{man['pin'][:10]}; environment {man['environment']['level']}; descriptive only")
            sys.exit(man["exit_code"] or (0 if man["state_after"] == "completed" else 1))
        else:
            print(single.report(P, sel))
    except (single.SingleRefused, freeze.FreezeInvalid) as exc:
        sys.exit(f"REFUSED: {exc}")
    sys.exit(0)

if args.action == "audit":
    from adgraph.a3 import audit

    res = audit.audit(P)
    for c in res["checks"]:
        tag = "WARN" if c.get("warning") else "PASS" if c["ok"] else "FAIL"
        print(f"[{tag}] {c['check']}" + (f": {c['detail']}" if c["detail"] else ""))
    env = res["environments"]
    print(f"numerical environments: {env['results']}; {env['n_in_final_plan']} of {res['n_completed']} results belong "
          "to the final plan")
    for r in env["summary"]:
        print(f"  {r['namespace']:8s} {r['family']:6s} {r['env_id']:40s} {r['device_kind']:7s} results {r['n_results']:3d} "
              f"(in final plan {r['n_in_final_plan']:3d}) contexts {r['contexts']} modes {r['modes']}")
    print(f"{res['n_completed']} completed development results, {res['n_not_completed']} not completed -> "
          f"{res.get('file')}")
    sys.exit(0 if res["ok"] else 1)

if args.action == "verify":
    from adgraph.a3 import freeze

    try:
        res = freeze.verify(P)
    except freeze.FreezeInvalid as exc:
        sys.exit(f"NOT VERIFIED: {exc}")
    m = res["master"]
    print(f"A3 freeze verifies: {m['n_units']} units, release pin {m['release_pin'][:10]}, A2 pin {m['a2_pin'][:10]}, "
          f"frozen {m['frozen_at']}")

if args.action == "holdout-status":
    from adgraph.a3 import holdout

    st = holdout.status(P)
    print(st.groupby(["part", "exp", "holdout_state"]).size().to_string())

if args.action == "holdout":
    from adgraph.a3 import holdout, freeze

    if not (args.part and args.exp):
        sys.exit("--part and --exp are required")
    try:
        man = holdout.run_part(P, args.part, args.exp, args.repo, Path(args.paths).resolve(), args.python, args.base)
    except (holdout.HoldoutRefused, freeze.FreezeInvalid) as exc:
        sys.exit(f"REFUSED: {exc}")
    print(f"{args.part}/{args.exp}: exit code {man['exit_code']} at pin {man['pin'][:10]} (descriptive only)")
    sys.exit(man["exit_code"])

if args.action == "report":
    from adgraph.a3 import report

    res = report.development_report(P)
    print(res)

if args.action == "holdout-report":
    from adgraph.a3 import freeze, report

    try:
        print(report.holdout_report(P))
    except freeze.FreezeInvalid as exc:
        sys.exit(f"REFUSED: {exc}")

if args.action == "d2-manifest":
    from adgraph.a3 import d2, report

    atlas, graph = report.decisions_of(P)
    print(d2.write(P, atlas, graph))
