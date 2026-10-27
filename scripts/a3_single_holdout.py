"""Run the holdout of ONE frozen Exp 8a unit with the code of the checkout this script is run in (the A2 pin).

    cd <worktree at the A2 pin> && PYTHONPATH=<worktree>/src python <A3 repo>/scripts/a3_single_holdout.py \\
        --processed <outputs> --variant <v> --inverse <i> --mode B --atlas lobes12 --graph dense

The pinned package's own ``amendment.holdout`` runs every frozen Exp 8a unit; this driver applies exactly the same gates
for one: the amendment is initialized, the namespace's frozen record exists, names THIS code's commit and lists the unit;
then the pinned ``open_context`` (verified store, complete QC) and ``run_experiment`` (xu_holdout; frozen-record gate;
checkpointing and resume) run only that unit. It imports nothing from the A3 package (the pinned worktree does not have
it) and changes no model, protocol or configuration. A3 (``scripts/a3.py single-holdout``) wraps it with its own checks.
"""

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--processed", required=True)
parser.add_argument("--exp", default="exp8a")
parser.add_argument("--variant", required=True)
parser.add_argument("--inverse", required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--atlas", required=True)
parser.add_argument("--graph", required=True)
parser.add_argument("--allow-ephemeral-storage", action="store_true")
parser.add_argument("--skip-environment-check", action="store_true", help="tests only")


def main(argv=None) -> int:
    args = parser.parse_args(argv)
    from adgraph.experiments import amendment
    from adgraph.experiments.plan import experiment_variants, run_experiment
    from adgraph.provenance import git_commit, verify_environment

    if not args.skip_environment_check:
        verify_environment()
    P = Path(args.processed)
    amendment.require_initialized(P, "A2")
    f = amendment.final_path(P, "A2")
    if not f.exists():
        sys.exit(f"nothing is frozen in amendment A2 ({f})")
    d = json.loads(f.read_text())
    if d.get("git_commit") != git_commit():
        sys.exit(f"{f} was frozen at commit {d.get('git_commit')}, this code is {git_commit()}")
    want = {"exp": args.exp, "variant": args.variant, "inverse": args.inverse, "mode": args.mode, "atlas": args.atlas,
            "graph": args.graph}
    if not any(all(str(u.get(k)) == v for k, v in want.items()) for u in d.get("units", [])):
        sys.exit(f"{want} is not a frozen unit of amendment A2 ({f}); the holdout is refused")
    cfg = amendment.experiment_configs("A2")[args.exp]
    vs = experiment_variants(cfg, args.inverse, args.mode, args.atlas, args.graph, [args.variant])
    if [v["variant"] for v in vs] != [args.variant]:
        sys.exit(f"the variant selector {args.variant!r} matches {[v['variant'] for v in vs]}, not exactly one unit")
    store, rec, qc = amendment.open_context(P, args.inverse, args.mode, args.atlas)
    t = run_experiment(cfg, P, args.inverse, args.mode, args.atlas, args.graph, [args.variant], ["xu_holdout"],
                       results_root=amendment.results_dir(P, "A2"), store=store, store_record=rec, qc_report=qc,
                       allow_ephemeral=args.allow_ephemeral_storage, frozen_path=f)
    print(t.drop(columns=["updated"]).to_string(index=False))
    print("holdout results are descriptive only (release.HOLDOUT_POLICY)")
    return 0 if len(t) == 1 and (t["state"] == "completed").all() else 1


if __name__ == "__main__":
    sys.exit(main())
