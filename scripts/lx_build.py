"""D3 LCMV / Xu-protocol dataset (methodology log entry 50, docs/lcmv_xu.md).

    python scripts/lx_build.py inspect          # recording QC of all 65 participants (no source model; minutes)
    python scripts/lx_build.py build --limit 1  # smoke test: one participant, full LCMV path (own _limit1 run)
    python scripts/lx_build.py build            # all 65 participants (resumable; completed participants skipped)
    python scripts/lx_build.py correct          # corr1: corrected copies of both stores (entry 53; no LCMV recompute)
    python scripts/lx_build.py validate         # boundary, selection, split, shape and finiteness checks of the stores
                                                # experiments use (the corrected ones when they exist; --original:
                                                # the original stores instead)
    python scripts/lx_build.py status

Experiments of this suite use only stores whose validation passed (``validate.require_validated``).
``build`` refuses to recompute participants whose checkpoints were made by another code version, and refuses to run
once corrected stores exist, unless ``--allow-recompute`` is given deliberately (it would rebuild all participants).

Reads the already downloaded ds004504 derivatives (paths.yaml dataset3.root); writes only under
<processed>/D3_LCMV_XU/. fsaverage: <cache_dir>/subjects (as the other pipelines).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from adgraph.provenance import verify_environment
from adgraph.config import load_paths, load_yaml
from adgraph.lcmvxu import build, correct
from adgraph.lcmvxu.validate import validate_run

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["inspect", "build", "correct", "validate", "status"])
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/lcmv_xu/pipeline.yaml")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--allow-ephemeral-storage", action="store_true")
parser.add_argument("--allow-recompute", action="store_true",
                    help="build: recompute participants made by another code version (rebuilds them)")
parser.add_argument("--original", action="store_true", help="validate: the original (uncorrected) stores")
args = parser.parse_args()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 40)
paths, cfg = load_paths(args.paths), load_yaml(args.config)
processed = paths["processed_dir"]
run_dir, run10 = build.run_dirs(processed, cfg, args.limit)
corr_b, corr_a = correct.corrected_dirs(run_dir, run10)
corrected = (corr_b / "assembly.done.json").exists() and (corr_a / "assembly.done.json").exists()
print(f"mode B (all windows): {run_dir}\nmode A (tenfold):     {run10}", flush=True)
if corrected:
    print(f"corrected stores ({correct.NAME}, used by the experiments):\n  {corr_b}\n  {corr_a}", flush=True)
print(flush=True)

if args.action == "inspect":
    df = build.inspect(paths, cfg, processed, args.limit)
    cols = ["key", "label", "original_duration_s", "n_boundary_events", "n_segments", "n_usable_segments",
            "usable_interior_s", "selection_mode", "n_pieces", "selected_duration_s", "n_candidate_windows", "status"]
    print(df[[c for c in cols if c in df]].to_string(index=False))
    if "selection_mode" in df:
        print("\nselection:", df["selection_mode"].value_counts().to_dict(),
              f"| participants with < 300 s selected: {int((df['selected_duration_s'] < 300).sum())}")
    errors = df[df.get("status", pd.Series("", index=df.index)).astype(str).str.startswith("error")]
    sys.exit(1 if len(errors) else 0)

if args.action == "build":
    verify_environment()
    from adgraph.checkpoint import code_identity

    if corrected and not args.allow_recompute:
        print(f"corrected stores exist ({corr_b.name}): the dataset is complete, nothing to build (at this commit "
              "`build` would recompute every participant). `validate` checks the corrected stores.")
        sys.exit(0)
    here = code_identity("lx_graph")
    other = [m.parent.name for m in sorted((run_dir / "subjects").glob("*/subject.done.json"))
             if json.loads(m.read_text())["identity"].get("code") != here] if (run_dir / "subjects").exists() else []
    if other and not args.allow_recompute and not args.limit:      # a --limit smoke test may be redone
        sys.exit(f"{len(other)} participant checkpoint(s) in {run_dir.name} were made by another code version "
                 f"(e.g. {other[:3]}); `build` would recompute them. Use the code pin they were built with, or pass "
                 "--allow-recompute deliberately.")
    a, b = build.run(paths, cfg, processed, f"{paths['cache_dir']}/subjects", args.limit,
                     allow_ephemeral=args.allow_ephemeral_storage)
    qc = pd.read_csv(a / "qc_report.csv")
    cols = ["key", "label", "n_boundary_events", "n_segments", "selection_mode", "selected_duration_s",
            "n_windows_candidate", "n_windows_accepted", "n_windows_rejected", "n_tenfold", "lcmv_rank",
            "lcmv_verification_rel_error", "status", "reason"]
    print(qc[[c for c in cols if c in qc]].to_string(index=False))
    print(f"\nassembled: {a}\n           {b}")

if args.action == "correct":
    if args.limit:
        sys.exit("correct applies to the full stores only")
    b, a, summary = correct.correct(run_dir, run10)
    print(f"participants changed: {summary['participants_changed']}")
    for w in summary["windows_removed"]:
        print(f"  removed {w['key']} window [{w['start_sample']}, {w['stop_sample']}) (segment {w['segment_index']}, "
              f"piece {w['piece']}, outside the corrected 300 s selection)")
    print(pd.DataFrame(summary["participants"]).query("changed").to_string(index=False))
    print(f"mode B windows: {summary['n_windows_mode_B']}\nmode A windows: {summary['n_windows_mode_A']}")
    print(f"beamforming repeated: {summary['beamforming_repeated']}; connectivity / features recomputed: "
          f"{summary['connectivity_or_features_recomputed']}\nwritten: {b}\n         {a}")

if args.action == "validate":
    bad = False
    targets = (run_dir, run10) if args.original or not corrected or args.limit else (corr_b, corr_a)
    for d in targets:
        res = validate_run(d)
        (d / "validation.json").write_text(json.dumps(res, indent=1, default=str))
        print(f"\n{d.name} ({res['mode']}): {res['n_windows']} windows, {res['n_participants_ok']} participants ok, "
              f"{res['n_participants_not_ok']} not ok")
        for name, c in res["checks"].items():
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name}" + (f": {c['detail']}" if c["detail"] else ""))
        bad |= not res["passed"]
    sys.exit(1 if bad else 0)

if args.action == "status":
    st = run_dir / "status.csv"
    print(pd.read_csv(st).to_string(index=False) if st.exists() else "no run yet")
