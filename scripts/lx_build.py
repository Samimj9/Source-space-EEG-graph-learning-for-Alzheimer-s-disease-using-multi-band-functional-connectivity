"""D3 LCMV / Xu-protocol dataset (methodology log entry 50, docs/lcmv_xu.md).

    python scripts/lx_build.py inspect          # recording QC of all 65 participants (no source model; minutes)
    python scripts/lx_build.py build --limit 1  # smoke test: one participant, full LCMV path (own _limit1 run)
    python scripts/lx_build.py build            # all 65 participants (resumable; completed participants skipped)
    python scripts/lx_build.py validate         # boundary, split, shape and finiteness checks of both stores
    python scripts/lx_build.py status

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
from adgraph.lcmvxu import build
from adgraph.lcmvxu.validate import validate_run

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["inspect", "build", "validate", "status"])
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/lcmv_xu/pipeline.yaml")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--allow-ephemeral-storage", action="store_true")
args = parser.parse_args()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 40)
paths, cfg = load_paths(args.paths), load_yaml(args.config)
processed = paths["processed_dir"]
run_dir, run10 = build.run_dirs(processed, cfg, args.limit)
print(f"mode B (all windows): {run_dir}\nmode A (tenfold):     {run10}\n", flush=True)

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
    a, b = build.run(paths, cfg, processed, f"{paths['cache_dir']}/subjects", args.limit,
                     allow_ephemeral=args.allow_ephemeral_storage)
    qc = pd.read_csv(a / "qc_report.csv")
    cols = ["key", "label", "n_boundary_events", "n_segments", "selection_mode", "selected_duration_s",
            "n_windows_candidate", "n_windows_accepted", "n_windows_rejected", "n_tenfold", "lcmv_rank",
            "lcmv_verification_rel_error", "status", "reason"]
    print(qc[[c for c in cols if c in qc]].to_string(index=False))
    print(f"\nassembled: {a}\n           {b}")

if args.action == "validate":
    bad = False
    for d in (run_dir, run10):
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
