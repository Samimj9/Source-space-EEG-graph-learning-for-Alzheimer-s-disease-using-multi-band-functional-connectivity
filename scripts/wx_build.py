"""Matched wMNE / Xu-protocol dataset (methodology log entry 55, docs/wmne_xu.md).

    python scripts/wx_build.py inspect          # recording QC (no source model); must equal the LCMV suite's
    python scripts/wx_build.py build --limit 1  # smoke test: one participant, full wMNE path (own _limit1 run)
    python scripts/wx_build.py build            # all 65 participants (resumable; completed participants skipped)
    python scripts/wx_build.py validate         # boundary, selection, split, shape, weight and count checks
    python scripts/wx_build.py match            # window-by-window correspondence with the corrected LCMV stores
    python scripts/wx_build.py status

Same code and configuration as scripts/lx_build.py except the inverse (configs/wmne_xu/pipeline.yaml). Reads the
ds004504 derivatives (paths.yaml dataset3.root) and, for `match`, the corrected LCMV stores (read-only); writes only
under <processed>/D3_WMNE_XU/. Nothing of D3_LCMV_XU, graphsD3_v4_* or D1/D2 is written.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from adgraph.config import load_paths, load_yaml
from adgraph.lcmvxu import build
from adgraph.lcmvxu.match import match
from adgraph.lcmvxu.validate import validate_run
from adgraph.pipeline import latest_complete_graph_run
from adgraph.provenance import verify_environment

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["inspect", "build", "validate", "match", "status"])
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/wmne_xu/pipeline.yaml")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--allow-ephemeral-storage", action="store_true")
parser.add_argument("--allow-recompute", action="store_true",
                    help="build: recompute participants made by another code version (rebuilds them)")
args = parser.parse_args()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 40)
paths, cfg = load_paths(args.paths), load_yaml(args.config)
if build.method(cfg) != "wmne" or cfg.get("output_subdir") != "D3_WMNE_XU":
    sys.exit(f"{args.config} is not the matched wMNE configuration (source.method wmne, output D3_WMNE_XU)")
processed = paths["processed_dir"]
run_dir, run10 = build.run_dirs(processed, cfg, args.limit)
print(f"mode B (all windows): {run_dir}\nmode A (tenfold):     {run10}\n", flush=True)

if args.action == "inspect":
    df = build.inspect(paths, cfg, processed, args.limit)
    cols = ["key", "label", "original_duration_s", "n_boundary_events", "n_segments", "selection_mode",
            "selected_duration_s", "n_candidate_windows", "status"]
    print(df[[c for c in cols if c in df]].to_string(index=False))
    errors = df[df.get("status", pd.Series("", index=df.index)).astype(str).str.startswith("error")]
    sys.exit(1 if len(errors) else 0)

if args.action == "build":
    verify_environment()
    from adgraph.checkpoint import code_identity

    here = code_identity("lx_graph")
    other = [m.parent.name for m in sorted((run_dir / "subjects").glob("*/subject.done.json"))
             if json.loads(m.read_text())["identity"].get("code") != here] if (run_dir / "subjects").exists() else []
    if other and not args.allow_recompute and not args.limit:
        sys.exit(f"{len(other)} participant checkpoint(s) in {run_dir.name} were made by another code version "
                 f"(e.g. {other[:3]}); `build` would recompute them. Use the WX code pin they were built with, or "
                 "pass --allow-recompute deliberately.")
    a, b = build.run(paths, cfg, processed, f"{paths['cache_dir']}/subjects", args.limit,
                     allow_ephemeral=args.allow_ephemeral_storage)
    qc = pd.read_csv(a / "qc_report.csv")
    cols = ["key", "label", "selection_mode", "selected_duration_s", "n_windows_candidate", "n_windows_accepted",
            "n_tenfold", "wmne_verification_rel_error", "status", "reason"]
    print(qc[[c for c in cols if c in qc]].to_string(index=False))
    print(f"\nassembled: {a}\n           {b}")

if args.action == "validate":
    bad = False
    for d in (run_dir, run10):
        res = validate_run(d)
        (d / "validation.json").write_text(json.dumps(res, indent=1, default=str))
        print(f"\n{d.name} ({res['mode']}): {res['n_windows']} windows, {res['n_participants_ok']} participants ok")
        for name, c in res["checks"].items():
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name}" + (f": {c['detail']}" if c["detail"] else ""))
        bad |= not res["passed"]
    sys.exit(1 if bad else 0)

if args.action == "match":
    if args.limit:
        sys.exit("match applies to the full stores only")
    bad = False
    for d, prefix in ((run_dir, "D3_LCMV_XU/graphsLX"), (run10, "D3_LCMV_XU/graphsLX10")):
        ref = latest_complete_graph_run(processed, prefix)        # the corrected, validated LCMV store
        res = match(d, ref)
        (d / "match.json").write_text(json.dumps(res, indent=1, default=str))
        print(f"\n{d.name} vs {ref.name}: {res['n_windows']} windows")
        for name, c in res["checks"].items():
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name}" + (f": {c['detail']}" if c["detail"] else ""))
        for k, v in res["arrays"].items():
            print(f"    {k}: {v}")
        bad |= not res["passed"]
    sys.exit(1 if bad else 0)

if args.action == "status":
    st = run_dir / "status.csv"
    print(pd.read_csv(st).to_string(index=False) if st.exists() else "no run yet")
