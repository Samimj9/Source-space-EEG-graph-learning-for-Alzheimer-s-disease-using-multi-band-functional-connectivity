"""Build multi-band source-space connectivity graphs for D1 and D2 (resumable).

    python scripts/build_graphs.py --paths configs/paths.yaml            # run / resume
    python scripts/build_graphs.py --paths configs/paths.yaml --status   # show progress only

Outputs go to <processed>/graphs_v<version>_<config hash>/. Every subject is saved as soon as
it finishes (subjects/<key>/ + validated marker); rerunning the same command skips completed
subjects and continues (--resume, default). The sensor->ROI operator is cached under
<cache_dir>/operators and reused while its source-model hash matches. --force starts a fresh
run directory (..._force<stamp>) and rebuilds the operator; nothing is overwritten.
"""

import argparse
import logging

from adgraph.config import load_paths, load_yaml
import pandas as pd

from adgraph.pipeline import run, status

parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/pipeline.yaml")
parser.add_argument("--out-root", help="defaults to processed_dir from paths.yaml")
parser.add_argument("--subjects-dir", help="fsaverage location; defaults to <cache_dir>/subjects")
parser.add_argument("--limit", type=int, help="only the first N subjects per dataset (smoke test)")
parser.add_argument("--run-dir", help="explicit run directory to resume (default: derived from the config hash)")
parser.add_argument("--operator-cache", help="defaults to <cache_dir>/operators")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--resume", dest="resume", action="store_true", default=True, help="continue (default)")
mode.add_argument("--no-resume", dest="resume", action="store_false", help="refuse an existing run directory")
mode.add_argument("--force", action="store_true", help="fresh run directory and operator rebuild")
parser.add_argument("--status", action="store_true", help="show pending/running/completed/failed and exit")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
paths, cfg = load_paths(args.paths), load_yaml(args.config)
out_root = args.out_root or paths["processed_dir"]
op_cache = args.operator_cache or f"{paths['cache_dir']}/operators"
if args.status:
    st = status(paths, cfg, out_root, args.limit, args.run_dir, op_cache)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(f"run dir:   {st['run_dir']}\nconfig:    {st['config_hash']}\noperator:  {st['operator_id']} "
          f"({st['operator_cache']})\nassembly:  {st['assembly']}\ncounts:    {st['counts']}\n")
    print(st["subjects"].to_string())
    raise SystemExit
out = run(paths, cfg, out_root=out_root, subjects_dir=args.subjects_dir or f"{paths['cache_dir']}/subjects",
          limit=args.limit, resume=args.resume, force=args.force, run_dir=args.run_dir, operator_cache=op_cache)
print(f"outputs: {out}")
