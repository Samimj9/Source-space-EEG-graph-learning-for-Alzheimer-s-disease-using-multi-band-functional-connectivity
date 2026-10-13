"""Build multi-band source-space connectivity graphs for D1 and D2 (resumable).

    python scripts/build_graphs.py --paths configs/paths.yaml            # run / resume
    python scripts/build_graphs.py --paths configs/paths.yaml --status   # show progress only

Outputs go to <processed>/graphs_v<version>_<config hash>/. Every subject is saved as soon as
it finishes (subjects/<key>/ + validated marker); rerunning the same command skips completed
subjects and continues (--resume, default). The sensor->ROI operator is cached under
<cache_dir>/operators and reused while its source-model hash matches. --force starts a fresh
run directory (..._force<stamp>) and rebuilds the operator; nothing is overwritten.

Every output and cache location must be on the mounted Google Drive (refused otherwise, unless
--allow-ephemeral-storage). Input EDFs and participant/layout files are identified by SHA-256,
cached in <cache_dir>/input_sha256.json (--rehash-inputs re-reads every file). Checkpoints are
also tied to the code (git commit + source hash), so resume with the same commit. The pinned
environment (requirements.txt) is verified before a run.
"""

import argparse
import logging

from adgraph.config import load_paths, load_yaml
import pandas as pd

from adgraph.pipeline import run, status
from adgraph.progress import log_handler_context
from adgraph.provenance import verify_environment

parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/pipeline.yaml")
parser.add_argument("--out-root", help="defaults to processed_dir from paths.yaml")
parser.add_argument("--subjects-dir", help="fsaverage location; defaults to <cache_dir>/subjects")
parser.add_argument("--limit", type=int, help="only the first N subjects per dataset (smoke test)")
parser.add_argument("--run-dir", help="explicit run directory to resume (default: derived from the config hash)")
parser.add_argument("--operator-cache", help="defaults to <cache_dir>/operators")
parser.add_argument("--input-hash-cache", help="defaults to <cache_dir>/input_sha256.json")
parser.add_argument("--rehash-inputs", action="store_true", help="recompute every input SHA-256 (ignore the cache)")
parser.add_argument("--allow-ephemeral-storage", action="store_true",
                    help="permit outputs/caches outside the mounted Drive (they are lost on disconnect)")
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
hash_cache = args.input_hash_cache or f"{paths['cache_dir']}/input_sha256.json"
if args.status:
    st = status(paths, cfg, out_root, args.limit, args.run_dir, op_cache, hash_cache)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(f"run dir:   {st['run_dir']}\nconfig:    {st['config_hash']}\ncommit:    {st['git_commit']} "
          f"(uncommitted changes: {st['git_dirty']}; units marked 'changed: code' were built with other code)\n"
          f"operator:  {st['operator_id']} "
          f"({st['operator_cache']})\nassembly:  {st['assembly']}\ncounts:    {st['counts']}\n")
    print(st["subjects"].to_string())
    raise SystemExit
verify_environment()
with log_handler_context():                        # log lines print above the progress bars
    out = run(paths, cfg, out_root=out_root, subjects_dir=args.subjects_dir or f"{paths['cache_dir']}/subjects",
              limit=args.limit, resume=args.resume, force=args.force, run_dir=args.run_dir, operator_cache=op_cache,
              input_hash_cache=hash_cache, rehash_inputs=args.rehash_inputs,
              allow_ephemeral=args.allow_ephemeral_storage)
print(f"outputs: {out}")
