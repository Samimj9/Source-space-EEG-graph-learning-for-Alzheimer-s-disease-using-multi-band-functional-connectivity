"""Build multi-band source-space connectivity graphs for D1 and D2.

    python scripts/build_graphs.py --paths configs/paths.yaml \
        --subjects-dir /content/drive/MyDrive/AD-LEB/processed/cache/subjects
"""

import argparse
import logging

from adgraph.config import load_paths, load_yaml
from adgraph.pipeline import run

parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/pipeline.yaml")
parser.add_argument("--out-root", help="defaults to processed_dir from paths.yaml")
parser.add_argument("--subjects-dir", help="fsaverage location; defaults to <cache_dir>/subjects")
parser.add_argument("--limit", type=int, help="only the first N subjects per dataset (smoke test)")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
paths, cfg = load_paths(args.paths), load_yaml(args.config)
out = run(paths, cfg,
          out_root=args.out_root or paths["processed_dir"],
          subjects_dir=args.subjects_dir or f"{paths['cache_dir']}/subjects",
          limit=args.limit)
print(f"outputs: {out}")
