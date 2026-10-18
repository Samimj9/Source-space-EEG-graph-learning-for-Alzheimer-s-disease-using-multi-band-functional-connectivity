"""Source-graph stores of the final framework, one tool for both inverses and every atlas (configs/source_graphs.yaml).

    python scripts/build_stores.py status   --inverse lcmv [--atlas dk68]
    python scripts/build_stores.py inspect  --inverse lcmv            # recording QC (no source model; minutes)
    python scripts/build_stores.py build    --inverse wmne --atlas r18 [--limit 1]   # NEW stores only (see below)
    python scripts/build_stores.py validate --inverse wmne --atlas r18
    python scripts/build_stores.py match    --atlas r18               # wMNE vs LCMV, window by window
    python scripts/build_stores.py match-atlas --inverse lcmv --atlas r18   # r18 vs the 68-ROI store, window by window
    python scripts/build_stores.py size     --inverse lcmv --atlas r18   # disk use of the two stores

The 68-ROI stores of both inverses already exist, are validated and are only READ by the final framework:
``build`` refuses atlas dk68 (the stores would be recomputed at this commit) and ``correct`` is not offered (the
corr1 correction of the LCMV stores is done). A coarser atlas (``--atlas dk34_gyral | r18 | lobes12``) is built into
its own new store directories by the unchanged pipeline (merged Desikan-Killiany labels, ``mean_flip`` over their own
sources; never averages of 68-ROI signals). Building one needs the author's approval (docs/final/parcellation.md;
CPU, about 40 min for LCMV and 30-45 min for wMNE per atlas).

``validate`` / ``match`` / ``match-atlas`` write ``validation.json`` / ``match.json`` / ``atlas_match.json`` into a
store only if the store has none yet; an existing record is never overwritten (a re-check is printed and saved under
``FINAL_v1/stores/``). ``match-atlas`` requires, against the validated 68-ROI store of the same inverse and mode,
identical participants, labels, five-minute selections, windows, split, bands, channels, sampling and sensor
features (``sourcegraphs.match.match_atlas``); experiments refuse a coarser-atlas store without it.

Coarser-atlas stores are written next to the 68-ROI stores (``<processed>/D3_LCMV_XU/graphsLX-<atlas>_v1_<hash>``,
``<processed>/D3_WMNE_XU/graphsWX-<atlas>_v1_<hash>``, and the ``10`` mode A variants), i.e. OUTSIDE ``FINAL_v1/``;
each carries its own manifest (configuration, code commit, software versions, input hashes, timestamps).
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adgraph import release
from adgraph.config import load_paths
from adgraph.sourcegraphs import build
from adgraph.sourcegraphs.config import resolve
from adgraph.sourcegraphs.match import match, match_atlas
from adgraph.sourcegraphs.validate import validate_run
from adgraph.stores import store_dir

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["status", "inspect", "build", "validate", "match", "match-atlas", "size"])
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--inverse", default="lcmv", choices=["lcmv", "wmne"])
parser.add_argument("--atlas", default="dk68")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--allow-ephemeral-storage", action="store_true")
args = parser.parse_args()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 40)
paths = load_paths(args.paths)
P = Path(paths["processed_dir"])
cfg = resolve(args.inverse, args.atlas)
dirs = {m: store_dir(P, args.inverse, m, args.atlas) for m in ("B", "A")}
print(f"{args.inverse} / {args.atlas}:\n  mode B: {dirs['B']}\n  mode A: {dirs['A']}\n", flush=True)


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_record(d: Path, name: str, res: dict) -> Path:
    """The store's own record if it has none, else a dated copy under FINAL_v1/stores/ (never overwritten)."""
    own = d / name
    if not own.exists():
        own.write_text(json.dumps(res, indent=1, default=str))
        return own
    out = release.root(P) / "stores" / f"{d.name}_{Path(name).stem}_{stamp()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    return out


def show(res: dict) -> None:
    for name, c in res["checks"].items():
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name}" + (f": {c['detail']}" if c["detail"] else ""))


if args.action == "status":
    for m, d in dirs.items():
        st = d / "status.csv"
        done = (d / "assembly.done.json").exists()
        print(f"mode {m}: {'assembled' if done else 'not assembled'}; validation "
              f"{'present' if (d / 'validation.json').exists() else 'missing'}"
              + (f"; match {'present' if (d / 'match.json').exists() else 'missing'}" if args.inverse == "wmne" else ""))
        if st.exists():
            print(pd.read_csv(st)["status"].value_counts().to_string())

if args.action == "inspect":
    df = build.inspect(paths, cfg, P, args.limit)
    cols = ["key", "label", "original_duration_s", "n_boundary_events", "n_segments", "selection_mode",
            "selected_duration_s", "n_candidate_windows", "status"]
    print(df[[c for c in cols if c in df]].to_string(index=False))
    sys.exit(1 if df.get("status", pd.Series("", index=df.index)).astype(str).str.startswith("error").any() else 0)

if args.action == "build":
    from adgraph.provenance import verify_environment

    if args.atlas == "dk68":
        sys.exit("the 68-ROI stores exist and are validated; the final framework only reads them (building them at "
                 "this commit would recompute every participant). Coarser atlases: --atlas dk34_gyral | r18 | lobes12")
    verify_environment()
    run_b, run_a = build.run_dirs(P, cfg, args.limit)
    b, a = build.run(paths, cfg, P, f"{paths['cache_dir']}/subjects", args.limit,
                     allow_ephemeral=args.allow_ephemeral_storage)
    qc = pd.read_csv(b / "qc_report.csv")
    print(qc.to_string(index=False))
    print(f"\nassembled: {b}\n           {a}")

if args.action == "validate":
    bad = False
    for m, d in dirs.items():
        res = validate_run(d)
        path = save_record(d, "validation.json", res)
        print(f"\nmode {m} {d.name}: {res['n_windows']} windows, {res['n_participants_ok']} participants ok -> {path}")
        show(res)
        bad |= not res["passed"]
    sys.exit(1 if bad else 0)

if args.action == "match":
    if args.limit:
        sys.exit("match applies to the full stores only")
    bad = False
    for m in ("B", "A"):
        target, ref = store_dir(P, "wmne", m, args.atlas), store_dir(P, "lcmv", m, args.atlas)
        res = match(target, ref)
        path = save_record(target, "match.json", res)
        print(f"\nmode {m}: {target.name} vs {ref.name}: {res['n_windows']} windows -> {path}")
        show(res)
        bad |= not res["passed"]
    sys.exit(1 if bad else 0)

if args.action == "match-atlas":
    if args.atlas == "dk68":
        sys.exit("match-atlas compares a coarser-atlas store with the 68-ROI store: pass --atlas r18 (or another)")
    bad = False
    for m in ("B", "A"):
        target, ref = dirs[m], store_dir(P, args.inverse, m, "dk68")
        res = match_atlas(target, ref, args.atlas)
        path = save_record(target, "atlas_match.json", res)
        print(f"\nmode {m}: {target.name} vs {ref.name}: {res['n_windows']} windows -> {path}")
        show(res)
        bad |= not res["passed"]
    sys.exit(1 if bad else 0)

if args.action == "size":
    for m, d in dirs.items():
        files = [f for f in d.rglob("*") if f.is_file()] if d.exists() else []
        top = sum(f.stat().st_size for f in files if f.parent == d)
        tot = sum(f.stat().st_size for f in files)
        print(f"mode {m}: {d} {tot / 1e9:.2f} GB (assembled arrays {top / 1e9:.2f} GB, participant checkpoints "
              f"{(tot - top) / 1e9:.2f} GB)")
