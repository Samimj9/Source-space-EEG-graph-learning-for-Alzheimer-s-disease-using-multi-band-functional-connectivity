"""Structural and numerical validation of an assembled graph run + the final dataset manifest.

    python scripts/legacy/validate_graph_run.py --paths configs/paths.yaml                 # latest full graph run
    python scripts/legacy/validate_graph_run.py --run-dir <processed>/graphs_v4_<hash>
    python scripts/legacy/validate_graph_run.py --paths configs/paths.yaml --reference-audit <audit dir>/part2

Checks (implementation checks, not a scientific analysis):
* every registered recording (20 D1 + 23 D2) has a row; status counts; failures listed;
* array shapes match the windows table; every array finite;
* PLV and |ciPLV| in [0, 1], symmetric, zero diagonal; signed ciPLV antisymmetric in [-1, 1];
* PSD: absolute > 0, log = log10(absolute), relative in [0, 1];
* windows: 20 s long, on the 10 s grid, unique per subject, kept count = windows per subject,
  sample weights sum to 1; for pipeline v4 every window lies at least the boundary margin inside the
  cropped interval (methodology log entry 41);
* manifest: code commit (clean tree), configuration, boundary processing, source operator.

Writes ``<run>/validation_<UTC stamp>/``: ``dataset_report.csv`` (one row per recording: status,
candidate / accepted / rejected windows, graphs, output paths), ``validation.json`` (every check and
the totals) and prints a completion report. Exit status 1 if any check fails. Nothing else is written.
With ``--reference-audit`` the accepted-window counts of the production run audited before the rebuild
(qc_boundary_audit part2 summaries) are listed next to the new counts.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from adgraph.checkpoint import atomic_write_csv, atomic_write_json
from adgraph.config import load_paths, load_yaml
from adgraph.graphs.store import GraphStore, check_adjacency
from adgraph.legacy.pipeline import build_registry, latest_complete_graph_run

parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--run-dir", help="graph run directory (default: latest assembled full run)")
parser.add_argument("--reference-audit", help="audit part2 folder with per-recording summaries (old window counts)")
parser.add_argument("--allow-dirty", action="store_true", help="synthetic tests only: accept uncommitted code")
parser.add_argument("--prefix", default="graphs", help="graph-run family: graphs (D1/D2) or graphsD3 (Dataset 3)")
args = parser.parse_args()

paths = load_paths(args.paths)
run = Path(args.run_dir) if args.run_dir else latest_complete_graph_run(paths["processed_dir"], args.prefix)
checks: list[dict] = []


def check(name: str, ok: bool, detail="") -> None:
    checks.append({"check": name, "passed": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


print(f"graph run: {run}")
check("assembly completion marker", (run / "assembly.done.json").exists())
manifest = json.loads((run / "manifest.json").read_text())
cfg = manifest["config"]
v4 = bool(cfg.get("boundary"))
check("manifest records code commit with a clean tree",
      bool(manifest.get("git_commit")) and (args.allow_dirty or not manifest.get("git_dirty")),
      f"{manifest.get('git_commit')} dirty={manifest.get('git_dirty')}")
check("manifest records the boundary processing", "boundary_processing" in manifest,
      json.dumps({k: manifest.get("boundary_processing", {}).get(k) for k in ("policy", "margin_samples", "margin_s")}))
op = manifest.get("source_operator", {})
check("source operator verified", op.get("verification_rel_error", 1) < 1e-6,
      f"{op.get('operator_id')} rel. err {op.get('verification_rel_error')}; compared with cached: "
      f"{op.get('compared_with_cached')}")

store = GraphStore.load(run)                 # validates shapes, adjacency contract, ciPLV antisymmetry
check("arrays match the windows table; adjacency contract (GraphStore.validate)", True,
      f"{store.n_windows} windows, arrays {sorted(store.arrays)}")
for k, a in store.arrays.items():
    check(f"{k}: all values finite", bool(np.isfinite(a).all()), f"shape {a.shape}")
for k in ("plv", "ciplv"):
    if k in store.arrays:
        try:
            check_adjacency(store.arrays[k], k)
            check(f"{k}: symmetric, zero diagonal, in [0, 1]", True)
        except ValueError as exc:
            check(f"{k}: symmetric, zero diagonal, in [0, 1]", False, str(exc))
for prefix in ("psd", "sensor_psd"):
    if f"{prefix}_abs" in store.arrays:
        ab, lg, rl = (store.arrays[f"{prefix}_{s}"].astype(float) for s in ("abs", "log", "rel"))
        check(f"{prefix}: absolute > 0, log = log10(absolute), relative in [0, 1]",
              bool((ab > 0).all() and np.allclose(np.log10(ab), lg, atol=1e-3) and (rl >= 0).all() and (rl <= 1).all()))

w, subj = store.windows, store.subjects
reg = build_registry(paths, cfg=cfg)[0]
win_len = manifest["segmentation"]["window_samples"]
step = manifest["segmentation"]["step_samples"]
check("every registered recording has a subject row", set(reg["key"]) <= set(subj["key"]),
      f"{len(reg)} registered, {len(subj)} rows; missing {sorted(set(reg['key']) - set(subj['key']))}")
check("windows are 20 s long", bool(((w["stop_sample"] - w["start_sample"]) == win_len).all()))
bad_grid, bad_margin, bad_count = [], [], []
margin = manifest.get("boundary_processing", {}).get("margin_samples", 0)
segmented = "segment_index" in w                    # Dataset 3: windows per continuous segment (entry 45)
bad_seg, crossing = [], {}
for key, g in w.groupby("key"):
    s = subj.set_index("key").loc[key]
    grid_groups = g.groupby("segment_index") if segmented else [(0, g)]
    rel_bad = any(((gg["start_sample"] - gg["start_sample"].min()) % step != 0).any() for _, gg in grid_groups)
    if g["window_index"].duplicated().any() or rel_bad:
        bad_grid.append(key)
    if segmented:
        if (g["start_sample"] < g["segment_start_sample"] + margin).any() or \
                (g["stop_sample"] > g["segment_stop_sample"] - margin).any():
            bad_seg.append(key)
        from adgraph.preprocessing.discontinuities import windows_crossing
        v = s.get("discontinuity_samples")
        cuts = json.loads(v) if isinstance(v, str) else []
        n_cross = int(windows_crossing(g["start_sample"], g["stop_sample"], cuts, margin).sum())
        if n_cross:
            crossing[key] = n_cross
    if int(s["n_windows_kept"]) != len(g) or not np.isclose(g["sample_weight"].sum(), 1):
        bad_count.append(key)
    if v4:
        lo = int(s["sample_offset"]) + margin
        hi = int(s["sample_offset"]) + int(s["processed_n_samples"]) - margin
        if (g["start_sample"] < lo).any() or (g["stop_sample"] > hi).any():
            bad_margin.append(key)
check("windows on the 10 s grid, unique per subject", not bad_grid, str(bad_grid or ""))
check("accepted windows = rows per subject; weights sum to 1", not bad_count, str(bad_count or ""))
if segmented:
    check("every window inside one continuous segment, at least the margin from both segment ends", not bad_seg,
          str(bad_seg or ""))
    check("ZERO accepted windows crossing an EEGLAB discontinuity or its margin", not crossing,
          f"margin {margin} samples; " + (f"crossing: {crossing}" if crossing else
                                          f"{int(subj.get('n_boundary_events', pd.Series(dtype=float)).fillna(0).sum())} "
                                          "boundary events in the files, all handled"))
if v4:
    check("every window at least the boundary margin inside the cropped interval", not bad_margin,
          f"margin {margin} samples ({margin / cfg['preprocessing']['sfreq']:.3f} s)" + (f"; {bad_margin}" if bad_margin else ""))

rows = []
n_bands = len(cfg["connectivity"]["bands"])
for r in subj.itertuples():
    kept = int(getattr(r, "n_windows_kept", 0) or 0) if str(r.status) == "ok" else 0
    rows.append({"key": r.key, "dataset": r.dataset, "label": r.label, "status": r.status,
                 "duration_s": getattr(r, "duration_s", np.nan),
                 "candidate_windows": getattr(r, "n_windows_total", np.nan),
                 "accepted_windows": getattr(r, "n_windows_kept", np.nan),
                 "rejected_windows": getattr(r, "n_windows_rejected", np.nan),
                 "graphs_in_dataset": kept, "band_adjacencies_per_metric": kept * n_bands,
                 "n_interpolated": getattr(r, "n_bad_channels", np.nan),
                 **({k: getattr(r, k, np.nan) for k in (
                     "n_boundary_events", "n_discontinuities_analysed", "n_segments", "n_segments_usable",
                     "usable_duration_s", "n_windows_continuous_grid", "n_windows_continuous_grid_crossing_discontinuity",
                     "n_windows_excluded_discontinuity")} if "n_boundary_events" in subj else {}),
                 "subject_unit": str(run / "subjects" / r.key)})
rep = pd.DataFrame(rows)
if args.reference_audit:
    ref = {}
    for f in Path(args.reference_audit).glob("*/summary.json"):
        d = json.loads(f.read_text())
        if d.get("status") == "ok":
            ref[d["key"]] = (d.get("n_candidates"), d.get("n_windows"))
    rep["reference_candidate_windows"] = rep["key"].map(lambda k: ref.get(k, (np.nan, np.nan))[0])
    rep["reference_accepted_windows"] = rep["key"].map(lambda k: ref.get(k, (np.nan, np.nan))[1])
    rep["accepted_change"] = rep["accepted_windows"] - rep["reference_accepted_windows"]

out = run / f"validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True, exist_ok=False)
atomic_write_csv(out / "dataset_report.csv", rep)
status_counts = rep["status"].astype(str).value_counts().to_dict()
totals = {"recordings_registered": int(len(reg)), "recordings_with_graphs": int((rep["graphs_in_dataset"] > 0).sum()),
          "status_counts": status_counts, "graphs": int(store.n_windows),
          "graphs_per_dataset": w.groupby("dataset").size().to_dict(),
          "graphs_per_label": w.groupby("label").size().to_dict(),
          "candidate_windows": int(rep["candidate_windows"].sum()), "rejected_windows": int(rep["rejected_windows"].sum()),
          "band_adjacencies_per_metric": int(store.n_windows * n_bands),
          "failures": rep.loc[rep["status"].astype(str).str.startswith("error"), ["key", "status"]].to_dict("records")}
if args.reference_audit:
    totals["reference_accepted_windows"] = float(rep["reference_accepted_windows"].sum())
atomic_write_json(out / "validation.json", {"run": str(run), "validated_at": datetime.now(timezone.utc).isoformat(),
                                            "git_commit": manifest.get("git_commit"), "graph_run_id": manifest.get("graph_run_id"),
                                            "config_version": cfg.get("version"), "checks": checks, "totals": totals,
                                            "boundary_processing": manifest.get("boundary_processing")})
pd.set_option("display.width", 220, "display.max_rows", 100)
print("\nper recording:")
print(rep.drop(columns=["subject_unit"]).to_string(index=False))
print("\nCOMPLETION REPORT")
print(json.dumps(totals, indent=1, default=str))
failed = [c["check"] for c in checks if not c["passed"]]
print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
print(f"saved to {out}")
sys.exit(1 if failed else 0)
