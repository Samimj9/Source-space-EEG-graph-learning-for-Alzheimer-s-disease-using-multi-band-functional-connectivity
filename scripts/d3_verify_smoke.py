"""Dataset 3 real-data montage checks on the one-subject smoke test (read-only; run before the full build).

    python scripts/d3_verify_smoke.py

Every input is selected explicitly, never "the newest":
* the D3 smoke run = graph_run_dir(processed, configs/pipeline_d3.yaml, limit=1) (graphsD3_v4_<hash>_limit1);
* the D3 operator = the one recorded in that run's manifest (source_operator: W, channels, ROI names, error);
* the frozen D1/D2 run = the graphs_v* run whose manifest graph_run_id is --d1d2-graph-run-id (6d75de9975f0d9ae);
* the header report = every d3_header_report_*.csv written by `scripts/d3_data.py check` (latest file).

Checks: (1) electrode labels (raw -> canonical) and template positions of all 19 channels; (2) no confirmed flat
channel in the processed smoke subjects (a flat channel stops the subject: FlatChannelError); (3) inverse
operator 68 x 19 with the configured channel names and order, verified against the full MNE inverse; (4) 68 ROI
names and order identical to the frozen D1/D2 run; (5) EEGLAB 'boundary' discontinuities are allowed only if
handled: ZERO accepted windows cross a discontinuity or its margin, checked on the smoke subject(s) and, in
memory (nothing saved), on the recording with the most boundary events.
Writes d3_montage_checks.json into the smoke run; exits 1 if any check fails.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from adgraph.config import common_channels, load_paths, load_yaml
from adgraph.io import bids
from adgraph.io.eeg import channel_mapping, make_montage, read_header
from adgraph.pipeline import graph_run_dir, list_graph_runs

parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/pipeline_d3.yaml")
parser.add_argument("--d1d2-graph-run-id", default="6d75de9975f0d9ae", help="frozen D1/D2 graph run")
args = parser.parse_args()
paths, cfg = load_paths(args.paths), load_yaml(args.config)
processed = Path(paths["processed_dir"])
channels = common_channels(cfg)
pd.set_option("display.width", 200, "display.max_rows", 100)
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "passed": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)


smoke = graph_run_dir(processed, cfg, 1)
man = json.loads((smoke / "manifest.json").read_text()) if (smoke / "manifest.json").exists() else None
check("D3 smoke run assembled", man is not None, str(smoke))
if man is None:
    sys.exit(1)
subj = pd.read_csv(smoke / "subjects.csv")
keys = subj["key"].tolist()

# (1) electrode labels and positions
pos = make_montage(cfg["channels"]["montage"]).get_positions()["ch_pos"]
first = Path(man["inputs"][0]["path"])
raw = read_header(first, cfg["input"]["format"])
mapping, unknown = channel_mapping(raw.ch_names, cfg["channels"]["recorded"])
table = pd.DataFrame({"raw_label": list(mapping), "channel": list(mapping.values())})
table = table.set_index("channel").reindex(channels).reset_index()
table[["x_mm", "y_mm", "z_mm"]] = np.round(np.array([pos[c] for c in channels]) * 1000, 1)
print(f"\nelectrodes of {first.name} (raw label -> configured channel, template position):")
print(table.to_string(index=False))
lr = all(pos[c][0] < 0 for c in channels if c[-1] in "1357") and all(pos[c][0] > 0 for c in channels if c[-1] in "2468")
check("(1) all 19 labels map to the configured channels; positions on the correct hemisphere",
      table["raw_label"].notna().all() and not unknown and lr and (subj["unrecognised_channels"].fillna("") == "").all(),
      f"unrecognised {unknown}" if unknown else "")
check("(1) old names T3/T4/T5/T6 at the T7/T8/P7/P8 positions",
      all(np.allclose(pos[a], pos[b]) for a, b in (("T3", "T7"), ("T4", "T8"), ("T5", "P7"), ("T6", "P8"))))

# (2) flat channels
flat = {}
for k in keys:
    q = Path(smoke, "subjects", k, "qc_channels.csv")
    if q.exists():
        d = pd.read_csv(q)
        flat[k] = d.loc[d["bad_reason"].fillna("").astype(str).str.startswith("flat"), "channel"].tolist()
failed = [k for k in keys if Path(smoke, "subjects", k, "failed.json").exists()]
check("(2) no confirmed flat channel entered reconstruction", all(not v for v in flat.values()) and not failed
      and (subj["status"] == "ok").all(), f"flat {flat}; failed {failed}")
print("bad-channel report (reported only, not interpolated):", dict(zip(subj["key"], subj["bad_channels"])))

# (3) operator selected from this run's manifest
op = man["source_operator"]
W = np.asarray(op["W"])
check("(3) inverse operator is 68 x 19 with the configured channels in order",
      W.shape == (68, len(channels)) == tuple(op["shape"]) and op["channels"] == channels,
      f"operator {op['operator_id']}, shape {W.shape}")
check("(3) operator verified against the full MNE inverse (< 1e-6)",
      op["verification_rel_error"] is not None and op["verification_rel_error"] < 1e-6, f"{op['verification_rel_error']}")

# (4) ROI names and order vs the frozen D1/D2 run (selected by graph_run_id)
runs = list_graph_runs(processed, "graphs")
ref = runs[runs["graph_run_id"] == args.d1d2_graph_run_id]
if ref.empty:
    check("(4) frozen D1/D2 run found", False, f"no graphs_v* run with graph_run_id {args.d1d2_graph_run_id}")
else:
    ref_rois = json.loads((Path(ref["run_dir"].iloc[0]) / "manifest.json").read_text())["source_operator"]["roi_names"]
    check("(4) 68 ROI names and order identical to the frozen D1/D2 run",
          len(op["roi_names"]) == 68 and op["roi_names"] == ref_rois, f"vs {ref['run_dir'].iloc[0]}")

# (5) discontinuities: allowed only if handled -- ZERO accepted windows crossing a cut or its margin
from adgraph.pipeline import build_registry, process_subject
from adgraph.preprocessing.discontinuities import enabled as split_enabled, windows_crossing

reports = sorted(processed.glob("d3_header_report_*.csv"))
n_expected = sum(bids.EXPECTED.values())                       # 36 AD + 29 HC
if not reports:
    check("(5) header report present (run scripts/d3_data.py check)", False)
    rep = pd.DataFrame(columns=["key", "n_boundary_events"])
else:
    rep = pd.read_csv(reports[-1])
    n_files = int((rep["n_boundary_events"] > 0).sum())
    check(f"(5) header report covers all {n_expected} recordings", len(rep) == n_expected,
          f"{reports[-1].name}: {n_files} recordings with boundary events, {int(rep['n_boundary_events'].sum())} events")
    check("(5) discontinuity handling enabled for this configuration (discontinuities.handling: split)",
          split_enabled(cfg) or n_files == 0)
margin = int(man["boundary_processing"]["margin_samples"])
win = int(man["segmentation"]["window_samples"])
probe = list(keys)
if len(rep) and rep["n_boundary_events"].max() > 0:
    worst = rep.sort_values("n_boundary_events").iloc[-1]["key"]       # most discontinuities: in-memory probe
    if worst not in probe:
        probe.append(worst)
reg = build_registry(paths, cfg=cfg)[0].set_index("key")
op_mem = {"W": W, "ch_names": op["channels"], "roi_names": op["roi_names"]}
rows = []
for k in probe:
    if k in keys:                                               # processed and saved by the smoke test
        row = json.loads(Path(smoke, "subjects", k, "subject.json").read_text())
        wdf = pd.read_csv(Path(smoke, "subjects", k, "windows.csv"))
        source = "smoke run"
    else:                                                       # processed in memory only; nothing is saved
        rec = reg.loc[[k]].reset_index().iloc[0]
        print(f"\nprocessing {k} in memory (most boundary events; nothing saved) ...", flush=True)
        _, wdf, row, _, _ = process_subject(rec, cfg, op_mem)
        source = "in-memory probe"
    cuts = json.loads(row.get("discontinuity_samples") or "[]")
    cross = int(windows_crossing(wdf["start_sample"], wdf["stop_sample"], cuts, margin).sum()) if len(wdf) else 0
    outside = int(((wdf["start_sample"] < wdf["segment_start_sample"] + margin)
                   | (wdf["stop_sample"] > wdf["segment_stop_sample"] - margin)).sum()) if len(wdf) and \
        "segment_start_sample" in wdf else (0 if not len(wdf) else -1)
    in_report = rep.set_index("key")["n_boundary_events"].get(k, np.nan) if len(rep) else np.nan
    rows.append({"key": k, "source": source, "boundary_events_file": in_report,
                 "boundary_events_pipeline": row.get("n_boundary_events"), "segments": row.get("n_segments"),
                 "usable_segments": row.get("n_segments_usable"), "continuous_grid": row.get("n_windows_continuous_grid"),
                 "excluded_discontinuity": row.get("n_windows_excluded_discontinuity"),
                 "candidates": row.get("n_windows_total"), "rejected_qc": row.get("n_windows_rejected"),
                 "accepted": row.get("n_windows_kept"), "crossing": cross, "outside_segment_margin": outside,
                 "status": row.get("status")})
tab = pd.DataFrame(rows)
print("\ndiscontinuity handling (windows of 20 s; margin %d samples = %.3f s):" % (margin, margin / 250))
print(tab.to_string(index=False))
check("(5) the pipeline found the same boundary events as the header check",
      all(pd.isna(r["boundary_events_file"]) or r["boundary_events_file"] == r["boundary_events_pipeline"] for r in rows))
check("(5) ZERO accepted windows crossing a discontinuity or its margin", (tab["crossing"] == 0).all(),
      f"{int(tab['crossing'].sum())} crossing")
check("(5) every accepted window inside one continuous segment with the margin", (tab["outside_segment_margin"] == 0).all())
check("(5) no recording failed or was dropped because of boundary events",
      tab["status"].isin(["ok", "below_min_windows"]).all() and (tab["status"] == "ok").any(), str(tab["status"].tolist()))

(smoke / "d3_montage_checks.json").write_text(json.dumps({"checks": checks, "smoke_run": str(smoke)}, indent=1))
bad = [c["check"] for c in checks if not c["passed"]]
print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
