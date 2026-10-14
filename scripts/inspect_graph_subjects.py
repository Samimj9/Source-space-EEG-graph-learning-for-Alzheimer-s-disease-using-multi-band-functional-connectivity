"""Inspect the per-subject outputs of a graph run before approving the full build (read-only).

    python scripts/inspect_graph_subjects.py --paths configs/paths.yaml --limit 1     # the smoke test
    python scripts/inspect_graph_subjects.py --run-dir <processed>/graphs_v3_<hash> --keys D1-3

For every subject unit in the run (or ``--keys``) it prints and plots:

* preprocessing QC: unit safeguard, amplitude plausibility (analysed interval), robust std,
  bad channels (and what whole-recording detection would have flagged), per-channel table;
* window rejection: every candidate window, kept or rejected and why, peak-to-peak against the
  adaptive threshold, windows within the band-pass reach of the discarded edges;
* the 68-ROI reconstruction: operator verification error, ROI band power (source PSD) per band,
  sensor band power;
* PLV and |ciPLV|: mean matrices per band over the kept windows and their value distributions.

Figures are diagnostics, not paper figures. Outputs go to a new folder
``<run>/inspection_<UTC stamp>/`` (``<key>.png``, ``summary.json``); nothing else is written.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from adgraph.config import load_paths, load_yaml
from adgraph.pipeline import graph_run_dir

parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/pipeline.yaml")
parser.add_argument("--run-dir", help="graph run directory (default: derived from the config and --limit)")
parser.add_argument("--limit", type=int, help="the smoke-test run (--limit used by build_graphs.py)")
parser.add_argument("--keys", nargs="*", help="subjects to inspect (default: all units in the run)")
args = parser.parse_args()

cfg = load_yaml(args.config)
run = Path(args.run_dir) if args.run_dir else graph_run_dir(load_paths(args.paths)["processed_dir"], cfg, args.limit)
if not (run / "subjects").is_dir():
    raise SystemExit(f"no subject units in {run}; run build_graphs.py first")
keys = args.keys or sorted(p.name for p in (run / "subjects").iterdir() if p.is_dir())
bands = list(cfg["connectivity"]["bands"])
manifest = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
op = manifest.get("source_operator", {})
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
print(f"run: {run}")
if op:
    print(f"operator {op['operator_id']}: {op['shape']} (ROIs x channels), {op['n_sources']} sources, "
          f"verification relative error {op['verification_rel_error']:.2g}")
last = run / "last_attempt.json"
if last.exists():
    la = json.loads(last.read_text())
    print(f"last attempt: {la['result']} ({la['reason']})")

out = run / f"inspection_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True, exist_ok=False)
summary = {}
for key in keys:
    unit = run / "subjects" / key
    print(f"\n{'=' * 100}\n{key}\n{'=' * 100}")
    if (unit / "failed.json").exists() and not (unit / "subject.done.json").exists():
        print("FAILED:", json.loads((unit / "failed.json").read_text())["status"])
        summary[key] = {"status": "failed"}
        continue
    if not (unit / "subject.done.json").exists():
        print("not completed")
        continue
    s = json.loads((unit / "subject.json").read_text())
    rows = ["status", "orig_sfreq", "duration_s", "unit_scale", "gross_robust_std_uv", "median_std_uv",
            "median_robust_std_uv", "whole_recording_std_uv", "n_bad_channels", "bad_channels",
            "bad_channels_whole_recording", "bad_fraction_warning", "rejection_threshold_uv", "n_windows_total",
            "n_windows_kept", "n_windows_rejected", "qc_edge_window_ptp_ratio", "qc_spatial_rho", "channel_naming"]
    print(pd.Series({k: s.get(k) for k in rows}).to_string())
    ch = pd.read_csv(unit / "qc_channels.csv") if (unit / "qc_channels.csv").exists() else None
    win = pd.read_csv(unit / "qc_windows.csv") if (unit / "qc_windows.csv").exists() else None
    if ch is not None:
        print("\nper channel (µV; analysed interval unless 'whole'; final = interpolated + average reference):")
        print(ch.round(2).to_string(index=False))
    if win is not None:
        print("\nwindows (all candidates):")
        print(win.round(1).to_string(index=False))
    with np.load(unit / "arrays.npz") as z:
        arr = {k: z[k] for k in z.files}
    n = arr["plv"].shape[0]
    roi_rel = arr["psd_rel"].mean(0) if n else np.full((len(bands), 68), np.nan)        # (bands, ROIs)
    sens_rel = arr["sensor_psd_rel"].mean(0) if n else np.full((len(bands), 30), np.nan)
    plv, ci = (arr["plv"].mean(0), arr["ciplv"].mean(0)) if n else (None, None)
    iu = np.triu_indices(arr["plv"].shape[-1], 1)
    stats = {}
    for b, name in enumerate(bands):
        if n:
            p, c = arr["plv"][:, b][:, iu[0], iu[1]], arr["ciplv"][:, b][:, iu[0], iu[1]]
            stats[name] = {"roi_rel_power_median": float(np.median(roi_rel[b])),
                           "sensor_rel_power_median": float(np.median(sens_rel[b])),
                           "plv_median": float(np.median(p)), "plv_p95": float(np.percentile(p, 95)),
                           "ciplv_median": float(np.median(c)), "ciplv_p95": float(np.percentile(c, 95))}
    print("\nper band over kept windows (relative power median over ROIs / sensors; PLV, |ciPLV| over edges):")
    print(pd.DataFrame(stats).T.round(4).to_string())
    summary[key] = {"subject": {k: s.get(k) for k in rows}, "bands": stats}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(4, 4, hspace=0.55, wspace=0.3)
    ax = fig.add_subplot(gs[0, :2])
    if win is not None:
        col = np.where(win["kept"], "tab:blue", "tab:red")
        ax.bar(win["tmin"] + 10, win["median_ptp_uv"], width=8, color=col,
               hatch=["//" if r else "" for r in win["within_edge_filter_reach"]])
        ax.scatter(win["tmin"] + 10, win["max_ptp_uv"], color=col, marker="_", s=200)
        ax.axhline(win["threshold_uv"].iloc[0], color="k", ls="--", lw=0.8, label="rejection threshold")
        trim = cfg["windowing"]["edge_trim_s"]
        for t0, t1 in ((0, trim), (s["duration_s"] - trim, s["duration_s"])):
            ax.axvspan(t0, t1, color="grey", alpha=0.3)
        if cfg.get("boundary"):                   # v4: boundary margin inside the cropped signal (entry 41)
            from adgraph.preprocessing.boundary import margins

            mg = margins(cfg)["margin_s"]
            for t0, t1 in ((trim, trim + mg), (s["duration_s"] - trim - mg, s["duration_s"] - trim)):
                ax.axvspan(t0, t1, color="grey", alpha=0.12)
        ax.set_xlabel("window centre (s); grey = discarded edges, light grey = v4 boundary margin; "
                      "hatched = within band-pass reach of the edges (v3)")
        ax.set_ylabel("ptp (µV): bar median, tick max")
        ax.set_title(f"{key}: windows ({int(win['kept'].sum())} kept blue / {int((~win['kept']).sum())} rejected red)")
        ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[0, 2:])
    if ch is not None:
        x = np.arange(len(ch))
        ax.plot(x, ch["std_uv"], "o-", ms=3, label="std (recording ref)")
        ax.plot(x, ch["robust_std_uv"], "o-", ms=3, label="robust std")
        ax.plot(x, ch["final_std_uv"], "o-", ms=3, label="final std (avg ref)")
        for i in np.flatnonzero(ch["bad_reason"].fillna("").astype(str).str.len() > 0):
            ax.axvline(i, color="red", alpha=0.3)
        ax.set_xticks(x, ch["channel"], rotation=90, fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel("µV (analysed interval)")
        ax.set_title("channels (red = bad, interpolated)")
        ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(roi_rel, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(bands)), bands)
    ax.set_xlabel("ROI (Desikan-Killiany order)")
    ax.set_title("source relative band power (mean over kept windows)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax = fig.add_subplot(gs[1, 2:])
    im = ax.imshow(sens_rel, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(bands)), bands)
    ax.set_xticks(range(len(ch)) if ch is not None else [], ch["channel"] if ch is not None else [],
                  rotation=90, fontsize=7)
    ax.set_title("sensor relative band power")
    fig.colorbar(im, ax=ax, fraction=0.03)
    for b, name in enumerate(bands):
        for row, (mat, label) in enumerate(((plv, "PLV"), (ci, "|ciPLV|"))):
            ax = fig.add_subplot(gs[2 + row, b])
            if mat is not None:
                im = ax.imshow(mat[b], cmap="magma", vmin=0)
                fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"{label} {name} (mean)", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{key}: graph-construction inspection (diagnostic; not a paper figure)")
    fig.savefig(out / f"{key}.png", dpi=90)
    plt.close(fig)

(out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(f"\nsaved to {out}")
