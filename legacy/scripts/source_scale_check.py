"""Read-only numerical check of the source features of the matched wMNE store against the LCMV store (entry 57).

    python scripts/source_scale_check.py [--mode B|A]

Nothing is changed or recomputed. Loads only the node-feature arrays (psd_abs / psd_log / psd_rel / psd_de, not the
connectivity) of the matched wMNE store and the corrected LCMV store of the same mode, and the windows tables
(identical by the match check), then reports:

1. finiteness and ranges of every array, per method;
2. realistic amplitudes: ROI root-mean-square amplitude sqrt(sum of the band powers 0.5-45 Hz) per window:
   wMNE in A*m (dipole moment), LCMV in units of the ad-hoc noise SD (unit-noise-gain beamformer);
3. pure-scale consistency: if the two inverse outputs differed only by a factor a, then
   d(log10 P) = 2 log10 a and d(DE) = ln a, i.e. d(DE) = d(log10 P) * ln(10) / 2 exactly; the residual
   d(DE) - 1.1513 d(log10 P) measures how far the spectra differ beyond a scale;
4. decomposition of d(log10 P) = wMNE - LCMV over participants x ROIs x bands (participant means of the windows):
   grand mean, ROI x band component, participant component, residual (variance shares);
5. between-participant agreement: per band and ROI, Spearman correlation across the 65 participants of the
   participant-mean log PSD, DE and relative PSD of the two methods (what z-scored models can actually use);
6. the same summaries per diagnosis group (descriptive only; no test, no model).

Writes <processed>/D3_WMNE_XU/reports/source_scale_<time>/ (report.json, components.csv, agreement.csv).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from adgraph.config import load_paths
from adgraph.pipeline import latest_complete_graph_run

FEATS = ("psd_abs", "psd_log", "psd_rel", "psd_de")
parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--mode", choices=["A", "B"], default="B")
args = parser.parse_args()
processed = load_paths(args.paths)["processed_dir"]
sfx = "10" if args.mode == "A" else ""
stores = {"wmne": latest_complete_graph_run(processed, f"D3_WMNE_XU/graphsWX{sfx}"),
          "lcmv": latest_complete_graph_run(processed, f"D3_LCMV_XU/graphsLX{sfx}")}
win = {m: pd.read_csv(d / "windows.csv") for m, d in stores.items()}
if not (win["wmne"][["window_id", "key"]].equals(win["lcmv"][["window_id", "key"]])):
    raise SystemExit("windows differ between the stores; run `python scripts/wx_build.py match` first")
meta = win["lcmv"]
arr = {}
for m, d in stores.items():
    with np.load(d / "graphs.npz") as z:
        arr[m] = {k: z[k].astype(np.float64) for k in FEATS}
        bands, rois = z["bands"].tolist(), z["roi_names"].tolist()
report = {"stores": {m: str(d) for m, d in stores.items()}, "mode": args.mode, "n_windows": int(len(meta))}

# 1. finiteness and ranges
report["ranges"] = {m: {k: {"finite": bool(np.isfinite(v).all()), "min": float(v.min()), "median": float(np.median(v)),
                            "max": float(v.max()), **({"n_nonpositive": int((v <= 0).sum())} if k == "psd_abs" else {})}
                        for k, v in a.items()} for m, a in arr.items()}
# 2. amplitudes: RMS of each ROI signal from its band powers (band powers sum the 0.5-45 Hz Welch PSD)
rms = {m: np.sqrt(a["psd_abs"].sum(axis=1)) for m, a in arr.items()}                    # (N, R)
report["roi_rms"] = {"wmne_A_m": {q: float(np.quantile(rms["wmne"], q)) for q in (0.05, 0.5, 0.95)},
                     "lcmv_noise_sd_units": {q: float(np.quantile(rms["lcmv"], q)) for q in (0.05, 0.5, 0.95)},
                     "median_ratio_wmne_over_lcmv": float(np.median(rms["wmne"] / rms["lcmv"]))}
# 3. pure-scale consistency
dlog = arr["wmne"]["psd_log"] - arr["lcmv"]["psd_log"]                                   # (N, K, R)
dde = arr["wmne"]["psd_de"] - arr["lcmv"]["psd_de"]
resid = dde - dlog * np.log(10) / 2
report["scale_consistency"] = {"mean_dlog10P": float(dlog.mean()), "mean_dDE": float(dde.mean()),
                               "predicted_dDE_from_dlog": float(dlog.mean() * np.log(10) / 2),
                               "implied_amplitude_factor_log10": float(dlog.mean() / 2),
                               "residual_abs_median": float(np.median(np.abs(resid))),
                               "residual_q05_q95": [float(np.quantile(resid, 0.05)), float(np.quantile(resid, 0.95))]}
# 4. variance decomposition of the participant-mean offset
keys = meta["key"].to_numpy()
order = sorted(set(keys))
pm = {m: {k: np.stack([a[k][keys == s].mean(axis=0) for s in order]) for k in ("psd_log", "psd_de", "psd_rel")}
      for m, a in arr.items()}                                                                # (S, K, R)
d = pm["wmne"]["psd_log"] - pm["lcmv"]["psd_log"]
g = d.mean()
roi_band = d.mean(axis=0, keepdims=True) - g
subj = d.mean(axis=(1, 2), keepdims=True) - g
res = d - g - roi_band - subj
tot = ((d - g) ** 2).mean()
share = (lambda x: float(x / tot)) if tot > 0 else (lambda x: None)     # identical stores: no variance to split
report["offset_decomposition_dlog10P"] = {
    "grand_mean": float(g), "sd_total": float(np.sqrt(tot)),
    "share_roi_x_band": share((roi_band ** 2).mean()), "share_participant": share((subj ** 2).mean()),
    "share_residual": share((res ** 2).mean()),
    "participant_offset_range": [float(subj.min() + g), float(subj.max() + g)]}
comp = pd.DataFrame([{"band": bands[b], "roi": rois[r], "mean_offset_dlog10P": float(g + roi_band[0, b, r]),
                      "sd_over_participants": float(d[:, b, r].std())} for b in range(len(bands)) for r in range(len(rois))])
# 5. between-participant agreement
rows = []
for k in ("psd_log", "psd_de", "psd_rel"):
    for b in range(len(bands)):
        for r in range(len(rois)):
            rows.append({"feature": k, "band": bands[b], "roi": rois[r],
                         "spearman": float(spearmanr(pm["wmne"][k][:, b, r], pm["lcmv"][k][:, b, r])[0])})
agree = pd.DataFrame(rows)
report["between_participant_spearman_median"] = agree.groupby(["feature", "band"])["spearman"].median().round(3) \
    .unstack().to_dict()
# 6. per diagnosis group (descriptive)
y = meta.drop_duplicates("key").set_index("key").loc[order, "y"].to_numpy()
report["offset_by_group_dlog10P"] = {lab: float(d[y == v].mean()) for lab, v in (("AD", 1), ("HC", 0))}
out = Path(processed) / "D3_WMNE_XU" / "reports" / f"source_scale_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True, exist_ok=True)
(out / "report.json").write_text(json.dumps(report, indent=1))
comp.to_csv(out / "components.csv", index=False)
agree.to_csv(out / "agreement.csv", index=False)
print(json.dumps({k: v for k, v in report.items() if k != "stores"}, indent=1))
print(f"\n-> {out}")
