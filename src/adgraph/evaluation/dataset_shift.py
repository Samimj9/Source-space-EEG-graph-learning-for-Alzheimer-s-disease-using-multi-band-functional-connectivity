"""Short, read-only diagnostic of why within-D1 classification is poor (Experiment 7 notebook).

Uses only the frozen graph run (QC tables, stored features, metadata) and saved experiment
predictions. Nothing is recomputed from EEG and nothing in the graph run or the results is changed.

1. Recording quality per dataset x group: amplitudes, bad / interpolated channels, accepted windows.
2. Channels interpolated repeatedly (e.g. persistently flat): how often, why, and whether the
   interpolated signal is non-flat afterwards (``final_std_uv``).
3. Feature distributions: subject means of ROI log / relative PSD, PLV and |ciPLV| per band;
   between-dataset shift within each group (standardized mean difference).
4. Class separability per dataset: subject-level univariate AUC (AD positive) of each summary
   feature, without any model.
5. Prediction behaviour of saved within-dataset results: near-constant scores, reversed ranking
   or no discrimination.

All statements are descriptive; nothing here is a test of significance or a reason to exclude data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

QC_COLUMNS = ["duration_s", "orig_sfreq", "gross_robust_std_uv", "median_std_uv", "median_robust_std_uv",
              "n_bad_channels", "n_windows_total", "n_windows_kept", "n_windows_rejected", "qc_spatial_rho",
              "age"]


def _auc(y, x) -> float:
    y, x = np.asarray(y), np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], x[ok]))


def subject_table(store) -> pd.DataFrame:
    s = store.subjects.copy()
    w = store.windows
    if s.empty:
        s = w.groupby("key").agg(dataset=("dataset", "first"), label=("label", "first")).reset_index()
    if "n_windows_kept" not in s:
        s = s.merge(w.groupby("key").size().rename("n_windows_kept"), on="key", how="left")
    if "y" not in s:
        s = s.merge(w.groupby("key")["y"].first(), on="key", how="left")
    return s


def quality_summary(subj: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in QC_COLUMNS if c in subj]
    g = subj.groupby(["dataset", "label"])[cols]
    out = pd.concat({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75)}, axis=1)
    out[("n", "subjects")] = subj.groupby(["dataset", "label"]).size()
    return out


def interpolated_channels(store, subj: pd.DataFrame, persistent_fraction: float = 0.3) -> pd.DataFrame:
    """Per dataset and channel: recordings with the channel interpolated, reasons, post-interpolation std."""
    rows = []
    root = getattr(store, "root", None)
    for key, ds in zip(subj["key"], subj["dataset"]):
        f = Path(root) / "subjects" / str(key) / "qc_channels.csv" if root else None
        if f is None or not f.exists():
            continue
        q = pd.read_csv(f)
        for _, r in q.iterrows():
            rows.append({"dataset": ds, "key": key, "channel": r["channel"],
                         "bad": bool(str(r.get("bad_reason", "")) not in ("", "nan")),
                         "reason": str(r.get("bad_reason", "")).split(" ")[0] if str(r.get("bad_reason", "")) not in ("", "nan") else "",
                         "interpolated": bool(r.get("interpolated", False)),
                         "final_std_uv": r.get("final_std_uv", np.nan)})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["dataset", "channel", "n_recordings", "n_bad", "fraction_bad", "reasons",
                                     "all_bad_interpolated", "min_final_std_uv_when_bad", "persistent"])
    agg = []
    for (ds, ch), g in df.groupby(["dataset", "channel"]):
        bad = g[g["bad"]]
        agg.append({"dataset": ds, "channel": ch, "n_recordings": g["key"].nunique(), "n_bad": len(bad),
                    "fraction_bad": len(bad) / g["key"].nunique(),
                    "reasons": ", ".join(f"{k}:{v}" for k, v in bad["reason"].value_counts().items()),
                    "all_bad_interpolated": bool(bad["interpolated"].all()) if len(bad) else True,
                    "min_final_std_uv_when_bad": float(bad["final_std_uv"].min()) if len(bad) else np.nan})
    out = pd.DataFrame(agg)
    out["persistent"] = out["fraction_bad"] >= persistent_fraction
    return out.sort_values(["dataset", "fraction_bad"], ascending=[True, False]).reset_index(drop=True)


def feature_table(store) -> pd.DataFrame:
    """Subject means of global features: ROI-mean log / rel PSD and mean PLV / |ciPLV| per band."""
    w = store.windows[["key"]].copy()
    iu = np.triu_indices(store.arrays["plv"].shape[-1], 1)
    for b, band in enumerate(store.bands):
        for kind in ("log", "rel"):
            k = f"psd_{kind}"
            if k in store.arrays:
                w[f"{k}_{band}"] = store.arrays[k][:, b, :].mean(-1)
        for conn in ("plv", "ciplv"):
            if conn in store.arrays:
                w[f"{conn}_{band}"] = store.arrays[conn][:, b][:, iu[0], iu[1]].mean(-1)
    return w.groupby("key").mean().reset_index()


def separability(subj: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Univariate subject-level AUC (AD positive) per feature and dataset, and dataset shift within group."""
    d = subj[["key", "dataset", "y"]].merge(feats, on="key")
    cols = [c for c in feats if c != "key"]
    rows = []
    for c in cols:
        rec = {"feature": c}
        for ds, g in d.groupby("dataset"):
            rec[f"auc_{ds}"] = _auc(g["y"], g[c])
            rec[f"mean_HC_{ds}"], rec[f"mean_AD_{ds}"] = g.loc[g["y"] == 0, c].mean(), g.loc[g["y"] == 1, c].mean()
        for grp, yv in (("HC", 0), ("AD", 1)):
            a = d[(d["dataset"] == "D1") & (d["y"] == yv)][c]
            b = d[(d["dataset"] == "D2") & (d["y"] == yv)][c]
            sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else np.nan
            rec[f"smd_D1_minus_D2_{grp}"] = (a.mean() - b.mean()) / sd if sd and sd > 0 else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    for ds in ("D1", "D2"):
        if f"auc_{ds}" in out:
            out[f"separation_{ds}"] = (out[f"auc_{ds}"] - 0.5).abs()
    return out


def prediction_behaviour(results_root: str | Path) -> pd.DataFrame:
    """Within-dataset saved results: is a poor AUC near-constant, reversed or non-discriminating?"""
    from ..visualization.results import index_results, read_csv

    try:
        idx = index_results(results_root, duplicates="latest")
    except Exception:                                  # noqa: BLE001  (no results yet)
        return pd.DataFrame()
    idx = idx[idx["setting"].isin(["within_D1", "within_D2"])]
    rows = []
    for _, r in idx.iterrows():
        try:
            s = read_csv(r, "predictions_subjects.csv")
        except FileNotFoundError:
            continue
        per = [(_auc(g["y"], g["score"]), float(g["score"].std()), float(g["score"].mean()))
               for _, g in s.groupby("repeat")]
        aucs = np.array([p[0] for p in per])
        mean_s = s.groupby("key").agg(y=("y", "first"), score=("score", "mean"))
        auc_avg = _auc(mean_s["y"], mean_s["score"])
        spread = np.median([p[1] for p in per])
        rows.append({"experiment": r["experiment"], "variant": r["variant"], "setting": r["setting"],
                     "auc_mean_over_repeats": float(np.nanmean(aucs)),
                     "fraction_repeats_auc_below_0_5": float(np.mean(aucs < 0.5)),
                     "auc_repeat_averaged_scores": auc_avg,
                     "median_within_repeat_score_sd": spread,
                     "score_range_repeat_averaged": float(mean_s["score"].max() - mean_s["score"].min()),
                     "score_type": r.get("score_type", "")})
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    def pattern(r):
        if r["auc_repeat_averaged_scores"] <= 0.3 and r["fraction_repeats_auc_below_0_5"] >= 0.8:
            return "consistently reversed ranking"
        if abs(r["auc_repeat_averaged_scores"] - 0.5) < 0.15:
            return "no discrimination"
        return "discriminates" if r["auc_repeat_averaged_scores"] > 0.5 else "weakly reversed"
    out["pattern"] = out.apply(pattern, axis=1)
    # near-constant: score spread within a repeat < 10 % of the same experiment/variant's spread in D2
    ref = out[out["setting"] == "within_D2"].set_index(["experiment", "variant"])["median_within_repeat_score_sd"]
    out["score_sd_ratio_to_D2"] = [r["median_within_repeat_score_sd"] / ref.get((r["experiment"], r["variant"]), np.nan)
                                   for _, r in out.iterrows()]
    out.loc[out["score_sd_ratio_to_D2"] < 0.1, "pattern"] += " (near-constant scores)"
    return out.sort_values(["experiment", "variant", "setting"]).reset_index(drop=True)


def direction_lines(sep: pd.DataFrame, min_effect: float = 0.15) -> list[str]:
    """Is the AD-vs-HC direction of each feature the same in D1 and D2 (both |AUC - 0.5| >= min_effect)?"""
    if not {"auc_D1", "auc_D2"} <= set(sep):
        return []
    a, b = sep["auc_D1"] - 0.5, sep["auc_D2"] - 0.5
    both = (a.abs() >= min_effect) & (b.abs() >= min_effect)
    same, opposite = both & (np.sign(a) == np.sign(b)), both & (np.sign(a) != np.sign(b))
    lines = ["", f"AD effect direction (features with |AUC - 0.5| >= {min_effect} in both datasets): "
             f"same {int(same.sum())}, opposite {int(opposite.sum())}"]
    if opposite.any():
        lines.append("opposite direction: " + ", ".join(sep.loc[opposite, "feature"]))
    return lines + [""]


def diagnose(store, results_root: str | Path | None, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    subj = subject_table(store)
    q = quality_summary(subj)
    ch = interpolated_channels(store, subj)
    feats = feature_table(store)
    sep = separability(subj, feats)
    pred = prediction_behaviour(results_root) if results_root else pd.DataFrame()
    q.to_csv(out / "quality_by_dataset_group.csv")
    ch.to_csv(out / "interpolated_channels.csv", index=False)
    sep.to_csv(out / "feature_separability_and_shift.csv", index=False)
    pred.to_csv(out / "prediction_behaviour.csv", index=False)

    lines = ["# D1 / D2 diagnostic (read-only; descriptive)", "",
             "Frozen graph run only; no EEG reprocessed, nothing excluded. Subject-level summaries.", ""]
    lines += ["## 1. Recording quality (median per dataset x group)", "", q.xs("median", axis=1, level=0)
              .round(2).to_string(), ""]
    pers = ch[ch["persistent"]] if len(ch) else ch
    lines += ["## 2. Channels bad in >= 30 % of a dataset's recordings", ""]
    if len(pers):
        lines.append(pers[["dataset", "channel", "n_bad", "n_recordings", "reasons", "all_bad_interpolated",
                           "min_final_std_uv_when_bad"]].to_string(index=False))
        handled = bool(pers["all_bad_interpolated"].all()) and bool(
            (pers["min_final_std_uv_when_bad"].fillna(1) > 0.1).all())
        lines += ["", "All of these were interpolated in every recording where they were flagged, and are non-flat "
                  "after interpolation." if handled else "WARNING: some flagged channels were not interpolated or "
                  "remain near-flat after interpolation; see interpolated_channels.csv.", ""]
    else:
        lines += ["none (or per-subject qc_channels.csv not available)", ""]
    if len(sep):
        top = lambda ds: sep.sort_values(f"separation_{ds}", ascending=False).head(5)[  # noqa: E731
            ["feature", f"auc_{ds}"]].round(3).to_string(index=False)
        lines += ["## 3. Class separability without a model (subject-level univariate AUC, AD positive)", "",
                  f"features with |AUC - 0.5| >= 0.25: D1 {int((sep.get('separation_D1', pd.Series(dtype=float)) >= 0.25).sum())}"
                  f" / D2 {int((sep.get('separation_D2', pd.Series(dtype=float)) >= 0.25).sum())} of {len(sep)}", "",
                  "D1 most separating:", top("D1") if "separation_D1" in sep else "n/a", "",
                  "D2 most separating:", top("D2") if "separation_D2" in sep else "n/a", "",
                  *direction_lines(sep),
                  "## 4. Dataset shift within group (|standardized mean difference| D1 vs D2, median over features)", "",
                  f"HC: {sep['smd_D1_minus_D2_HC'].abs().median():.2f}   AD: {sep['smd_D1_minus_D2_AD'].abs().median():.2f}", ""]
    if len(pred):
        lines += ["## 5. Prediction behaviour of saved within-dataset results", "",
                  pred[["experiment", "variant", "setting", "auc_repeat_averaged_scores",
                        "fraction_repeats_auc_below_0_5", "score_sd_ratio_to_D2", "pattern"]].round(3).to_string(index=False), ""]
    lines += ["Interpretation rules: 'no discrimination' = repeat-averaged AUC within 0.35-0.65; 'consistently "
              "reversed' = repeat-averaged AUC <= 0.30 with >= 80 % of repeats below 0.5; 'near-constant' = "
              "within-repeat score SD < 10 % of the same model's D2 value. With 20 subjects (10/10) a subject-level "
              "AUC below 0.5 is expected often under no signal; it is not by itself evidence of a defect."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return {"dir": out, "quality": q, "channels": ch, "separability": sep, "predictions": pred,
            "report": "\n".join(lines)}
