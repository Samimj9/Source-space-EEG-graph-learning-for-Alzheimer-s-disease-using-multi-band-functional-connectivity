"""Cross-experiment comparison tables and figures from saved outputs only (Experiment 7 report).

Nothing is trained or re-evaluated. Every number comes from a validated result directory
(``summary.json``, ``metrics_repeats.csv``, ``hparams.csv``) of one graph run. Comparisons are
descriptive: per-repeat paired differences on the repeats two results share (identical subject
splits for repeat r in every experiment); no significance test is computed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..visualization.results import index_results, matches, read_csv, read_manifest, read_summary

SETTINGS = ["within_D1", "within_D2", "pooled", "cross_D1_to_D2", "cross_D2_to_D1"]
METRICS = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc"]

# (label, experiment, where) -- PLV and |ciPLV| of graph models are separate rows (condition column)
MODELS = [
    ("EARM-GNN full", "exp7_earm_gnn", {"model.adaptive_adjacency": True, "model.fusion": "attention"}),
    ("EARM-GNN fixed adjacency (abl. A)", "exp7_earm_gnn", {"model.adaptive_adjacency": False}),
    ("EARM-GNN mean fusion (abl. B)", "exp7_earm_gnn", {"model.adaptive_adjacency": True, "model.fusion": "mean"}),
    ("Exp6 GCN full", "exp6_multiband_psd_strength_fixedhp",
     {"node_features.components": ["psd", "strength"], "model.fusion": "attention",
      "graph.bands": ["theta", "alpha1", "alpha2", "beta"]}),
    ("Exp5 GCN attention", "exp5_multiband_fixedhp", {"model.fusion": "attention"}),
    ("Exp5 GCN mean", "exp5_multiband_fixedhp", {"model.fusion": "mean"}),
    ("Exp4 GCN band-mean", "exp4_band_mean_fixedhp", {}),
    ("Exp1 source PSD SVM", "exp1_source_psd", {"model.name": "svm"}),
    ("Exp1 source PSD LR", "exp1_source_psd", {"model.name": "logreg"}),
    ("Exp0 sensor PSD SVM", "exp0_sensor_psd", {"model.name": "svm"}),
    ("Exp0 sensor PSD LR", "exp0_sensor_psd", {"model.name": "logreg"}),
]
REFERENCE = "EARM-GNN full"

# Dataset 3 (OpenNeuro ds004504; methodology log entry 44): within-D3 only, its own experiments
SETTINGS_D3 = ["within_D3"]
MODELS_D3 = [
    ("EARM-GNN full", "d3_exp7_earm_gnn", {"model.adaptive_adjacency": True, "model.fusion": "attention"}),
    ("EARM-GNN fixed adjacency (abl. A)", "d3_exp7_earm_gnn", {"model.adaptive_adjacency": False}),
    ("EARM-GNN mean fusion (abl. B)", "d3_exp7_earm_gnn", {"model.adaptive_adjacency": True, "model.fusion": "mean"}),
    ("Exp6 GCN full", "d3_exp6_gcn_full", {}),
    ("Exp1 source PSD SVM", "d3_exp1_source_psd", {"model.name": "svm"}),
    ("Exp1 source PSD LR", "d3_exp1_source_psd", {"model.name": "logreg"}),
    ("Exp0 sensor PSD SVM", "d3_exp0_sensor_psd", {"model.name": "svm"}),
    ("Exp0 sensor PSD LR", "d3_exp0_sensor_psd", {"model.name": "logreg"}),
    ("Age only", "d3_baseline_age", {}),
]


def family_models(family: str | None) -> tuple[list, list]:
    return (MODELS_D3, SETTINGS_D3) if family == "D3" else (MODELS, SETTINGS)


def model_rows(index: pd.DataFrame, models: list | None = None) -> pd.DataFrame:
    """The saved results of ``models`` (default MODELS; one row per label, condition, setting)."""
    rows = []
    for label, exp, where in models or MODELS:
        sel = index[(index["experiment"] == exp)]
        sel = sel[[matches(c, where) for c in sel["config"]]] if len(sel) else sel
        for _, r in sel.iterrows():
            rows.append({**r.to_dict(), "label": label})
    out = pd.DataFrame(rows)
    if len(out) and out.duplicated(["label", "condition", "setting"]).any():
        dup = out[out.duplicated(["label", "condition", "setting"], keep=False)]
        raise ValueError(f"several variants match one model label:\n{dup[['label', 'variant', 'setting']]}")
    return out


def comparison_table(rows: pd.DataFrame, models: list | None = None,
                     settings: list | None = None) -> pd.DataFrame:
    out = []
    for _, r in rows.iterrows():
        s = read_summary(r)
        rec = {"model": r["label"], "experiment": r["experiment"], "variant": r["variant"],
               "connectivity": r["condition"], "setting": r["setting"], "n_subjects": s.get("n_subjects"),
               "n_repeats": s.get("n_repeats")}
        for k in METRICS:
            st = s["subject_level"]["all"].get(k, {})
            rec[f"{k}_mean"], rec[f"{k}_sd"] = st.get("mean"), st.get("sd")
        for k in ("roc_auc", "balanced_accuracy"):
            ci = s.get("bootstrap_ci_95", {}).get(k)
            rec[f"{k}_ci95_low"], rec[f"{k}_ci95_high"] = (ci[0], ci[1]) if ci else (np.nan, np.nan)
        try:
            hp = read_csv(r, "hparams.csv")
            rec["n_parameters"] = float(hp["n_parameters"].median()) if "n_parameters" in hp else np.nan
            rec["fit_seconds_mean"] = float(hp["fit_seconds"].mean()) if "fit_seconds" in hp else np.nan
            rec["fit_seconds_total"] = float(hp["fit_seconds"].sum()) if "fit_seconds" in hp else np.nan
        except FileNotFoundError:
            pass
        rec["git_commit"] = r.get("git_commit", "")
        out.append(rec)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    order = {m[0]: i for i, m in enumerate(models or MODELS)}
    df["_m"], df["_s"] = df["model"].map(order), df["setting"].map({s: i for i, s in enumerate(settings or SETTINGS)})
    return df.sort_values(["_s", "_m", "connectivity"]).drop(columns=["_m", "_s"]).reset_index(drop=True)


def _repeats(r: pd.Series, metric: str) -> pd.Series:
    m = read_csv(r, "metrics_repeats.csv")
    return m[m["scope"] == "all"].set_index("repeat")[metric]


def paired_differences(rows: pd.DataFrame, metric: str = "roc_auc", reference: str = REFERENCE) -> pd.DataFrame:
    """reference minus comparator, per repeat shared by both (same subject splits), per setting and
    connectivity. Classical baselines (no connectivity) are compared with both PLV and |ciPLV|."""
    out = []
    ref = rows[rows["label"] == reference]
    for _, a in ref.iterrows():
        comp = rows[(rows["label"] != reference) & (rows["setting"] == a["setting"])
                    & rows["condition"].isin([a["condition"], "none"])]
        ra = _repeats(a, metric)
        for _, b in comp.iterrows():
            rb = _repeats(b, metric)
            common = ra.index.intersection(rb.index)
            if not len(common):
                continue
            d = (ra.loc[common] - rb.loc[common]).astype(float)
            out.append({"setting": a["setting"], "connectivity": a["condition"], "reference": reference,
                        "comparator": b["label"], "metric": metric, "n_shared_repeats": int(len(common)),
                        "reference_mean": float(ra.loc[common].mean()), "comparator_mean": float(rb.loc[common].mean()),
                        "difference_mean": float(d.mean()), "difference_sd": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
                        "fraction_repeats_reference_higher": float((d > 0).mean())})
    return pd.DataFrame(out)


def auc_pivot(table: pd.DataFrame, settings: list | None = None) -> pd.DataFrame:
    if table.empty:
        return table
    t = table.assign(row=table["model"] + " [" + table["connectivity"] + "]")
    return t.pivot_table(index="row", columns="setting", values="roc_auc_mean", sort=False).reindex(
        columns=[s for s in settings or SETTINGS if s in set(t["setting"])])


def figure_config_d3() -> dict:
    """Dataset 3: ROC, confusion and performance figures (within-D3 only)."""
    def ent(label, conditions=None):
        exp, where = next((e, w) for lab, e, w in MODELS_D3 if lab == label)
        return {"label": label, "experiment": exp, "where": where, **({"conditions": conditions} if conditions else {})}

    figs = []
    for cond in ("plv", "ciplv"):
        graph = [ent(x, [cond]) for x in ("EARM-GNN full", "EARM-GNN fixed adjacency (abl. A)",
                                          "EARM-GNN mean fusion (abl. B)", "Exp6 GCN full")]
        base = [ent("Exp1 source PSD SVM"), ent("Exp0 sensor PSD SVM"), ent("Age only")]
        figs += [{"id": f"D3_roc_{cond}", "family": "roc", "status": "diagnostic", "setting": "within_D3",
                  "bootstrap_band": 200, "entries": graph[:1] + graph[3:] + base},
                 {"id": f"D3_confusion_{cond}", "family": "confusion", "status": "diagnostic", "setting": "within_D3",
                  "entries": graph[:1] + graph[3:] + base[:2]},
                 {"id": f"D3_performance_{cond}", "family": "performance", "status": "diagnostic", "width": "double",
                  "settings": SETTINGS_D3, "metrics": ["roc_auc", "balanced_accuracy"], "entries": graph + base}]
    return {"figures": figs, "duplicates": "latest"}


def figure_config() -> dict:
    """ROC, confusion and performance figures: EARM-GNN (and ablations) vs GCN and PSD baselines."""
    def ent(label, conditions=None):
        exp, where = next((e, w) for lab, e, w in MODELS if lab == label)
        return {"label": label, "experiment": exp, "where": where, **({"conditions": conditions} if conditions else {})}

    figs = []
    for cond in ("plv", "ciplv"):
        graph = [ent(x, [cond]) for x in ("EARM-GNN full", "EARM-GNN fixed adjacency (abl. A)",
                                          "EARM-GNN mean fusion (abl. B)", "Exp6 GCN full", "Exp4 GCN band-mean")]
        base = [ent("Exp1 source PSD SVM"), ent("Exp0 sensor PSD SVM")]
        for s in ("within_D1", "within_D2", "pooled"):
            figs.append({"id": f"E7_roc_{s}_{cond}", "family": "roc", "status": "diagnostic", "setting": s,
                         "bootstrap_band": 200, "entries": graph[:1] + graph[3:] + base})
        for s in SETTINGS:
            figs.append({"id": f"E7_confusion_{s}_{cond}", "family": "confusion", "status": "diagnostic",
                         "setting": s, "entries": graph[:1] + graph[3:4] + base})
        figs.append({"id": f"E7_performance_{cond}", "family": "performance", "status": "diagnostic",
                     "width": "double", "settings": SETTINGS, "metrics": ["roc_auc", "balanced_accuracy"],
                     "entries": graph + base})
    return {"figures": figs, "duplicates": "latest"}


def write_comparison(results_root: str | Path, out_dir: str | Path, style_path: str | Path | None = None,
                     family: str | None = None) -> dict:
    """Tables (+ figures) under ``out_dir``; returns the tables. ``family`` D3: Dataset 3 experiments only."""
    from ..visualization.figures import make_figures

    models, settings = family_models(family)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    index = index_results(results_root, duplicates="latest")
    rows = model_rows(index, models)
    table = comparison_table(rows, models, settings) if len(rows) else pd.DataFrame()
    pivot = auc_pivot(table, settings)
    diffs = pd.concat([paired_differences(rows, m) for m in ("roc_auc", "balanced_accuracy")],
                      ignore_index=True) if len(rows) else pd.DataFrame()
    table.to_csv(out / "comparison_table.csv", index=False)
    pivot.to_csv(out / "auc_by_setting.csv")
    diffs.to_csv(out / "paired_differences.csv", index=False)
    (out / "README.txt").write_text(
        "Experiment 7 comparison (saved outputs only; nothing retrained).\n"
        "comparison_table.csv: subject-level metrics (mean, SD over CV repeats; bootstrap 95% CI over subjects),\n"
        "  parameter count and fit time per outer fold.\n"
        "paired_differences.csv: EARM-GNN full minus comparator, per repeat shared by both results (same subject\n"
        "  splits). DESCRIPTIVE: no significance test. Cross-dataset settings have one repeat.\n"
        "EARM-GNN is an exploratory extension developed after the Tier 1 GCN results were observed.\n")
    figures, skipped = [], []
    exp7 = "d3_exp7_earm_gnn" if family == "D3" else "exp7_earm_gnn"
    if len(rows) and (rows["experiment"] == exp7).any():
        w = make_figures(figure_config_d3() if family == "D3" else figure_config(), out / "figures", results_root, None,
                         style_path=style_path if style_path and Path(style_path).exists() else None,
                         synthetic=any(read_manifest(r).get("graph_run", {}).get("config_hash") == "synthetic"
                                       for _, r in rows.iterrows()))
        figures, skipped = [f["id"] for f in w.figures], w.skipped
    return {"dir": out, "table": table, "pivot": pivot, "differences": diffs, "figures": figures,
            "skipped": skipped}
