"""Single-experiment report: results table + diagnostic figures, from saved outputs only.

Written to ``<results>/<experiment>/_reports/report_<UTC time>/`` (never overwritten):

* ``results_summary.csv``: one row per completed (variant, setting): subject-level
  metrics (mean ± SD over CV repeats, subject-bootstrap 95 % CI of the repeat-averaged
  scores), window-level AUC, number of subjects/repeats, score type and threshold.
* ``progress.csv``: every (variant, setting) of the experiment with its checkpoint state.
* diagnostic figures (``D1_performance``, ``D2_roc_<setting>``, ``D3_confusion_<setting>``,
  ``D4_subject_scores_<setting>``), each with its data, metadata and a manifest.

These are **diagnostics of one experiment** for checking it before choosing the next
one. They are not the comparative paper figures (``configs/figures/paper.yaml``, notebook
``03_comparative_figures``), and no model is selected or ranked here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..evaluation.splits import CROSS_SETTINGS
from ..visualization.figures import make_figures
from ..visualization.results import index_results, read_manifest, read_summary

SETTING_ORDER = ["within_D1", "within_D2", "pooled", "cross_D1_to_D2", "cross_D2_to_D1", "within_D3"]
CHUNK = 6                                   # variants per ROC / confusion / score figure
METRICS = ["balanced_accuracy", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]
README = """Single-experiment diagnostics ({experiment})

Generated from saved outputs only, for checking this experiment before starting the next.
These are NOT the comparative paper figures (notebook 03_comparative_figures). No model is
selected or ranked here. Subject-level metrics are primary; values are mean ± SD over CV
repeats, with the subject-bootstrap 95 % CI of the repeat-averaged subject scores.
Settings missing from a figure have not finished yet (see progress.csv).
"""


def results_table(index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in index.iterrows():
        s, m = read_summary(r), read_manifest(r)
        row = {"variant": r["variant"], "setting": r["setting"], "n_subjects": s.get("n_subjects"),
               "n_repeats": s.get("n_repeats"), "score": f"{s['score']['type']} > {s['score']['threshold']}"}
        for k in METRICS:
            stat = s["subject_level"]["all"].get(k, {})
            row[f"{k}_mean"], row[f"{k}_sd"] = stat.get("mean"), stat.get("sd")
            ci = s.get("bootstrap_ci_95", {}).get(k)
            if ci and k in ("balanced_accuracy", "roc_auc"):
                row[f"{k}_ci95"] = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        row["window_roc_auc_mean"] = s.get("window_level", {}).get("roc_auc", {}).get("mean")
        if "permutation" in s:
            row["permutation_p"] = s["permutation"]["p_value"]
        row["graph_run_id"], row["git_commit"] = r["graph_run_id"], m.get("git_commit", "")
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["variant", "setting", "n_subjects", "n_repeats"])
    df["_o"] = df["setting"].map({s: i for i, s in enumerate(SETTING_ORDER)})
    return df.sort_values(["_o", "variant"], kind="stable").drop(columns="_o").reset_index(drop=True)


def figure_config(experiment: str, index: pd.DataFrame) -> dict:
    variants = sorted(set(index["variant"]))
    settings = [s for s in SETTING_ORDER if s in set(index["setting"])]
    entry = lambda v: {"label": v, "experiment": experiment, "where": {"variant": v}}  # noqa: E731
    figs = [{"id": "D1_performance", "family": "performance", "status": "diagnostic", "width": "double",
             "settings": settings, "metrics": ["balanced_accuracy", "roc_auc"],
             "entries": [entry(v) for v in variants]}]
    for s in settings:
        present = sorted(set(index.loc[index["setting"] == s, "variant"]))
        chunks = [present[i:i + CHUNK] for i in range(0, len(present), CHUNK)]
        for c, vs in enumerate(chunks):
            sfx = f"_{s}" + (f"_part{c + 1}" if len(chunks) > 1 else "")
            ents = [entry(v) for v in vs]
            if s not in CROSS_SETTINGS:
                figs.append({"id": f"D2_roc{sfx}", "family": "roc", "status": "diagnostic", "setting": s,
                             "bootstrap_band": 200, "entries": ents})
            figs.append({"id": f"D3_confusion{sfx}", "family": "confusion", "status": "diagnostic", "setting": s,
                         "entries": ents})
            figs.append({"id": f"D4_subject_scores{sfx}", "family": "subject_scores", "status": "diagnostic",
                         "setting": s, "entries": ents})
    return {"figures": figs, "duplicates": "error"}


def experiment_report(results_root: str | Path, experiment: str, progress: pd.DataFrame | None = None,
                      style_path: str | Path | None = "configs/figures/style.yaml") -> dict:
    """Write the report for one experiment; returns paths and the results table."""
    root = Path(results_root) / experiment                 # this experiment's results only
    index = index_results(root) if root.exists() else pd.DataFrame(columns=["experiment", "variant", "setting"])
    index = index[index["experiment"] == experiment].reset_index(drop=True)
    out = root / "_reports" / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "README.txt").write_text(README.format(experiment=experiment))
    table = results_table(index)
    table.to_csv(out / "results_summary.csv", index=False)
    if progress is not None:
        progress.to_csv(out / "progress.csv", index=False)
    figures, skipped = [], []
    if len(index):
        synthetic = any(read_manifest(r).get("graph_run", {}).get("config_hash") == "synthetic"
                        for _, r in index.iterrows())
        w = make_figures(figure_config(experiment, index), out / "figures", root, None,
                         style_path=style_path if style_path and Path(style_path).exists() else None,
                         synthetic=synthetic)
        figures = [f["id"] for f in w.figures]
        skipped = w.skipped
    return {"report_dir": out, "table": table, "figures": figures, "skipped": skipped}
