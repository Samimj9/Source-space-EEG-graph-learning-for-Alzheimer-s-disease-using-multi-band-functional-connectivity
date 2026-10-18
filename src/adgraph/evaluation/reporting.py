"""Participant-level evaluation of one (variant, setting) from its window predictions (docs/final/framework.md §6).

Primary metric (development CV): ROC-AUC of the participant scores pooled over the five outer folds (one out-of-fold
score per participant). Reported with it:

* mean (SD, min, max) of the five within-fold participant AUCs: pooling ranks scores of five separately trained and
  calibrated fold models together and can differ from the mean fold AUC; both are shown, neither replaces the other;
* balanced accuracy, sensitivity (AD), specificity (HC), F1, MCC, accuracy at the model's fixed threshold
  (0.5 for probabilities, 0 for SVM margins) and the confusion matrix (tn, fp, fn, tp; AD = positive);
* uncertainty: DeLong 95 % CI of the AUC, participant bootstrap 95 % CIs (AUC, balanced accuracy, sensitivity,
  specificity), Wilson 95 % CIs of sensitivity and specificity;
* window-level AUC and balanced accuracy (secondary; windows of one participant are not independent).

Holdout (``xu_holdout``): the same participant metrics on the 13 holdout participants (no folds). With more than one
repeat, metrics are computed per repeat and averaged; the CIs use the repeat-averaged participant scores.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .metrics import aggregate_subjects, auc_delong, binary_metrics, bootstrap_ci, wilson_ci

CORE = ("roc_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc", "accuracy")


def participant_predictions(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per (repeat, participant): y, score (mean window score), n_windows, fold, predicted label, correct."""
    subj = aggregate_subjects(pred, by=("repeat",))
    folds = pred.groupby(["repeat", "key"])["fold"].first().rename("fold")
    subj = subj.merge(folds.reset_index(), on=["repeat", "key"], how="left")
    subj["predicted"] = (subj["score"] >= threshold).astype(int)
    subj["correct"] = subj["predicted"] == subj["y"]
    subj["label"] = np.where(subj["y"] == 1, "AD", "HC")
    return subj


def fold_metrics(subj: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (r, f), g in subj.groupby(["repeat", "fold"]):
        m = binary_metrics(g["y"], g["score"], threshold)
        rows.append({"repeat": r, "fold": f, "n": len(g), "n_ad": int(g["y"].sum()),
                     **{k: m[k] for k in CORE}})
    return pd.DataFrame(rows)


def evaluate_setting(pred: pd.DataFrame, setting: str, threshold: float, n_boot: int = 2000,
                     seed: int = 0) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """(metrics dict, participant predictions, fold metrics)."""
    subj = participant_predictions(pred, threshold)
    per_rep = []
    for r, g in subj.groupby("repeat"):
        per_rep.append({"repeat": int(r), **binary_metrics(g["y"], g["score"], threshold)})
    per_rep = pd.DataFrame(per_rep)
    mean_subj = subj.groupby("key").agg(y=("y", "first"), score=("score", "mean"))
    y, s = mean_subj["y"].to_numpy(), mean_subj["score"].to_numpy()
    avg = binary_metrics(y, s, threshold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        participant = {k: float(np.nanmean(per_rep[k])) for k in CORE}
    out = {"setting": setting, "threshold": threshold, "n_participants": int(len(mean_subj)),
           "n_ad": int(y.sum()), "n_hc": int((y == 0).sum()), "n_windows": int(pred["window_id"].nunique()),
           "n_repeats": int(subj["repeat"].nunique()),
           "participant": participant,
           "confusion": {k: int(avg[k]) for k in ("tn", "fp", "fn", "tp")},
           "participant_auc_delong": auc_delong(y, s),
           "participant_bootstrap_ci95": {k: list(v) for k, v in bootstrap_ci(
               y, s, n_boot, seed, threshold, metrics=("roc_auc", "balanced_accuracy", "sensitivity",
                                                       "specificity")).items()},
           "sensitivity_wilson_ci95": list(wilson_ci(int(avg["tp"]), int(avg["tp"] + avg["fn"]))),
           "specificity_wilson_ci95": list(wilson_ci(int(avg["tn"]), int(avg["tn"] + avg["fp"]))),
           "score_distribution": {"mean_ad": float(s[y == 1].mean()) if (y == 1).any() else float("nan"),
                                  "mean_hc": float(s[y == 0].mean()) if (y == 0).any() else float("nan"),
                                  "sd": float(s.std()), "n_unique": int(np.unique(np.round(s, 7)).size)}}
    win = [binary_metrics(g["y"], g["score"], threshold) for _, g in pred.groupby("repeat")]
    out["window"] = {k: float(np.nanmean([w[k] for w in win])) for k in ("roc_auc", "balanced_accuracy")}
    folds = fold_metrics(subj, threshold) if setting == "xu_dev_cv" else pd.DataFrame()
    if len(folds):
        a = folds["roc_auc"].to_numpy(float)
        out["folds"] = {"n_folds": int(len(folds)), "mean_auc": float(np.nanmean(a)),
                        "sd_auc": float(np.nanstd(a, ddof=1)) if len(a) > 1 else float("nan"),
                        "min_auc": float(np.nanmin(a)), "max_auc": float(np.nanmax(a)),
                        "mean_balanced_accuracy": float(np.nanmean(folds["balanced_accuracy"])),
                        "per_fold": folds.to_dict("records")}
    return out, subj, folds


def summary_row(metrics: dict) -> dict:
    """Flat row for the combined result tables."""
    p, d, b = metrics["participant"], metrics["participant_auc_delong"], metrics["participant_bootstrap_ci95"]
    row = {"n_participants": metrics["n_participants"], "auc": p["roc_auc"], "auc_ci_low": d["ci_low"],
           "auc_ci_high": d["ci_high"], "auc_boot_low": b["roc_auc"][0], "auc_boot_high": b["roc_auc"][1],
           "balanced_accuracy": p["balanced_accuracy"], "sensitivity": p["sensitivity"],
           "specificity": p["specificity"], "f1": p["f1"], "mcc": p["mcc"], **metrics["confusion"],
           "window_auc": metrics["window"]["roc_auc"], "score_sd": metrics["score_distribution"]["sd"],
           "n_unique_scores": metrics["score_distribution"]["n_unique"]}
    if "folds" in metrics:
        f = metrics["folds"]
        row.update({"mean_fold_auc": f["mean_auc"], "sd_fold_auc": f["sd_auc"], "min_fold_auc": f["min_auc"],
                    "max_fold_auc": f["max_auc"]})
    return row
