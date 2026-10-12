"""Window -> subject aggregation and binary classification metrics (DESIGN.md §4)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

METRICS = ("accuracy", "balanced_accuracy", "f1", "sensitivity", "specificity", "roc_auc", "mcc")


def aggregate_subjects(pred: pd.DataFrame, by: tuple[str, ...] = ()) -> pd.DataFrame:
    """Subject score = mean of that subject's window scores.

    Scores are probabilities (LR, GNNs, calibrated SVM) or SVM decision-function margins;
    the mean is taken on the model's own scale. ``pred`` needs columns key, dataset, y,
    score (+ any ``by`` columns, e.g. repeat).
    """
    cols = list(by) + ["key"]
    out = (pred.groupby(cols, sort=False)
           .agg(dataset=("dataset", "first"), y=("y", "first"), score=("score", "mean"),
                n_windows=("score", "size"))
           .reset_index())
    if (pred.groupby(cols)["y"].nunique() > 1).any():
        raise ValueError("label varies within a subject")
    return out


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return {"tn": int(((y == 0) & (pred == 0)).sum()), "fp": int(((y == 0) & (pred == 1)).sum()),
            "fn": int(((y == 1) & (pred == 0)).sum()), "tp": int(((y == 1) & (pred == 1)).sum())}


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, balanced accuracy, F1 (positive = AD), sensitivity, specificity, ROC-AUC, MCC and the
    confusion matrix (tn, fp, fn, tp). Undefined values (e.g. AUC with one class) are NaN.

    ``score`` is any monotone score for class 1 (probability or decision value); ROC-AUC uses it
    directly, the threshold-based metrics use ``score >= threshold`` (0.5 for probabilities,
    0 for SVM margins)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    c = confusion(y, pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]

    def div(a, b):
        return a / b if b else float("nan")

    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    prec = div(tp, tp + fp)
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    out = {
        "n": len(y),
        "accuracy": div(tp + tn, len(y)),
        "balanced_accuracy": np.nanmean([sens, spec]) if not (np.isnan(sens) and np.isnan(spec)) else float("nan"),
        "f1": div(2 * prec * sens, prec + sens) if not (np.isnan(prec) or np.isnan(sens)) else float("nan"),
        "sensitivity": sens,
        "specificity": spec,
        "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else float("nan"),
        "mcc": div(tp * tn - fp * fn, denom),
        **c,
    }
    return out


def bootstrap_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 2000, seed: int = 0, threshold: float = 0.5,
                 level: float = 0.95, metrics=METRICS) -> dict[str, tuple[float, float]]:
    """Percentile CI by resampling *subjects* (one row per subject) with replacement."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    draws = {m: [] for m in metrics}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m = binary_metrics(y[i], score[i], threshold)
        for k in metrics:
            draws[k].append(m[k])
    a = (1 - level) / 2
    with warnings.catch_warnings():                      # metrics undefined in some draws -> NaN, ignored
        warnings.simplefilter("ignore", RuntimeWarning)
        return {k: (float(np.nanquantile(v, a)), float(np.nanquantile(v, 1 - a))) if v else (np.nan, np.nan)
                for k, v in draws.items()}


def summarize_repeats(per_repeat: pd.DataFrame, metrics=METRICS) -> dict[str, dict[str, float]]:
    """Mean, SD and 2.5/97.5 percentiles of each metric across CV repeats."""
    out = {}
    with warnings.catch_warnings():                      # all-NaN metrics (e.g. AUC of one class) stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        for m in metrics:
            v = per_repeat[m].to_numpy(float)
            out[m] = {"mean": float(np.nanmean(v)),
                      "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"),
                      "p2.5": float(np.nanpercentile(v, 2.5)), "p97.5": float(np.nanpercentile(v, 97.5))}
    return out


def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> dict[str, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test for paired CV differences.

    ``diffs``: per-fold (or per-repeat) metric differences of two models on identical splits;
    variance inflated by (1/J + n_test/n_train).
    """
    d = np.asarray(diffs, float)
    j = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / j + n_test / n_train) * var) if var > 0 else float("nan")
    p = 2 * stats.t.sf(abs(t), df=j - 1) if np.isfinite(t) else float("nan")
    return {"mean_diff": float(d.mean()), "t": float(t), "p": float(p), "df": j - 1}


def permutation_p_value(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, float)
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))
