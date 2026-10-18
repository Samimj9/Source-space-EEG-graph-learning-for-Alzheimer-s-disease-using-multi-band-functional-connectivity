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


# ------------------------------------------------------------------------------------------- AUC uncertainty
def _delong_components(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DeLong placement values: V10 (one per positive), V01 (one per negative); AUC = mean(V10) = mean(V01)."""
    pos, neg = score[y == 1], score[y == 0]
    cmp = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    return cmp.mean(1), cmp.mean(0)


def auc_delong(y, score, level: float = 0.95) -> dict[str, float]:
    """ROC-AUC with its DeLong (1988) standard error and a normal-approximation CI clipped to [0, 1]; participant
    level (one score per participant). NaN when a class is missing."""
    y, score = np.asarray(y).astype(int), np.asarray(score, float)
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    v10, v01 = _delong_components(y, score)
    auc = float(v10.mean())
    var = (v10.var(ddof=1) / len(v10) if len(v10) > 1 else 0.0) + (v01.var(ddof=1) / len(v01) if len(v01) > 1 else 0.0)
    se = float(np.sqrt(var))
    z = stats.norm.ppf(0.5 + level / 2)
    return {"auc": auc, "se": se, "ci_low": float(max(0.0, auc - z * se)), "ci_high": float(min(1.0, auc + z * se))}


def delong_paired(y, score_a, score_b) -> dict[str, float]:
    """Paired DeLong test of two AUCs on the same participants (e.g. LCMV vs wMNE, mode A vs B). Descriptive in this
    study (no multiplicity correction is implied)."""
    y = np.asarray(y).astype(int)
    a10, a01 = _delong_components(y, np.asarray(score_a, float))
    b10, b01 = _delong_components(y, np.asarray(score_b, float))
    s10, s01 = np.cov(np.vstack([a10, b10])), np.cov(np.vstack([a01, b01]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / len(a10) + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / len(a01)
    diff = float(a10.mean() - b10.mean())
    se = float(np.sqrt(max(var, 0.0)))
    if se > 0:
        z = diff / se
        p = float(2 * stats.norm.sf(abs(z)))
    else:                                             # no variance: identical AUCs give p = 1, otherwise undefined
        z, p = (0.0, 1.0) if diff == 0 else (float("nan"), float("nan"))
    return {"auc_a": float(a10.mean()), "auc_b": float(b10.mean()), "diff": diff, "se": se, "z": float(z), "p": p}


def wilson_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of a proportion k / n (sensitivity, specificity)."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
