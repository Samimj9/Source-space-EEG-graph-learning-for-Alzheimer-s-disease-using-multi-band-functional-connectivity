"""Subject-level prediction figures (families 8, 9, 11).

Input: ``predictions_subjects.csv`` (subject score per CV repeat = mean of that subject's
window scores). Each subject's score is averaged over repeats, giving exactly **one
point per subject**, the same quantity ``summary.json`` uses for its bootstrap CI.
Windows are never plotted as observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from .results import MissingInputs, anonymize, select_unique, subject_scores
from .style import GROUP_STYLE, OKABE_ITO, SETTING_LABEL, condition_style, figure_size, marker_kwargs

FPR_GRID = np.linspace(0, 1, 101)


def _entry_rows(index, spec):
    out = []
    for e in spec["entries"]:
        sel = select_unique(index, e, [spec["setting"]])
        for _, r in sel.iterrows():
            out.append((e["label"], r))
    if not out:
        raise MissingInputs("no saved results matched the figure entries")
    return out


def bootstrap_roc_band(y, s, n_boot, seed=0, level=0.95):
    """Pointwise percentile band of TPR on a fixed FPR grid, resampling subjects."""
    rng = np.random.default_rng(seed)
    tprs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[i], s[i])
        tprs.append(np.interp(FPR_GRID, fpr, tpr))
    a = (1 - level) / 2
    return np.quantile(tprs, a, axis=0), np.quantile(tprs, 1 - a, axis=0)


# ------------------------------------------------------------------ family 8
def fig_roc(index, spec, style, stats=None):
    rows, data = _entry_rows(index, spec), []
    fig, ax = plt.subplots(figsize=figure_size(style, spec.get("width", "single"), height_ratio=0.9))
    linestyles = ["-", "--", "-.", ":"]
    for k, (label, r) in enumerate(rows):
        s = subject_scores(r)
        y, sc = s["y"].to_numpy(int), s["score"].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, sc)
        auc = roc_auc_score(y, sc)
        spec_c = condition_style(r["condition"])
        name = f"{label} [{spec_c['label']}]" if r["condition"] != "none" else label
        ax.plot(fpr, tpr, color=spec_c["color"], ls=linestyles[k % 4], lw=style["line_width"] * 1.2,
                label=f"{name}: AUC {auc:.2f} (n={len(y)})")
        data += [{"model": name, "fpr": a, "tpr": b, "kind": "curve", "auc": auc, "n_subjects": len(y)}
                 for a, b in zip(fpr, tpr)]
        if spec.get("bootstrap_band", 0):
            lo, hi = bootstrap_roc_band(y, sc, spec["bootstrap_band"])
            ax.fill_between(FPR_GRID, lo, hi, color=spec_c["color"], alpha=0.15, lw=0)
            data += [{"model": name, "fpr": f, "tpr_lo": a, "tpr_hi": b, "kind": "bootstrap_band_95"}
                     for f, a, b in zip(FPR_GRID, lo, hi)]
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(SETTING_LABEL.get(spec["setting"], spec["setting"]))
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=5)
    fig.tight_layout()
    meta = {"kind": "descriptive", "statistical_unit": "subject (score averaged over CV repeats)",
            "notes": "ROC from one repeat-averaged score per subject" +
                     (f"; band = pointwise 95% subject bootstrap ({spec['bootstrap_band']} resamples)"
                      if spec.get("bootstrap_band") else "")}
    return fig, pd.DataFrame(data), meta, [r["path"] for _, r in rows]


# ------------------------------------------------------------------ family 9
def fig_confusion(index, spec, style, stats=None):
    rows = _entry_rows(index, spec)
    fig, axes = plt.subplots(2, len(rows), squeeze=False,
                             figsize=figure_size(style, spec.get("width", "double"),
                                                 height_mm=spec.get("height_mm", 75)))
    data = []
    for j, (label, r) in enumerate(rows):
        s = subject_scores(r)
        pred = (s["score"] >= r["score_threshold"]).astype(int)
        cm = np.array([[((s["y"] == t) & (pred == p)).sum() for p in (0, 1)] for t in (0, 1)])
        norm = cm / cm.sum(axis=1, keepdims=True)
        name = label + ("" if r["condition"] == "none" else f" [{condition_style(r['condition'])['label']}]")
        for i, (mat, fmt, title) in enumerate(((cm, "d", "counts"), (norm, ".2f", "row-normalized"))):
            ax = axes[i, j]
            ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, format(mat[a, b], fmt), ha="center", va="center",
                            color="white" if norm[a, b] > 0.6 else "black", fontsize=7)
            ax.set_xticks([0, 1], ["HC", "AD"])
            ax.set_yticks([0, 1], ["HC", "AD"])
            ax.set_xlabel("Predicted")
            if j == 0:
                ax.set_ylabel(f"True ({title})")
            if i == 0:
                ax.set_title(name, fontsize=6)
        for t, tl in enumerate(("HC", "AD")):
            for p, pl in enumerate(("HC", "AD")):
                data.append({"model": name, "true": tl, "pred": pl, "count": int(cm[t, p]),
                             "row_fraction": norm[t, p], "threshold": r["score_threshold"],
                             "score_type": r["score_type"]})
    fig.tight_layout()
    meta = {"kind": "descriptive", "statistical_unit": "subject (repeat-averaged score, fixed threshold)",
            "notes": "threshold fixed on the model's score scale (0.5 probability, 0 SVM margin); never tuned"}
    return fig, pd.DataFrame(data), meta, [r["path"] for _, r in rows]


# ------------------------------------------------------------------ family 11
def fig_subject_scores(index, spec, style, stats=None):
    rows = _entry_rows(index, spec)
    fig, axes = plt.subplots(1, len(rows), squeeze=False,
                             figsize=figure_size(style, spec.get("width", "double"),
                                                 height_mm=spec.get("height_mm", 60)))
    rng = np.random.default_rng(0)
    data = []
    for j, (label, r) in enumerate(rows):
        s = subject_scores(r)
        anon = anonymize(s["key"])
        ax = axes[0, j]
        for xi, (grp, yval) in enumerate((("HC", 0), ("AD", 1))):
            d = s[s["y"] == yval]
            x = xi + rng.uniform(-0.15, 0.15, len(d))
            g = GROUP_STYLE[grp]
            ax.errorbar(x, d["score"], yerr=[d["score"] - d["score_min"], d["score_max"] - d["score"]],
                        linestyle="none", elinewidth=0.4, capsize=0, color="0.6", zorder=1)
            ax.plot(x, d["score"], linestyle="none", zorder=2, **marker_kwargs(g))
            ax.boxplot(d["score"], positions=[xi], widths=0.5, showfliers=False, medianprops={"color": "k"},
                       boxprops={"linewidth": 0.5}, whiskerprops={"linewidth": 0.5}, capprops={"linewidth": 0.5})
            if spec.get("label_subjects", False):
                for xx, (_, row) in zip(x, d.iterrows()):
                    ax.annotate(anon[row["key"]], (xx, row["score"]), xytext=(2, 0), textcoords="offset points",
                                fontsize=4)
            for xx, (_, row) in zip(x, d.iterrows()):
                data.append({"model": label, "condition": r["condition"], "subject": anon[row["key"]],
                             "dataset": row["dataset"], "true_group": grp, "score": row["score"],
                             "score_min_over_repeats": row["score_min"], "score_max_over_repeats": row["score_max"],
                             "n_repeats": row["n_repeats"], "threshold": r["score_threshold"]})
        ax.axhline(r["score_threshold"], color=OKABE_ITO["black"], lw=0.6, ls="--")
        ax.set_xticks([0, 1], ["HC", "AD"])
        ax.set_title(label + ("" if r["condition"] == "none" else f" [{condition_style(r['condition'])['label']}]"),
                     fontsize=6)
        if j == 0:
            ax.set_ylabel("Subject score" + (" (SVM margin)" if r["score_type"] == "decision_function"
                                             else " (P(AD))"))
    fig.tight_layout()
    meta = {"kind": "descriptive", "statistical_unit": "subject (one point per subject; bars = min-max over repeats)",
            "notes": "anonymized IDs S01... (mapping not saved); windows are never plotted as observations"}
    return fig, pd.DataFrame(data), meta, [r["path"] for _, r in rows]
