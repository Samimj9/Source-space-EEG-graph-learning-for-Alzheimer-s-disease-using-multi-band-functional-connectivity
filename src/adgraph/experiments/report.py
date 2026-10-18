"""Result tables, comparisons and diagnostic figures, from saved outputs only (nothing is trained).

* ``collect``: one row per completed (context, experiment, variant, setting) with ``evaluation.reporting.summary_row``;
* ``experiment_report``: the table of one experiment in one context plus diagnostic figures (participant ROC curves,
  confusion matrices, fold AUCs, training-loss curves), written to a new time-stamped directory;
* ``compare_inverses`` / ``compare_modes``: matched LCMV vs wMNE and mode A vs B pairs (same experiment, variant,
  setting, atlas, graph construction and participants) with the paired DeLong test of the participant AUCs
  (descriptive; the comparisons are not corrected for multiplicity);
* ``development_only``: the rows a selection may use (development CV only; the holdout is never used to select).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..checkpoint import atomic_write_csv, atomic_write_text
from ..evaluation.metrics import delong_paired
from ..evaluation.reporting import summary_row

CONTEXT = ["inverse", "mode", "atlas", "graph"]
KEY = CONTEXT + ["experiment", "variant", "setting"]


def completed_dirs(results_root: str | Path):
    for done in sorted(Path(results_root).glob("*/mode_*/*/*/*/*/*/final.done.json")):
        yield done.parent


def collect(results_root: str | Path) -> pd.DataFrame:
    rows = []
    for d in completed_dirs(results_root):
        m = json.loads((d / "metrics.json").read_text())
        rows.append({**m["context"], "experiment": m["experiment"], "variant": m["variant"], "setting": m["setting"],
                     "score_type": m.get("score_type"), "threshold": m.get("threshold"), **summary_row(m),
                     "dir": str(d)})
    cols = KEY + ["auc"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


def development_only(table: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for any selection: development CV only (release.HOLDOUT_POLICY)."""
    return table[table["setting"] == "xu_dev_cv"].copy()


def _participants(d: str | Path) -> pd.DataFrame:
    p = pd.read_csv(Path(d) / "predictions_participants.csv")
    return p.groupby("key").agg(y=("y", "first"), score=("score", "mean"))


def paired(table: pd.DataFrame, factor: str, a: str, b: str) -> pd.DataFrame:
    """Pairs of results that differ only in ``factor`` (a vs b), with the paired DeLong test on the shared
    participants."""
    other = [c for c in KEY if c != factor]
    ta, tb = table[table[factor] == a], table[table[factor] == b]
    m = ta.merge(tb, on=other, suffixes=(f"_{a}", f"_{b}"))
    rows = []
    for _, r in m.iterrows():
        pa, pb = _participants(r[f"dir_{a}"]), _participants(r[f"dir_{b}"])
        common = sorted(set(pa.index) & set(pb.index))
        same = len(common) == len(pa) == len(pb) and (pa.loc[common, "y"] == pb.loc[common, "y"]).all()
        t = delong_paired(pa.loc[common, "y"], pa.loc[common, "score"], pb.loc[common, "score"]) if common else {}
        rows.append({**{c: r[c] for c in other}, f"auc_{a}": r[f"auc_{a}"], f"auc_{b}": r[f"auc_{b}"],
                     f"auc_diff_{a}_minus_{b}": t.get("diff", np.nan), "diff_se": t.get("se", np.nan),
                     "delong_p": t.get("p", np.nan), f"balanced_accuracy_{a}": r[f"balanced_accuracy_{a}"],
                     f"balanced_accuracy_{b}": r[f"balanced_accuracy_{b}"], "same_participants": bool(same)})
    return pd.DataFrame(rows)


def compare_inverses(table: pd.DataFrame) -> pd.DataFrame:
    return paired(table, "inverse", "lcmv", "wmne")


def compare_modes(table: pd.DataFrame) -> pd.DataFrame:
    return paired(table, "mode", "B", "A")


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df]
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df[cols].itertuples(index=False):
        out += "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |\n"
    return out


def write_tables(results_root: str | Path, out_dir: str | Path) -> dict:
    """All result tables of the release (new directory, never overwritten)."""
    out = Path(out_dir) / f"tables_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    t = collect(results_root)
    files = {"all_results": atomic_write_csv(out / "all_results.csv", t.drop(columns=["dir"], errors="ignore"))}
    if len(t):
        files["inverse"] = atomic_write_csv(out / "lcmv_vs_wmne.csv", compare_inverses(t))
        files["modes"] = atomic_write_csv(out / "mode_B_vs_A.csv", compare_modes(t))
        cols = KEY + ["auc", "auc_ci_low", "auc_ci_high", "mean_fold_auc", "min_fold_auc", "max_fold_auc",
                      "balanced_accuracy", "sensitivity", "specificity"]
        files["md"] = atomic_write_text(out / "all_results.md", "# All completed results (participant level)\n\n"
                                        "Holdout rows are descriptive only (release.HOLDOUT_POLICY).\n\n"
                                        + markdown_table(t.sort_values(KEY), cols))
    return {"dir": str(out), "n_results": int(len(t)), "files": {k: str(v) for k, v in files.items()}}


# ---------------------------------------------------------------------------------------------- one experiment
def experiment_report(results_root: str | Path, experiment: str, context: dict, out_dir: str | Path | None = None):
    """Table + diagnostic figures of one experiment in one context (diagnostics, not paper figures)."""
    t = collect(results_root)
    sel = t[(t["experiment"] == experiment)] if len(t) else t
    for c in CONTEXT:
        if len(sel):
            sel = sel[sel[c] == context[c]]
    base = Path(results_root) / context["inverse"] / f"mode_{context['mode']}" / context["atlas"] / context["graph"]
    out = Path(out_dir or base / experiment / "_reports") / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(out / "results.csv", sel.drop(columns=["dir"], errors="ignore"))
    figs = []
    if len(sel):
        try:
            figs = _figures(sel, out)
        except Exception as exc:                       # figures never block the table
            (out / "figures_error.txt").write_text(f"{type(exc).__name__}: {exc}")
    return {"dir": str(out), "table": sel.drop(columns=["dir"], errors="ignore"), "figures": figs}


def _figures(sel: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    made = []
    for setting, g in sel.groupby("setting"):
        fig, ax = plt.subplots(figsize=(5, 5))
        for _, r in g.iterrows():
            p = _participants(r["dir"])
            if p["y"].nunique() == 2:
                fpr, tpr, _ = roc_curve(p["y"], p["score"])
                ax.plot(fpr, tpr, lw=1, label=f"{r['variant']} ({r['auc']:.2f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("sensitivity")
        ax.set_title(f"participant ROC, {setting} (diagnostic)", fontsize=9)
        ax.legend(fontsize=5, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / f"roc_{setting}.png", dpi=150)
        plt.close(fig)
        made.append(f"roc_{setting}.png")
        fig, axes = plt.subplots(1, len(g), figsize=(1.6 * len(g) + 1, 2.0), squeeze=False)
        for ax, (_, r) in zip(axes[0], g.iterrows()):
            ax.imshow([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], cmap="Blues")
            for (i, j), val in np.ndenumerate([[r["tn"], r["fp"]], [r["fn"], r["tp"]]]):
                ax.text(j, i, int(val), ha="center", va="center", fontsize=7)
            ax.set_xticks([0, 1], ["HC", "AD"], fontsize=6)
            ax.set_yticks([0, 1], ["HC", "AD"], fontsize=6)
            ax.set_title(r["variant"], fontsize=5)
        fig.suptitle(f"confusion (rows true, columns predicted), {setting}", fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{setting}.png", dpi=150)
        plt.close(fig)
        made.append(f"confusion_{setting}.png")
    dev = sel[sel["setting"] == "xu_dev_cv"]
    if len(dev):
        fig, ax = plt.subplots(figsize=(max(4, 0.5 * len(dev)), 3))
        for i, (_, r) in enumerate(dev.iterrows()):
            fm = pd.read_csv(Path(r["dir"]) / "fold_metrics.csv")
            ax.scatter([i] * len(fm), fm["roc_auc"], s=10, c="grey")
            ax.scatter([i], [r["auc"]], marker="D", c="C3", s=18)
        ax.axhline(0.5, ls=":", c="k", lw=0.8)
        ax.set_xticks(range(len(dev)), dev["variant"], rotation=90, fontsize=5)
        ax.set_ylabel("participant AUC")
        ax.set_title("fold AUCs (grey) and pooled out-of-fold AUC (red)", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fold_aucs.png", dpi=150)
        plt.close(fig)
        made.append("fold_aucs.png")
        tr = [(r["variant"], pd.read_csv(Path(r["dir"]) / "training.csv")) for _, r in dev.iterrows()]
        if any("history_train_loss" in t for _, t in tr):
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for name, t in tr:
                if "history_train_loss" in t:
                    curves = np.array([json.loads(h) for h in t["history_train_loss"]])
                    ax.plot(range(1, curves.shape[1] + 1), curves.mean(0), lw=1, label=name)
            ax.set_xlabel("epoch")
            ax.set_ylabel("training loss (mean of the 5 folds)")
            ax.legend(fontsize=5)
            ax.set_title("convergence over the fixed 60 epochs", fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "training_loss.png", dpi=150)
            plt.close(fig)
            made.append("training_loss.png")
    return made
