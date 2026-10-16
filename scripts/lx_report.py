"""Comparison of all completed D3 LCMV / Xu experiments (Exp 0-7, modes A and B); saved outputs only.

    python scripts/lx_report.py

Reads every completed result (final.done.json) under <processed>/D3_LCMV_XU/experiments_*/ and writes a new
<processed>/D3_LCMV_XU/reports/report_<time>/ (never overwritten):
  comparison.csv / comparison.md   one row per mode x experiment x variant x setting:
                                   PARTICIPANT-level metrics (score = mean of the participant's window scores) and
                                   WINDOW-level metrics, clearly prefixed (participant_*, window_*): accuracy,
                                   balanced accuracy, sensitivity, specificity, precision, F1, ROC-AUC, confusion
                                   counts; score type and threshold; compute device(s) of the result
  figures/roc_<mode>_<setting>.png participant-level ROC curves per experiment
  figures/confusion_<mode>_<setting>.png  participant-level confusion matrices
Results computed on CPU and on GPU are combined; each row keeps its own environment record.
Nothing is selected or ranked: rows are listed in experiment / variant order.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from adgraph.config import load_paths
from adgraph.experiments.config import FAMILIES

parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
args = parser.parse_args()
processed = Path(load_paths(args.paths)["processed_dir"])
MODES = {"LX10": "A_xu_tenfold", "LX": "B_all_windows"}


def metrics(y, score, thr) -> dict:
    y, score = np.asarray(y, int), np.asarray(score, float)
    pred = (score >= thr).astype(int) if np.isfinite(thr) else (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec and sens and np.isfinite(prec + sens) else 0.0
    auc = roc_auc_score(y, score) if len(set(y)) == 2 else np.nan
    return {"accuracy": (tp + tn) / len(y), "balanced_accuracy": np.nanmean([sens, spec]), "sensitivity": sens,
            "specificity": spec, "precision": prec, "f1": f1, "roc_auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


rows, curves = [], {}
for fam, mode in MODES.items():
    root = processed / FAMILIES[fam][1]
    for done in sorted(root.glob("*/*/*/final.done.json")):
        d = done.parent
        exp, variant, setting = d.parent.parent.name, d.parent.name, d.name
        man = json.loads((d / "manifest.json").read_text())
        thr = float(man.get("score_threshold", 0.5))
        subj = pd.read_csv(d / "predictions_subjects.csv")
        win = pd.read_csv(d / "predictions_windows.csv")
        env = man.get("compute_environment") or {}
        hp = pd.read_csv(d / "hparams.csv") if (d / "hparams.csv").exists() else pd.DataFrame()
        row = {"mode": mode, "experiment": exp, "variant": variant, "setting": setting,
               "n_participants": int(subj["key"].nunique()), "n_windows": int(win["window_id"].nunique()),
               "n_repeats": int(subj["repeat"].nunique()), "score_type": man.get("score_type"), "threshold": thr,
               "device": env.get("device"), "gpu": env.get("gpu"),
               "n_compute_environments": int(hp["compute_fingerprint"].nunique()) if "compute_fingerprint" in hp else 1,
               "git_commit": man.get("git_commit"), "graph_run": (man.get("graph_run") or {}).get("graph_run_id")}
        ps = subj.groupby("key").agg(y=("y", "first"), score=("score", "mean"))     # mean over repeats
        row.update({f"participant_{k}": v for k, v in metrics(ps["y"], ps["score"], thr).items()})
        row.update({f"window_{k}": v for k, v in metrics(win["y"], win["score"], thr).items()})
        rows.append(row)
        curves.setdefault((mode, setting, exp), []).append((variant, ps["y"].to_numpy(), ps["score"].to_numpy(), thr))

out = processed / "D3_LCMV_XU" / "reports" / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
(out / "figures").mkdir(parents=True, exist_ok=True)
table = pd.DataFrame(rows)
table.to_csv(out / "comparison.csv", index=False)
if len(table):
    cols = ["mode", "experiment", "variant", "setting", "n_participants", "participant_accuracy",
            "participant_balanced_accuracy", "participant_sensitivity", "participant_specificity",
            "participant_precision", "participant_f1", "participant_roc_auc", "window_roc_auc",
            "window_balanced_accuracy", "device"]
    (out / "comparison.md").write_text(
        "# D3 LCMV / Xu-protocol suite: comparison (descriptive; participant-level metrics are primary)\n\n"
        "Participant score = mean of the participant's window scores; window-level metrics treat overlapping "
        "windows of one participant as separate samples and are secondary.\n\n"
        + "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        + "\n".join("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
                    for r in table[cols].itertuples(index=False)) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for (mode, setting) in sorted({(m, s) for m, s, _ in curves}):
        exps = sorted(e for m, s, e in curves if (m, s) == (mode, setting))
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
        figc, axc = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
        for ax, axm, exp in zip(axes.ravel(), axc.ravel(), exps):
            best_cm = None
            for variant, y, sc, thr in curves[(mode, setting, exp)]:
                if len(set(y)) == 2:
                    fpr, tpr, _ = roc_curve(y, sc)
                    ax.plot(fpr, tpr, lw=1, label=f"{variant} ({roc_auc_score(y, sc):.2f})")
                if best_cm is None:                     # first variant in config order (no selection)
                    best_cm = (variant, confusion_matrix(y, (sc >= thr).astype(int), labels=[0, 1]))
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set_title(exp, fontsize=9)
            ax.legend(fontsize=5, loc="lower right")
            if best_cm:
                axm.imshow(best_cm[1], cmap="Blues")
                for (i, j), v in np.ndenumerate(best_cm[1]):
                    axm.text(j, i, str(v), ha="center", va="center")
                axm.set_xticks([0, 1], ["pred HC", "pred AD"])
                axm.set_yticks([0, 1], ["HC", "AD"])
                axm.set_title(f"{exp}\n{best_cm[0]}", fontsize=8)
        for a in list(axes.ravel()[len(exps):]) + list(axc.ravel()[len(exps):]):
            a.axis("off")
        fig.suptitle(f"participant-level ROC, mode {mode}, {setting}")
        figc.suptitle(f"participant-level confusion matrices (first variant of each experiment), {mode}, {setting}")
        fig.tight_layout()
        figc.tight_layout()
        fig.savefig(out / "figures" / f"roc_{mode}_{setting}.png", dpi=150)
        figc.savefig(out / "figures" / f"confusion_{mode}_{setting}.png", dpi=150)
        plt.close("all")
print(f"{len(table)} completed results -> {out}")
if len(table):
    print(table[["mode", "experiment", "variant", "setting", "participant_roc_auc", "participant_balanced_accuracy",
                 "window_roc_auc", "device"]].round(3).to_string(index=False))
