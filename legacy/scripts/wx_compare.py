"""wMNE versus LCMV under the frozen Xu protocol (methodology log entry 55); saved outputs only, nothing is trained.

    python scripts/wx_compare.py

Reads every completed result (final.done.json) of
  D3_LCMV_XU/experiments_{A_xu_tenfold,B_all_windows}/      LCMV        (lx*_exp*)
  D3_WMNE_XU/experiments_{A_xu_tenfold,B_all_windows}/      matched wMNE (wx*_exp*): identical definitions
  D3_WMNE_XU/experiments_H_historical_graphs/               historical wMNE graphs, frozen split (wxh_exp*)
and writes a new D3_WMNE_XU/reports/compare_<time>/ (never overwritten):
  paired.csv / paired.md      one row per mode x experiment x variant x setting present for BOTH LCMV and matched
                              wMNE (60-epoch and 200-epoch early-stopping experiments pair separately, by name): participant-level ROC-AUC, balanced accuracy, F1 (and window-level ROC-AUC) of
                              each, the wMNE - LCMV difference, and whether both used the same participants
  historical.csv              the historical-graph results (validation-matched only; different windows and bands,
                              so not paired with the matched comparison)
Descriptive: differences are reported, not ranked or tested; the holdout has 13 participants.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from adgraph.config import load_paths

parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
args = parser.parse_args()
processed = Path(load_paths(args.paths)["processed_dir"])
ROOTS = {("LCMV", "A"): "D3_LCMV_XU/experiments_A_xu_tenfold", ("LCMV", "B"): "D3_LCMV_XU/experiments_B_all_windows",
         ("wMNE", "A"): "D3_WMNE_XU/experiments_A_xu_tenfold", ("wMNE", "B"): "D3_WMNE_XU/experiments_B_all_windows",
         ("wMNE-historical", "B"): "D3_WMNE_XU/experiments_H_historical_graphs"}


def metrics(y, score, thr) -> dict:
    y, score = np.asarray(y, int), np.asarray(score, float)
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, spec = (tp / (tp + fn) if tp + fn else np.nan), (tn / (tn + fp) if tn + fp else np.nan)
    return {"roc_auc": roc_auc_score(y, score) if len(set(y)) == 2 else np.nan,
            "balanced_accuracy": np.nanmean([sens, spec]), "f1": f1_score(y, pred, zero_division=0),
            "n_score_values": int(len(np.unique(np.round(score, 7))))}


def fold_aucs(win: pd.DataFrame) -> list[float]:
    """Participant-level ROC-AUC within each outer fold (development CV), from the window predictions. Reported
    next to the pooled out-of-fold AUC, which ranks scores of differently calibrated fold models together."""
    out = []
    if "fold" not in win or win["fold"].nunique() < 2:
        return out
    for _, g in win.groupby(["repeat", "fold"] if "repeat" in win else "fold"):
        sg = g.groupby("key").agg(y=("y", "first"), score=("score", "mean"))
        if sg["y"].nunique() == 2:
            out.append(float(roc_auc_score(sg["y"], sg["score"])))
    return out


rows = []
for (method, mode), rel in ROOTS.items():
    for done in sorted((processed / rel).glob("*/*/*/final.done.json")):
        d = done.parent
        exp, variant, setting = d.parent.parent.name, d.parent.name, d.name
        man = json.loads((d / "manifest.json").read_text())
        thr = float(man.get("score_threshold", 0.5))
        subj = pd.read_csv(d / "predictions_subjects.csv")
        win = pd.read_csv(d / "predictions_windows.csv")
        ps = subj.groupby("key").agg(y=("y", "first"), score=("score", "mean"))
        m = {f"participant_{k}": v for k, v in metrics(ps["y"], ps["score"], thr).items()}
        m["window_roc_auc"] = roc_auc_score(win["y"], win["score"]) if win["y"].nunique() == 2 else np.nan
        fa = fold_aucs(win)
        m["participant_mean_fold_auc"] = float(np.mean(fa)) if fa else np.nan       # dev CV only; pooled is primary
        m["participant_fold_auc_min_max"] = f"{min(fa):.3f}-{max(fa):.3f}" if fa else ""
        rows.append({"method": method, "mode": mode, "experiment": exp.split("_", 1)[1], "variant": variant,
                     "setting": setting, "participants": ",".join(sorted(ps.index)), "n_participants": len(ps),
                     "graph_run": (man.get("graph_run") or {}).get("graph_run_id"), "git_commit": man.get("git_commit"),
                     **m})
res = pd.DataFrame(rows)
out = processed / "D3_WMNE_XU" / "reports" / f"compare_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True, exist_ok=True)
key = ["mode", "experiment", "variant", "setting"]
if len(res):
    lc, wm = res[res.method == "LCMV"], res[res.method == "wMNE"]
    paired = lc.merge(wm, on=key, suffixes=("_lcmv", "_wmne"))
    paired["same_participants"] = paired["participants_lcmv"] == paired["participants_wmne"]
    paired["same_code_commit"] = paired["git_commit_lcmv"] == paired["git_commit_wmne"]   # e.g. the ES200 pins
    for k in ("participant_roc_auc", "participant_balanced_accuracy", "participant_f1", "window_roc_auc"):
        paired[f"{k}_diff_wmne_minus_lcmv"] = paired[f"{k}_wmne"] - paired[f"{k}_lcmv"]
    paired = paired.drop(columns=[c for c in paired if c.startswith("participants_") or c.startswith("method")])
    paired.to_csv(out / "paired.csv", index=False)
    cols = key + ["n_participants_lcmv", "participant_roc_auc_lcmv", "participant_roc_auc_wmne",
                  "participant_roc_auc_diff_wmne_minus_lcmv", "participant_balanced_accuracy_lcmv",
                  "participant_balanced_accuracy_wmne", "same_participants", "same_code_commit"]
    (out / "paired.md").write_text(
        "# wMNE vs LCMV, frozen Xu protocol (descriptive; participant-level metrics primary)\n\n"
        + "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        + "\n".join("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
                    for r in paired[cols].itertuples(index=False)) + "\n")
    res[res.method == "wMNE-historical"].drop(columns=["participants"]).to_csv(out / "historical.csv", index=False)
    print(paired[cols].round(3).to_string(index=False) if len(paired) else "no experiment completed for both methods")
    if (~paired["same_participants"]).any():
        print("WARNING: some pairs were not evaluated on the same participants")
    if (~paired["same_code_commit"]).any():
        print("NOTE: some pairs were trained at different code commits (see same_code_commit)")
else:
    print("no completed results")
print(f"\n{len(res)} completed results -> {out}")
