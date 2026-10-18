"""The controlled dk68 vs r18 ROI comparison (``configs/final/roi_comparison.yaml``, notebook 04), from saved outputs.

The runs themselves are ordinary experiment units (``scripts/final.py run --exp ... --atlas ... --settings
xu_dev_cv``), so they share the folds, seeds, protocol, checkpoints and QC gate of every other result; the dk68 runs
are the same units the later experiment notebooks reuse. This module only reads their saved participant
predictions and puts two kinds of evidence side by side, never merged into one score:

* ``classification``: per cell (experiment, variant, inverse) and atlas, the participant-level pooled out-of-fold
  ROC-AUC with its DeLong CI, mean fold AUC, balanced accuracy, F1, sensitivity, specificity; and the paired
  r18 - dk68 differences on the SAME participants: paired DeLong (difference, SE, p), participant bootstrap 95 % CIs
  of the AUC and balanced-accuracy differences (participants resampled jointly for both atlases);
* ``summary``: descriptive wording of the classification differences (``descriptive_rule``): "no material
  difference", "tendency" or "inconsistent"; never a declaration of superiority, uncorrected for multiplicity;
* ``source_evidence``: the label-free resolution evidence of both atlases (participant operators against the stores'
  forward model, the analytic resolution table) and split-half feature reliability, read from notebook 02's outputs.

Only development-CV results are read (``report.development_only``); a holdout row can never enter.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..evaluation.metrics import binary_metrics, delong_paired
from .config import CONFIG_ROOT, experiment_configs
from .report import development_only

CLASS_COLS = ["auc", "auc_ci_low", "auc_ci_high", "mean_fold_auc", "balanced_accuracy", "f1", "sensitivity",
              "specificity", "n_participants"]


def config(root: str | Path | None = None) -> dict:
    return yaml.safe_load((Path(root or CONFIG_ROOT) / "roi_comparison.yaml").read_text())


def runs(cfg: dict | None = None) -> list[dict]:
    """Every experiment unit of the comparison: {exp, experiment, variant, atlas, inverse, mode, graph, setting}."""
    cfg = cfg or config()
    ids = {p.stem: k for k, p in experiment_configs().items()}
    out = []
    for atlas in cfg["atlases"]:
        for inv in cfg["inverses"]:
            for experiment, variants in cfg["models"].items():
                for var in variants:
                    out.append({"exp": ids[experiment], "experiment": experiment, "variant": var, "atlas": atlas,
                                "inverse": inv, "mode": cfg["mode"], "setting": cfg["setting"],
                                "graph": "none" if ids[experiment] in ("exp0", "exp1") else cfg["graph"]})
    return out


def _participants(d: str | Path) -> pd.DataFrame:
    p = pd.read_csv(Path(d) / "predictions_participants.csv")
    return p.groupby("key").agg(y=("y", "first"), score=("score", "mean"))


def paired_bootstrap(y, sa, sb, ta: float, tb: float, n: int, seed: int) -> dict:
    """Participant bootstrap of (b - a) AUC and balanced-accuracy differences, participants resampled jointly."""
    y, sa, sb = map(np.asarray, (y, sa, sb))
    rng = np.random.default_rng(seed)
    d_auc, d_ba = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        ma, mb = binary_metrics(y[i], sa[i], ta), binary_metrics(y[i], sb[i], tb)
        d_auc.append(mb["roc_auc"] - ma["roc_auc"])
        d_ba.append(mb["balanced_accuracy"] - ma["balanced_accuracy"])
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if v else [np.nan, np.nan]
    return {"auc_diff_boot_ci95": q(d_auc), "balanced_accuracy_diff_boot_ci95": q(d_ba), "n_boot": len(d_auc)}


def classification(table: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """(one row per cell with both atlases' metrics and the paired differences, missing units)."""
    cfg = cfg or config()
    a, b = cfg["atlases"]
    t = development_only(table) if len(table) else table
    rows, missing = [], []
    cells = sorted({(r["experiment"], r["variant"], r["inverse"], r["graph"]) for r in runs(cfg)})
    for exp, var, inv, graph in cells:
        sel = {}
        for atlas in (a, b):
            m = t[(t["experiment"] == exp) & (t["variant"] == var) & (t["inverse"] == inv) & (t["atlas"] == atlas)
                  & (t["mode"] == cfg["mode"]) & (t["graph"] == graph)] if len(t) else t
            if len(m) != 1:
                missing.append(f"{atlas}/{inv}/{exp}/{var}")
            else:
                sel[atlas] = m.iloc[0]
        if len(sel) < 2:
            continue
        ra, rb = sel[a], sel[b]
        pa, pb = _participants(ra["dir"]), _participants(rb["dir"])
        common = sorted(set(pa.index) & set(pb.index))
        same = (len(common) == len(pa) == len(pb)) and (pa.loc[common, "y"] == pb.loc[common, "y"]).all()
        y = pa.loc[common, "y"].to_numpy()
        sa, sb = pa.loc[common, "score"].to_numpy(), pb.loc[common, "score"].to_numpy()
        dl = delong_paired(y, sb, sa)                      # r18 (b) - dk68 (a)
        bs = paired_bootstrap(y, sa, sb, float(ra["threshold"]), float(rb["threshold"]), cfg["n_bootstrap"],
                              cfg["seed"])
        row = {"experiment": exp, "variant": var, "inverse": inv, "graph": graph, "same_participants": bool(same),
               "n_participants": len(common)}
        for atlas, r in ((a, ra), (b, rb)):
            row.update({f"{c}_{atlas}": r.get(c, np.nan) for c in CLASS_COLS if c != "n_participants"})
        row.update({f"auc_diff_{b}_minus_{a}": dl.get("diff", np.nan), "auc_diff_se": dl.get("se", np.nan),
                    "delong_p": dl.get("p", np.nan),
                    "auc_diff_boot_low": bs["auc_diff_boot_ci95"][0], "auc_diff_boot_high": bs["auc_diff_boot_ci95"][1],
                    f"balanced_accuracy_diff_{b}_minus_{a}": rb["balanced_accuracy"] - ra["balanced_accuracy"],
                    "balanced_accuracy_diff_boot_low": bs["balanced_accuracy_diff_boot_ci95"][0],
                    "balanced_accuracy_diff_boot_high": bs["balanced_accuracy_diff_boot_ci95"][1]})
        rows.append(row)
    return pd.DataFrame(rows), missing


def summary(cls: pd.DataFrame, missing: list[str], cfg: dict | None = None) -> dict:
    """Descriptive summary of the classification differences (no decision; see module docstring)."""
    cfg = cfg or config()
    a, b = cfg["atlases"]
    rule = cfg["descriptive_rule"]
    n_cells = len({(r["experiment"], r["variant"], r["inverse"]) for r in runs(cfg)})
    if missing or len(cls) < n_cells:
        return {"complete": False, "missing": missing, "statement": "comparison incomplete: no summary"}
    d = cls[f"auc_diff_{b}_minus_{a}"].to_numpy(float)
    mean, n_pos, n_neg = float(np.mean(d)), int((d > 0).sum()), int((d < 0).sum())
    excl = int(((cls["auc_diff_boot_low"] > 0) | (cls["auc_diff_boot_high"] < 0)).sum())
    frac = max(n_pos, n_neg) / len(d)
    if abs(mean) < rule["margin"]:
        st = (f"no material classification difference between {b} and {a} (mean AUC difference {mean:+.3f}, "
              f"|.| < {rule['margin']})")
    elif frac >= rule["min_fraction_of_cells_same_sign"]:
        better = b if mean > 0 else a
        st = (f"descriptive tendency towards {better} (mean AUC difference {b} - {a} {mean:+.3f}; same sign in "
              f"{max(n_pos, n_neg)}/{len(d)} cells; bootstrap CI excludes 0 in {excl}/{len(d)} cells); "
              "not a test of superiority")
    else:
        st = (f"inconsistent across models and inverses (mean AUC difference {mean:+.3f}; {n_pos} cells favour {b}, "
              f"{n_neg} favour {a})")
    return {"complete": True, "n_cells": int(len(d)), "mean_auc_diff": mean, "median_auc_diff": float(np.median(d)),
            f"cells_{b}_higher": n_pos, f"cells_{a}_higher": n_neg, "cells_boot_ci_excludes_0": excl,
            "statement": st, "caveat": ("52 development participants, 5 folds; paired comparisons are descriptive and "
                                        "not corrected for multiplicity; classification differences are not "
                                        "evidence about source localization")}


def _latest(pattern: str) -> str | None:
    found = sorted(glob.glob(pattern))
    return found[-1] if found else None


def source_evidence(final_root: str | Path, cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Label-free evidence of both atlases from notebook 02's saved outputs (latest of each)."""
    cfg = cfg or config()
    root = Path(final_root) / "parcellation"
    out = {}
    d = _latest(str(root / "resolution_*"))
    if d and Path(d, "resolution_by_atlas.csv").exists():
        t = pd.read_csv(Path(d) / "resolution_by_atlas.csv")
        out["analytic_resolution"] = t[t["atlas"].isin(cfg["atlases"])].assign(source=Path(d).name)
    rows = []
    for atlas in cfg["atlases"]:
        for inv in cfg["inverses"]:
            d = _latest(str(root / f"store_resolution_{inv}_{atlas}_*"))
            if d and Path(d, "per_participant.csv").exists():
                t = pd.read_csv(Path(d) / "per_participant.csv")
                num = t.select_dtypes("number")
                rows.append({"atlas": atlas, "inverse": inv, "n_participants": len(t),
                             **{f"{c}_median": float(num[c].median()) for c in num.columns}, "source": Path(d).name})
    if rows:
        out["participant_operator_resolution"] = pd.DataFrame(rows)
    rows = []
    for atlas in cfg["atlases"]:
        for inv in cfg["inverses"]:
            d = _latest(str(root / f"reliability_{inv}_{cfg['mode']}_{atlas}_*"))
            if d and Path(d, "split_half_reliability.csv").exists():
                t = pd.read_csv(Path(d) / "split_half_reliability.csv")
                for (feat, unit), g in t.groupby(["feature", "unit"]):
                    rows.append({"atlas": atlas, "inverse": inv, "feature": feat, "unit": unit,
                                 "median_over_bands": float(g["median"].median()), "source": Path(d).name})
    if rows:
        out["feature_reliability"] = pd.DataFrame(rows).pivot_table(
            index=["inverse", "feature", "unit"], columns="atlas", values="median_over_bands").reset_index()
    return out


def report_markdown(cls: pd.DataFrame, summ: dict, evidence: dict[str, pd.DataFrame], cfg: dict | None = None) -> str:
    from .report import markdown_table

    cfg = cfg or config()
    a, b = cfg["atlases"]
    md = [f"# ROI comparison {a} vs {b} (development CV only; holdout not evaluated)\n",
          "Two kinds of evidence, reported separately. The atlas is recorded by the author; classification AUC alone "
          "never decides it.\n", "## 1. Source localization and spatial resolution (no labels)\n"]
    for name, t in evidence.items():
        md += [f"### {name}\n", markdown_table(t, list(t.columns)), ""]
    if not evidence:
        md.append("(no notebook 02 outputs found)\n")
    md += ["## 2. Classification (participant level, pooled out-of-fold, 52 development participants)\n",
           f"**Summary:** {summ['statement']}\n", f"*{summ.get('caveat', '')}*\n"]
    if len(cls):
        cols = ["experiment", "variant", "inverse", f"auc_{a}", f"auc_{b}", f"auc_diff_{b}_minus_{a}",
                "auc_diff_boot_low", "auc_diff_boot_high", "delong_p", f"balanced_accuracy_{a}",
                f"balanced_accuracy_{b}", f"f1_{a}", f"f1_{b}", f"sensitivity_{a}", f"sensitivity_{b}",
                f"specificity_{a}", f"specificity_{b}"]
        md.append(markdown_table(cls, cols))
    return "\n".join(md)
