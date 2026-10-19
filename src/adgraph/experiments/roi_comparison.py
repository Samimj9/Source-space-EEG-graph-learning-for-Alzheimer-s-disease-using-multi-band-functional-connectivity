"""The controlled atlas comparison (``configs/final/roi_comparison.yaml``, notebook 04), from saved outputs: dk68
(reference) vs lobes12 (rule-recommended) vs r18 (exploratory, post-result approval), or any atlases the config lists.

The runs themselves are ordinary experiment units (``scripts/final.py run --exp ... --atlas ... --settings
xu_dev_cv``), so they share the folds, seeds, protocol, checkpoints and QC gate of every other result; the dk68 runs
are the same units the later experiment notebooks reuse. This module only reads their saved participant
predictions and puts two kinds of evidence side by side, never merged into one score:

* ``per_atlas``: per cell (experiment, variant, inverse) and atlas, the participant-level pooled out-of-fold ROC-AUC
  with its DeLong CI, mean fold AUC, balanced accuracy, F1, sensitivity, specificity;
* ``classification``: for every configured comparison ``a - b`` (e.g. lobes12 - dk68, r18 - dk68, r18 - lobes12) the
  paired differences on the SAME participants: paired DeLong (difference, SE, p), participant bootstrap 95 % CIs of
  the AUC and balanced-accuracy differences (participants resampled jointly for both atlases);
* ``summary``: per comparison, descriptive wording of the differences (``descriptive_rule``): "no material
  difference", "tendency" or "inconsistent"; never a declaration of superiority, uncorrected for multiplicity; a
  comparison with a missing unit is "incomplete" (the others are still reported);
* ``source_evidence``: the label-free resolution evidence of every atlas (participant operators against the stores'
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


def _cells(cfg: dict) -> list[tuple]:
    return sorted({(r["experiment"], r["variant"], r["inverse"], r["graph"]) for r in runs(cfg)})


def _row(t: pd.DataFrame, cfg: dict, cell: tuple, atlas: str):
    exp, var, inv, graph = cell
    if not len(t):
        return None
    m = t[(t["experiment"] == exp) & (t["variant"] == var) & (t["inverse"] == inv) & (t["atlas"] == atlas)
          & (t["mode"] == cfg["mode"]) & (t["graph"] == graph)]
    return m.iloc[0] if len(m) == 1 else None


def per_atlas(table: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """(one row per cell and atlas with its participant-level metrics, missing units ``atlas/inverse/exp/variant``)."""
    cfg = cfg or config()
    t = development_only(table) if len(table) else table
    rows, missing = [], []
    for cell in _cells(cfg):
        for atlas in cfg["atlases"]:
            r = _row(t, cfg, cell, atlas)
            if r is None:
                missing.append(f"{atlas}/{cell[2]}/{cell[0]}/{cell[1]}")
                continue
            rows.append({"experiment": cell[0], "variant": cell[1], "inverse": cell[2], "atlas": atlas,
                         "role": cfg.get("roles", {}).get(atlas, ""), **{c: r.get(c, np.nan) for c in CLASS_COLS}})
    return pd.DataFrame(rows), missing


def classification(table: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """(one row per cell and comparison a - b with both atlases' metrics and the paired differences, missing
    units). Columns ``*_a`` / ``*_b`` belong to the atlases named in ``atlas_a`` / ``atlas_b``."""
    cfg = cfg or config()
    t = development_only(table) if len(table) else table
    _, missing = per_atlas(table, cfg)
    rows = []
    for cell in _cells(cfg):
        exp, var, inv, graph = cell
        for a, b in cfg["comparisons"]:
            ra, rb = _row(t, cfg, cell, a), _row(t, cfg, cell, b)
            if ra is None or rb is None:
                continue
            pa, pb = _participants(ra["dir"]), _participants(rb["dir"])
            common = sorted(set(pa.index) & set(pb.index))
            same = (len(common) == len(pa) == len(pb)) and (pa.loc[common, "y"] == pb.loc[common, "y"]).all()
            y = pa.loc[common, "y"].to_numpy()
            sa, sb = pa.loc[common, "score"].to_numpy(), pb.loc[common, "score"].to_numpy()
            dl = delong_paired(y, sa, sb)                      # a - b
            bs = paired_bootstrap(y, sb, sa, float(rb["threshold"]), float(ra["threshold"]), cfg["n_bootstrap"],
                                  cfg["seed"])               # (a) - (b): the second argument is subtracted from
            row = {"comparison": f"{a} - {b}", "atlas_a": a, "atlas_b": b, "experiment": exp, "variant": var,
                   "inverse": inv, "graph": graph, "same_participants": bool(same), "n_participants": len(common)}
            for side, r in (("a", ra), ("b", rb)):
                row.update({f"{c}_{side}": r.get(c, np.nan) for c in CLASS_COLS if c != "n_participants"})
            row.update({"auc_diff": dl.get("diff", np.nan), "auc_diff_se": dl.get("se", np.nan),
                        "delong_p": dl.get("p", np.nan), "auc_diff_boot_low": bs["auc_diff_boot_ci95"][0],
                        "auc_diff_boot_high": bs["auc_diff_boot_ci95"][1],
                        "balanced_accuracy_diff": ra["balanced_accuracy"] - rb["balanced_accuracy"],
                        "balanced_accuracy_diff_boot_low": bs["balanced_accuracy_diff_boot_ci95"][0],
                        "balanced_accuracy_diff_boot_high": bs["balanced_accuracy_diff_boot_ci95"][1]})
            rows.append(row)
    return pd.DataFrame(rows), missing


def summary(cls: pd.DataFrame, missing: list[str], cfg: dict | None = None) -> dict:
    """Per comparison, a descriptive summary of the classification differences (no decision; module docstring)."""
    cfg = cfg or config()
    rule = cfg["descriptive_rule"]
    n_cells = len({c[:3] for c in _cells(cfg)})
    out = {}
    for a, b in cfg["comparisons"]:
        name = f"{a} - {b}"
        miss = [m for m in missing if m.split("/")[0] in (a, b)]
        g = cls[cls["comparison"] == name] if len(cls) else cls
        if miss or len(g) < n_cells:
            out[name] = {"complete": False, "missing": miss, "statement": f"{name}: comparison incomplete: no summary"}
            continue
        d = g["auc_diff"].to_numpy(float)
        mean, n_pos, n_neg = float(np.mean(d)), int((d > 0).sum()), int((d < 0).sum())
        excl = int(((g["auc_diff_boot_low"] > 0) | (g["auc_diff_boot_high"] < 0)).sum())
        frac = max(n_pos, n_neg) / len(d)
        if abs(mean) < rule["margin"]:
            st = (f"no material classification difference between {a} and {b} (mean AUC difference {mean:+.3f}, "
                  f"|.| < {rule['margin']})")
        elif frac >= rule["min_fraction_of_cells_same_sign"]:
            better = a if mean > 0 else b
            st = (f"descriptive tendency towards {better} (mean AUC difference {a} - {b} {mean:+.3f}; same sign in "
                  f"{max(n_pos, n_neg)}/{len(d)} cells; bootstrap CI excludes 0 in {excl}/{len(d)} cells); "
                  "not a test of superiority")
        else:
            st = (f"inconsistent across models and inverses (mean AUC difference {a} - {b} {mean:+.3f}; {n_pos} cells "
                  f"favour {a}, {n_neg} favour {b})")
        out[name] = {"complete": True, "n_cells": int(len(d)), "mean_auc_diff": mean,
                     "median_auc_diff": float(np.median(d)), "cells_a_higher": n_pos, "cells_b_higher": n_neg,
                     "cells_boot_ci_excludes_0": excl, "statement": st}
    return {"comparisons": out, "complete": all(v["complete"] for v in out.values()),
            "caveat": ("exploratory, not corrected for multiplicity; 52 development participants, 5 folds; "
                       "classification differences are not evidence about source localization; no atlas is selected")}


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


def report_markdown(atlas_table: pd.DataFrame, cls: pd.DataFrame, summ: dict, evidence: dict[str, pd.DataFrame],
                    cfg: dict | None = None, authorization: dict | None = None) -> str:
    from .report import markdown_table

    cfg = cfg or config()
    md = [f"# Atlas comparison: {', '.join(cfg['atlases'])} (development CV only; holdout not evaluated)\n",
          "Exploratory comparison, not corrected for multiplicity. Source-resolution evidence and classification "
          "performance are reported separately; classification AUC alone never decides an atlas, and no atlas is "
          "selected here.\n", "| atlas | role | authorization |\n|---|---|---|"]
    for atlas in cfg["atlases"]:
        md.append(f"| {atlas} | {cfg.get('roles', {}).get(atlas, '')} | {(authorization or {}).get(atlas, '')} |")
    md += ["", "## 1. Source localization and spatial resolution (no labels)\n"]
    for name, t in evidence.items():
        md += [f"### {name}\n", markdown_table(t, list(t.columns)), ""]
    if not evidence:
        md.append("(no notebook 02 outputs found)\n")
    md += ["## 2. Classification per atlas (participant level, pooled out-of-fold, 52 development participants)\n"]
    if len(atlas_table):
        md.append(markdown_table(atlas_table, ["experiment", "variant", "inverse", "atlas", "auc", "auc_ci_low",
                                               "auc_ci_high", "mean_fold_auc", "balanced_accuracy", "f1",
                                               "sensitivity", "specificity"]))
    md += ["## 3. Paired differences (same participants; descriptive)\n", f"*{summ.get('caveat', '')}*\n"]
    for name, s in summ["comparisons"].items():
        md.append(f"### {name}\n\n**Summary:** {s['statement']}\n")
        g = cls[cls["comparison"] == name] if len(cls) else cls
        if len(g):
            md.append(markdown_table(g, ["experiment", "variant", "inverse", "auc_a", "auc_b", "auc_diff",
                                         "auc_diff_boot_low", "auc_diff_boot_high", "delong_p",
                                         "balanced_accuracy_diff", "balanced_accuracy_diff_boot_low",
                                         "balanced_accuracy_diff_boot_high"]))
    return "\n".join(md)
