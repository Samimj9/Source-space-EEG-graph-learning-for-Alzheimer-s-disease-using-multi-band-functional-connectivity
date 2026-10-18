"""Pre-specified, development-only selection of the graph construction (configs/final/selection.yaml).

``select(table)`` takes the combined result table (``experiments.report.collect``) and applies the rule of
``selection.yaml`` (the atlas is not chosen here: it follows the source-resolution rule of
``source.resolution.recommend_atlas``, independent of classification results):

* only development-CV rows (``report.development_only``); a holdout row can never enter;
* one cell = (reference experiment, variant, inverse) in the pre-specified mode / atlas / graph;
* a candidate must have completed every cell (an incomplete comparison selects nothing: the default is kept and
  the decision says which cells are missing);
* the candidate with the highest mean participant AUC replaces the default only if that mean is at least the
  default's mean + ``margin`` AND it beats the default in at least ``min_fraction_of_cells_better`` of the cells.

The decision (every number it used) is returned and saved by ``scripts/final.py select`` to
``FINAL_v1/decisions/graph_construction.json``; it is never computed from, or changed by, holdout results.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml

from .config import CONFIG_ROOT
from .report import development_only

FACTOR_COLUMN = {"graph_construction": "graph"}


def rules(root: str | Path | None = None) -> dict:
    return yaml.safe_load((Path(root or CONFIG_ROOT) / "selection.yaml").read_text())


def cells(table: pd.DataFrame, factor: str, rule: dict) -> pd.DataFrame:
    """Development-CV rows of the reference cells: columns cell, option, auc."""
    from .. import release

    col = FACTOR_COLUMN[factor]
    t = development_only(table)
    fixed = {k: rule[k] for k in ("mode", "atlas", "graph") if k in rule and k != col}
    if fixed.get("atlas") == "primary":
        fixed["atlas"] = release.PRIMARY["atlas"]
    rows = []
    for exp, variants in rule["reference"].items():
        for var in variants:
            for inv in rule["inverses"]:
                sel = t[(t["experiment"] == exp) & (t["variant"] == var) & (t["inverse"] == inv)]
                for k, val in fixed.items():
                    if k == "graph" and sel["graph"].eq("none").all():
                        continue                        # classical experiments have no graph construction
                    sel = sel[sel[k] == val]
                for _, r in sel.iterrows():
                    rows.append({"cell": f"{exp}/{var}/{inv}", "option": r[col], "auc": float(r["auc"])})
    return pd.DataFrame(rows, columns=["cell", "option", "auc"])


def select(table: pd.DataFrame, factor: str = "graph_construction", rule: dict | None = None) -> dict:
    rule = rule or rules()[factor]
    c = cells(table, factor, rule)
    expected = [f"{e}/{v}/{i}" for e, vs in rule["reference"].items() for v in vs for i in rule["inverses"]]
    default = rule["default"]
    options = [default] + list(rule["candidates"]) + list(rule.get("reported_only") or [])
    wide = c.pivot_table(index="cell", columns="option", values="auc").reindex(index=expected)
    missing = {o: [cl for cl in expected if o not in wide or pd.isna(wide.loc[cl, o])] for o in options}
    summary = []
    for o in options:
        if o not in wide:
            summary.append({"option": o, "complete": False, "mean_auc": float("nan"), "cells_better": 0})
            continue
        better = int((wide[o] > wide[default]).sum()) if default in wide and o != default else 0
        summary.append({"option": o, "complete": not missing[o], "mean_auc": float(wide[o].mean()),
                        "cells_better": better, "eligible": o in rule["candidates"]})
    s = pd.DataFrame(summary)
    need = math.ceil(rule["min_fraction_of_cells_better"] * len(expected))
    decision = {"factor": factor, "default": default, "selected": default, "rule": rule, "cells": expected,
                "n_cells": len(expected), "cells_better_required": need, "summary": s.to_dict("records"),
                "per_cell": wide.reset_index().to_dict("records"), "missing": {k: v for k, v in missing.items() if v},
                "source": "development CV only (xu_dev_cv); holdout never read"}
    if missing.get(default) or any(missing.get(o) for o in rule["candidates"]):
        decision["reason"] = "comparison incomplete: the default is kept until every reference cell is complete"
        return decision
    base = float(wide[default].mean())
    cand = s[s["option"].isin(rule["candidates"])].sort_values("mean_auc", ascending=False)
    best = cand.iloc[0] if len(cand) else None
    if best is not None and best["mean_auc"] >= base + rule["margin"] and best["cells_better"] >= need:
        decision["selected"] = best["option"]
        decision["reason"] = (f"{best['option']}: mean development AUC {best['mean_auc']:.3f} >= {default} "
                              f"{base:.3f} + {rule['margin']} and better in {best['cells_better']}/{len(expected)} cells")
    else:
        decision["reason"] = (f"no candidate is better than {default} (mean development AUC {base:.3f}) by "
                              f">= {rule['margin']} in >= {need}/{len(expected)} cells; the default is kept")
    return decision
