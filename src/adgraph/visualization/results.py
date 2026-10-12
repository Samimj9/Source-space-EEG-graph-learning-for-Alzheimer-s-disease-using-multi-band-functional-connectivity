"""Index and load *saved* experiment outputs (never re-runs anything).

An experiment output directory is any folder containing ``manifest.json`` and
``metrics_repeats.csv`` as written by ``adgraph.experiments.runner.run_variant``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..experiments.config import get_dotted

INDEX_COLUMNS = ["experiment", "variant", "setting", "condition", "bands", "band_mode", "encoder", "model",
                 "covariates", "score_type", "score_threshold", "timestamp", "path"]


def _get(cfg: dict, path: str, default=None):
    try:
        return get_dotted(cfg, path)
    except (KeyError, TypeError):
        return default


def index_results(root: str | Path, duplicates: str = "error") -> pd.DataFrame:
    """One row per saved (experiment, variant, setting) result, with its resolved variant config.

    ``duplicates``: ``error`` (default) if the same (experiment, variant, setting) appears in
    more than one run under ``root``; ``latest`` keeps the most recent by manifest timestamp.
    """
    rows = []
    for man in sorted(Path(root).rglob("manifest.json")):
        d = man.parent
        if not (d / "metrics_repeats.csv").exists():
            continue
        m = json.loads(man.read_text())
        v = m["variant_config"]
        g = v.get("graph", {}) if v.get("input") == "graph" else {}
        rows.append({
            "experiment": v["experiment"], "variant": v["variant"], "setting": m["setting"],
            "condition": g.get("connectivity", "none"),
            "bands": "+".join(g.get("bands", [])) if g else "",
            "band_mode": g.get("band_mode", ""), "encoder": _get(v, "model.encoder", ""),
            "model": _get(v, "model.name", "") if v.get("input") == "classical" else _get(v, "model.encoder", ""),
            "covariates": "+".join(v.get("covariates") or []),
            "score_type": m.get("score_type", "probability"), "score_threshold": m.get("score_threshold", 0.5),
            "timestamp": m.get("timestamp", ""), "path": str(d), "config": v,
        })
    idx = pd.DataFrame(rows, columns=INDEX_COLUMNS + ["config"])
    dup = idx.duplicated(["experiment", "variant", "setting"], keep=False)
    if dup.any():
        if duplicates == "latest":
            idx = (idx.sort_values("timestamp").drop_duplicates(["experiment", "variant", "setting"], keep="last")
                   .reset_index(drop=True))
        else:
            shown = idx.loc[dup, ["experiment", "variant", "setting", "path"]].to_string()
            raise ValueError(f"the same result appears in several runs under {root}:\n{shown}\n"
                             "pass a root with one run per result or duplicates='latest'")
    return idx


def matches(cfg: dict, where: dict | None) -> bool:
    for key, val in (where or {}).items():
        got = _get(cfg, key, object())
        if got != val:
            return False
    return True


def select(index: pd.DataFrame, experiment: str | None = None, settings=None, where: dict | None = None,
           conditions=None) -> pd.DataFrame:
    sel = index
    if experiment:
        sel = sel[sel["experiment"] == experiment]
    if settings:
        sel = sel[sel["setting"].isin(list(settings))]
    if conditions:
        sel = sel[sel["condition"].isin(list(conditions))]
    if where:
        sel = sel[[matches(c, where) for c in sel["config"]]]
    return sel.reset_index(drop=True)


def select_unique(index: pd.DataFrame, entry: dict, settings) -> pd.DataFrame:
    """Rows for one figure entry; at most one variant per (setting, condition), else an error."""
    sel = select(index, entry.get("experiment"), settings, entry.get("where"), entry.get("conditions"))
    order = {"plv": 0, "ciplv": 1, "none": 2}                                 # PLV (primary) first
    sel = sel.sort_values("condition", key=lambda c: c.map(order).fillna(3), kind="stable").reset_index(drop=True)
    dup = sel.duplicated(["setting", "condition"], keep=False)
    if dup.any():
        raise ValueError(f"entry {entry.get('label')} matches several variants per setting/condition: "
                         f"{sorted(set(sel.loc[dup, 'variant']))}; refine 'where'")
    return sel


def read_csv(row, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(Path(row["path"]) / name, **kw)


def read_manifest(row) -> dict:
    return json.loads((Path(row["path"]) / "manifest.json").read_text())


def read_summary(row) -> dict:
    return json.loads((Path(row["path"]) / "summary.json").read_text())


def subject_scores(row) -> pd.DataFrame:
    """Per subject: score averaged over repeats (as in summary.json), plus min/max over repeats."""
    s = read_csv(row, "predictions_subjects.csv")
    out = s.groupby("key").agg(dataset=("dataset", "first"), y=("y", "first"), score=("score", "mean"),
                               score_min=("score", "min"), score_max=("score", "max"),
                               n_repeats=("repeat", "nunique")).reset_index()
    return out


def same_splits(row_a, row_b) -> bool:
    """True if two results used identical subject folds (required for paired comparisons)."""
    fa = read_csv(row_a, "folds.csv").sort_values(["repeat", "fold", "side", "key"]).reset_index(drop=True)
    fb = read_csv(row_b, "folds.csv").sort_values(["repeat", "fold", "side", "key"]).reset_index(drop=True)
    return fa.equals(fb)


def anonymize(keys) -> dict:
    """Stable anonymous labels S01, S02, ... (order of sorted keys); the mapping is never saved."""
    return {k: f"S{i + 1:02d}" for i, k in enumerate(sorted(set(keys)))}


def finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]
