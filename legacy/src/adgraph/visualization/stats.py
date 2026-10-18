"""Statistics supplied by a *separate, saved* statistical analysis.

Figures never compute significance themselves. Significance marks are drawn only
when a statistics CSV is given and every used row is valid: it names the test,
the number of subjects, and a multiple-comparison-corrected p-value with its
correction method. Windows are never the statistical unit (``unit`` must be
``subject``).

Required columns:
    comparison         identifier the figure looks up (documented per figure in docs/figures.md)
    metric             quantity compared
    test               name of the statistical test
    unit               must be "subject"
    n_subjects         number of subjects entering the test
    p_value            uncorrected p
    p_corrected        corrected p (e.g. Holm, FDR-BH)
    correction_method  name of the correction, non-empty
    family             the set of comparisons the correction was applied over
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ("comparison", "metric", "test", "unit", "n_subjects", "p_value", "p_corrected", "correction_method",
            "family")


class InvalidStatistics(ValueError):
    pass


def load_stats(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise InvalidStatistics(f"statistics file lacks columns {missing}")
    bad = (df["p_corrected"].isna() | df["correction_method"].isna()
           | (df["correction_method"].astype(str).str.strip() == "") | (df["n_subjects"] <= 0)
           | (df["unit"].astype(str) != "subject") | ~df["p_corrected"].between(0, 1))
    if bad.any():
        raise InvalidStatistics(f"invalid statistics rows (need subject-level, corrected p): "
                                f"{df.loc[bad, 'comparison'].tolist()[:10]}")
    if (df["p_corrected"] < df["p_value"] - 1e-12).any():
        raise InvalidStatistics("corrected p-values smaller than uncorrected ones")
    return df


def lookup(stats: pd.DataFrame | None, comparison: str) -> dict | None:
    if stats is None:
        return None
    rows = stats[stats["comparison"] == comparison]
    if len(rows) > 1:
        raise InvalidStatistics(f"several rows for comparison {comparison!r}")
    return None if rows.empty else rows.iloc[0].to_dict()


def stars(p_corrected: float, levels=(0.001, 0.01, 0.05)) -> str:
    return "***" if p_corrected < levels[0] else "**" if p_corrected < levels[1] else "*" if p_corrected < levels[2] \
        else ""
