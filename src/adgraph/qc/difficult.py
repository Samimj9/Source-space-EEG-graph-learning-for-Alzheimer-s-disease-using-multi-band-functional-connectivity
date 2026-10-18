"""Category C: participants that are difficult to classify (after the experiments; out-of-fold predictions only).

For every completed development-CV result (one participant score per model), a participant is *misclassified* by
that model when its score falls on the wrong side of the model's fixed threshold. Across the models:

    misclassification_rate   fraction of models that misclassify the participant
    mean_signed_margin       mean of (score - threshold) x (+1 for AD, -1 for HC), on each model's own scale
                             divided by that model's score SD across participants (negative = wrong side)

Participants misclassified by at least ``rate`` (default 0.75) of the models are listed as difficult. They are
reported together with their QC flags (category A / B) so that a difficult participant with a technical problem can
be told apart from a valid but atypical one. Nobody is excluded; holdout predictions are not used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def difficult_participants(results: list[tuple[str, pd.DataFrame, float]], rate: float = 0.75) -> pd.DataFrame:
    """``results``: (model name, participant predictions with key / y / score, threshold) of development-CV runs."""
    rows = []
    for name, subj, thr in results:
        s = subj.groupby("key").agg(y=("y", "first"), score=("score", "mean"))
        sign = np.where(s["y"] == 1, 1.0, -1.0)
        sd = float(s["score"].std()) or 1.0
        rows.append(pd.DataFrame({"key": s.index, "model": name, "y": s["y"].to_numpy(),
                                  "wrong": (s["score"] >= thr).astype(int).to_numpy() != s["y"].to_numpy(),
                                  "margin": sign * (s["score"].to_numpy() - thr) / sd}))
    if not rows:
        return pd.DataFrame(columns=["key", "y", "n_models", "misclassification_rate", "mean_signed_margin",
                                     "difficult"])
    long = pd.concat(rows, ignore_index=True)
    out = long.groupby("key").agg(y=("y", "first"), n_models=("model", "nunique"),
                                  misclassification_rate=("wrong", "mean"), mean_signed_margin=("margin", "mean"))
    out["difficult"] = out["misclassification_rate"] >= rate
    return out.sort_values("misclassification_rate", ascending=False).reset_index()
