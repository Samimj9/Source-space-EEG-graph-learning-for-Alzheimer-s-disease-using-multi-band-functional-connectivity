"""Fold-level QC: outlier thresholds fitted on each outer fold's TRAINING participants only.

For every outer split the runner calls ``fold_qc(summaries, train_keys, test_keys)``: the modified z-scores of every
participant are computed against the training participants' median / MAD alone, and each participant is flagged
(category B) if any summary exceeds the threshold. The result is saved with the fold (``units/<fold>.qc.csv``) and
summarized in ``training.csv``; it documents how the test participants relate to the training distribution. The
policy of the final protocol is ``report``: no participant is removed or reweighted, so these thresholds never
influence training or prediction.
"""

from __future__ import annotations

import pandas as pd

from .participants import Z_THRESHOLD, modified_z


def fold_qc(summaries: pd.DataFrame, train_keys, test_keys, z_threshold: float = Z_THRESHOLD) -> pd.DataFrame:
    train_keys, test_keys = list(train_keys), list(test_keys)
    if set(train_keys) & set(test_keys):
        raise RuntimeError("fold QC: training and test participants overlap")
    keys = [k for k in train_keys + test_keys if k in summaries.index]
    z = modified_z(summaries.loc[keys], train_keys)
    flagged = z.abs() > z_threshold
    out = pd.DataFrame({"key": keys, "side": ["train" if k in set(train_keys) else "test" for k in keys],
                        "n_features_flagged": flagged.sum(axis=1).to_numpy(),
                        "max_abs_modified_z": z.abs().replace(float("inf"), float("nan")).max(axis=1).to_numpy()})
    out["flagged"] = out["n_features_flagged"] > 0
    out["top_feature"] = [z.loc[k].abs().replace(float("inf"), float("nan")).idxmax()
                          if z.loc[k].notna().any() else "" for k in keys]
    return out


def fold_summary(table: pd.DataFrame) -> dict:
    t = table[table["side"] == "test"]
    r = table[table["side"] == "train"]
    return {"qc_train_flagged": int(r["flagged"].sum()), "qc_test_flagged": int(t["flagged"].sum()),
            "qc_test_flagged_keys": ",".join(t.loc[t["flagged"], "key"])}
