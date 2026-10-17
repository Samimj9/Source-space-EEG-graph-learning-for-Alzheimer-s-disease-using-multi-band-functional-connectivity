"""The frozen LCMV / Xu evaluation protocol applied to another graph store (methodology log entry 55).

Authoritative reference: the validated LCMV store of the same mode (``graphsLX_*`` for all-window stores,
``graphsLX10_*`` for tenfold stores; the corrected ``_corr1`` stores) and its ``xu_split.json``. A target store
(matched wMNE ``graphsWX*`` or the historical wMNE ``graphsD3_v4_*``) may be evaluated with that protocol only if

1. it holds exactly the reference participants (keys) with the same diagnosis labels;
2. ``xu_partition`` gives exactly the saved 52 development / 13 holdout participants;
3. every outer split of ``xu_dev_cv`` and ``xu_holdout`` (``evaluation.splits.make_splits`` with the experiment's
   own k / repeats / seed, exactly as the runner calls it) has the same training and test participants as on the
   reference store;
4. no split puts a participant on both sides, and no holdout participant is in any development fold.

Nothing here is fitted or changed; the check only reads participant tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..evaluation.splits import XU_SETTINGS, make_splits, subject_table, xu_partition

REFERENCE_PREFIX = {"all": "D3_LCMV_XU/graphsLX", "tenfold": "D3_LCMV_XU/graphsLX10"}


class ProtocolMismatch(SystemExit):
    pass


def reference_store(processed: str | Path, tenfold: bool) -> Path:
    from ..pipeline import latest_complete_graph_run

    return latest_complete_graph_run(processed, REFERENCE_PREFIX["tenfold" if tenfold else "all"])


def splits_of(meta: pd.DataFrame, ev: dict) -> dict:
    """{setting: [(repeat, fold, train keys, test keys)]} as the runner builds them."""
    out = {}
    for s in XU_SETTINGS:
        sp = make_splits(meta, s, ev["k"], ev["n_repeats"], ev["seed"], "y", ev.get("scheme", "kfold"))
        out[s] = [(x.repeat, x.fold, tuple(sorted(x.train_keys)), tuple(sorted(x.test_keys))) for x in sp]
    return out


def compare(target_meta: pd.DataFrame, reference_meta: pd.DataFrame, saved_split: dict, ev: dict) -> dict:
    """All protocol checks; ``passed`` and the reasons for any failure."""
    problems = []
    t, r = subject_table(target_meta), subject_table(reference_meta)
    tk, rk = dict(zip(t["key"], t["y"])), dict(zip(r["key"], r["y"]))
    if set(tk) != set(rk):
        problems.append(f"participants differ: only in target {sorted(set(tk) - set(rk))}, "
                        f"only in reference {sorted(set(rk) - set(tk))}")
    lab = sorted(k for k in set(tk) & set(rk) if tk[k] != rk[k])
    if lab:
        problems.append(f"diagnosis labels differ for {lab}")
    dev, test = xu_partition(t[["key", "dataset", "y"]])
    if (dev, test) != (saved_split["development"], saved_split["test"]):
        problems.append("xu_partition of the target differs from the saved reference split")
    st, sr = splits_of(target_meta, ev), splits_of(reference_meta, ev)
    for s in XU_SETTINGS:
        if st[s] != sr[s]:
            problems.append(f"{s}: outer split membership differs from the reference ({len(st[s])} vs {len(sr[s])})")
        for rep, fold, tr, te in st[s]:
            if set(tr) & set(te):
                problems.append(f"{s} repeat {rep} fold {fold}: participants on both sides")
            if s == "xu_dev_cv" and set(tr + te) & set(saved_split["test"]):
                problems.append(f"{s} repeat {rep} fold {fold}: holdout participants in development CV")
        if s == "xu_holdout" and any(set(te) != set(saved_split["test"]) or set(tr) != set(saved_split["development"])
                                     for _, _, tr, te in st[s]):
            problems.append("xu_holdout: training set is not the development set or test set is not the holdout")
    return {"passed": not problems, "problems": problems, "n_participants": len(tk),
            "n_development": len(dev), "n_holdout": len(test),
            "dev_folds": [list(te) for _, _, _, te in st["xu_dev_cv"]]}


def verify(target: str | Path, processed: str | Path, ev: dict, tenfold: bool) -> dict:
    """Compare a target store with the reference LCMV store; raises ``ProtocolMismatch`` on any difference."""
    from ..graphs.store import GraphStore

    ref = reference_store(processed, tenfold)
    saved = json.loads((ref / "xu_split.json").read_text())
    res = compare(GraphStore.load(target, keys=()).windows, GraphStore.load(ref, keys=()).windows, saved, ev)
    res.update({"target": str(target), "reference": str(ref),
                "reference_graph_run_id": json.loads((ref / "manifest.json").read_text()).get("graph_run_id")})
    if not res["passed"]:
        raise ProtocolMismatch(f"{Path(target).name} does not reproduce the frozen protocol of {ref.name}: "
                               + "; ".join(res["problems"]))
    return res
