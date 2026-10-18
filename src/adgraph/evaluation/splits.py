"""Participant-level splits of the final framework (DESIGN.md).

Folds are always built from the list of *participants* (one row per ``key``) and only then expanded to windows, so
overlapping windows of one participant can never be on both sides of a split. ``fold_indices`` re-checks this for
every split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SETTINGS = {
    # name: (datasets used, test dataset or None for CV, stratify columns). One frozen participant split
    # (xu_partition): 52 development / 13 holdout participants of ds004504 (methodology log entry 50).
    "xu_dev_cv": (("D3",), None, ("y",)),       # 5-fold participant CV inside the development set only
    "xu_holdout": (("D3",), None, ("y",)),      # fit on the whole development set, test once on the holdout
}
XU_SETTINGS = ("xu_dev_cv", "xu_holdout")
XU_SPLIT = {"test_fraction": 0.2, "seed": 20250702, "dev_folds": 5}


@dataclass(frozen=True)
class Split:
    setting: str
    repeat: int
    fold: int
    train_keys: tuple[str, ...]
    test_keys: tuple[str, ...]


def subject_table(meta: pd.DataFrame, target: str = "y") -> pd.DataFrame:
    """One row per subject; fails if a subject's label or dataset is not constant over its windows."""
    g = meta.groupby("key")
    for col in ("dataset", target):
        if (g[col].nunique() > 1).any():
            raise ValueError(f"{col} varies within a subject")
    subj = g.agg(dataset=("dataset", "first"), y=(target, "first"), n_windows=("key", "size")).reset_index()
    return subj.sort_values(["dataset", "key"]).reset_index(drop=True)


def _strata(subj: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    return subj[list(cols)].astype(str).agg("|".join, axis=1).to_numpy()


def stratified_subject_folds(subj: pd.DataFrame, k: int, seed: int,
                             strata: tuple[str, ...] = ("y",)) -> list[tuple[list[str], list[str]]]:
    """K disjoint test folds over subjects, stratified by ``strata`` columns."""
    if k < 2 or k > len(subj):
        raise ValueError(f"k={k} invalid for {len(subj)} subjects")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    keys = subj["key"].to_numpy()
    return [(keys[tr].tolist(), keys[te].tolist()) for tr, te in skf.split(keys, _strata(subj, strata))]


def xu_partition(subj: pd.DataFrame, test_fraction: float = XU_SPLIT["test_fraction"],
                 seed: int = XU_SPLIT["seed"]) -> tuple[list[str], list[str]]:
    """Frozen participant split of the LCMV / Xu suite: (development keys, holdout test keys).

    Stratified by diagnosis: per class, round(test_fraction * n) participants (at least one) are drawn with a
    fixed seed from the sorted keys. Depends only on the participant table, so every experiment (Exp 0-7) and both
    modes get the identical split; the dataset build saves it (xu_split.json)."""
    rng = np.random.default_rng(seed)
    test = []
    for y in sorted(subj["y"].unique()):
        keys = sorted(subj.loc[subj["y"] == y, "key"])
        n = max(1, int(round(test_fraction * len(keys))))
        test += sorted(rng.choice(keys, size=n, replace=False).tolist())
    test = sorted(test)
    return sorted(set(subj["key"]) - set(test)), test


def make_splits(meta: pd.DataFrame, setting: str, k: int = 5, n_repeats: int = 1, seed: int = 0,
                target: str = "y", scheme: str = "kfold") -> list[Split]:
    """Outer splits of one setting.

    ``xu_dev_cv``: stratified participant K-fold (K = 5) over the 52 development participants only (the holdout never
    enters); repeat r uses seed + r. ``xu_holdout``: one split, all development participants -> the 13 holdout
    participants (repeats are model seeds only)."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; choose from {list(SETTINGS)}")
    if scheme != "kfold":
        raise ValueError("only participant k-fold CV is part of the final protocol")
    train_ds, _, strata = SETTINGS[setting]
    subj = subject_table(meta, target)
    subj = subj[subj["dataset"].isin(train_ds)].reset_index(drop=True)
    dev, test = xu_partition(subj)
    if setting == "xu_holdout":
        return [Split(setting, r, 0, tuple(dev), tuple(test)) for r in range(n_repeats)]
    subj = subj[subj["key"].isin(dev)].reset_index(drop=True)          # holdout participants never enter the CV
    k = XU_SPLIT["dev_folds"]
    splits = []
    for r in range(n_repeats):
        for f, (tr, te) in enumerate(stratified_subject_folds(subj, k, seed + r, strata)):
            splits.append(Split(setting, r, f, tuple(tr), tuple(te)))
    return splits


def inner_folds(train_meta: pd.DataFrame, k: int, seed: int, strata: tuple[str, ...] = ("y",),
                target: str = "y") -> list[tuple[np.ndarray, np.ndarray]]:
    """Subject-level inner CV *within a training view*, as window index arrays into ``train_meta``."""
    subj = subject_table(train_meta, target)
    k = min(k, int(subj.groupby(list(strata)).size().min()))
    if k < 2:
        raise ValueError("too few training subjects per stratum for inner CV")
    out = []
    keys = train_meta["key"].to_numpy()
    for tr, te in stratified_subject_folds(subj, k, seed, strata):
        out.append((np.flatnonzero(np.isin(keys, tr)), np.flatnonzero(np.isin(keys, te))))
    return out


def assert_disjoint(train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> None:
    shared = set(train_meta["key"]) & set(test_meta["key"])
    if shared:
        raise RuntimeError(f"subjects on both sides of a split: {sorted(shared)}")
    shared_w = set(train_meta["window_id"]) & set(test_meta["window_id"])
    if shared_w:
        raise RuntimeError(f"windows on both sides of a split: {sorted(shared_w)[:5]}")


def fold_indices(meta: pd.DataFrame, split: Split) -> tuple[np.ndarray, np.ndarray]:
    """Window indices of a split, with subject- and window-level disjointness checked."""
    keys = meta["key"].to_numpy()
    tr = np.flatnonzero(np.isin(keys, split.train_keys))
    te = np.flatnonzero(np.isin(keys, split.test_keys))
    assert_disjoint(meta.iloc[tr], meta.iloc[te])
    missing = set(split.test_keys) - set(keys[te])
    if missing:
        raise RuntimeError(f"test subjects without windows: {sorted(missing)}")
    return tr, te


def setting_subset(meta: pd.DataFrame, setting: str) -> np.ndarray:
    """Window indices of the datasets involved in a setting."""
    train_ds, test_ds, _ = SETTINGS[setting]
    involved = set(train_ds) | ({test_ds} if test_ds else set())
    return np.flatnonzero(meta["dataset"].isin(involved).to_numpy())
