"""Participant-level splits of the final protocol: the frozen 52 / 13 partition of ds004504 and its five
development folds, participant disjointness, stratification, reproducibility and inner folds (synthetic tables)."""

import numpy as np
import pandas as pd
import pytest

from adgraph.evaluation.splits import (SETTINGS, XU_SETTINGS, Split, fold_indices, inner_folds, make_splits,
                                       setting_subset, subject_table, xu_partition)

LABELS = {f"D3-{n}": int(n <= 36) for n in range(1, 66)}          # ds004504: AD = participants 1-36
# The frozen partition and folds of the LCMV / wMNE stores (computed by commit 607b5eac, whose folds equal the fold
# columns saved with the earlier results on Drive). Any change of the split code that alters them fails here.
HOLDOUT = ["D3-3", "D3-12", "D3-15", "D3-16", "D3-22", "D3-30", "D3-33", "D3-38", "D3-42", "D3-45", "D3-51",
           "D3-57", "D3-64"]
FOLDS = [["D3-1", "D3-2", "D3-5", "D3-6", "D3-26", "D3-31", "D3-43", "D3-49", "D3-54", "D3-59", "D3-61"],
         ["D3-14", "D3-21", "D3-27", "D3-29", "D3-35", "D3-36", "D3-44", "D3-46", "D3-53", "D3-56", "D3-65"],
         ["D3-8", "D3-9", "D3-13", "D3-20", "D3-23", "D3-25", "D3-37", "D3-39", "D3-40", "D3-52"],
         ["D3-7", "D3-10", "D3-11", "D3-24", "D3-32", "D3-34", "D3-47", "D3-50", "D3-58", "D3-63"],
         ["D3-4", "D3-17", "D3-18", "D3-19", "D3-28", "D3-41", "D3-48", "D3-55", "D3-60", "D3-62"]]


def meta(n_windows=3) -> pd.DataFrame:
    return pd.DataFrame([{"key": k, "dataset": "D3", "y": y, "window_id": f"{k}_w{i}"}
                         for k, y in LABELS.items() for i in range(n_windows)])


def num(keys):
    return sorted(keys, key=lambda k: int(k[3:]))


def test_settings_registry():
    assert set(SETTINGS) == set(XU_SETTINGS) == {"xu_dev_cv", "xu_holdout"}
    with pytest.raises(ValueError):
        make_splits(meta(), "within_D1")
    with pytest.raises(ValueError):
        make_splits(meta(), "xu_dev_cv", scheme="loso")


def test_frozen_partition_and_folds():
    dev, test = xu_partition(subject_table(meta()))
    assert num(test) == HOLDOUT and len(dev) == 52 and not set(dev) & set(test)
    assert sum(LABELS[k] for k in test) == 7                            # 7 AD / 6 HC
    sp = make_splits(meta(), "xu_dev_cv", 5, 1, 0)
    assert [num(s.test_keys) for s in sp] == FOLDS
    (h,) = make_splits(meta(), "xu_holdout", 5, 1, 0)
    assert num(h.test_keys) == HOLDOUT and sorted(h.train_keys) == sorted(dev)


def test_each_development_participant_tested_once_and_holdout_never_in_cv():
    for r_splits in (make_splits(meta(), "xu_dev_cv", 5, 2, 0),):
        for rep in (0, 1):
            tested = [k for s in r_splits if s.repeat == rep for k in s.test_keys]
            assert sorted(tested) == sorted(set(LABELS) - set(HOLDOUT))
            for s in r_splits:
                if s.repeat == rep:
                    assert not set(s.train_keys) & set(s.test_keys)
                    assert not (set(s.train_keys) | set(s.test_keys)) & set(HOLDOUT)


def test_stratification():
    for s in make_splits(meta(), "xu_dev_cv", 5, 1, 0):
        ad = sum(LABELS[k] for k in s.test_keys)
        assert 5 <= ad <= 6 and 4 <= len(s.test_keys) - ad <= 5         # 29 AD / 23 HC over 5 folds


def test_repeats_differ_and_are_reproducible():
    a = make_splits(meta(), "xu_dev_cv", 5, 2, 0)
    b = make_splits(meta(), "xu_dev_cv", 5, 2, 0)
    assert [s.test_keys for s in a] == [s.test_keys for s in b]
    assert [s.test_keys for s in a if s.repeat == 0] != [s.test_keys for s in a if s.repeat == 1]


def test_overlap_is_detected():
    m = meta()
    keys = sorted(set(LABELS) - set(HOLDOUT))
    with pytest.raises(RuntimeError):
        fold_indices(m, Split("xu_dev_cv", 0, 0, tuple(keys[:30]), tuple(keys[25:])))


def test_overlapping_windows_never_cross_folds():
    m = meta(5)
    for s in make_splits(m, "xu_dev_cv", 5, 1, 0):
        tr, te = fold_indices(m, s)
        assert not set(m.iloc[tr]["key"]) & set(m.iloc[te]["key"])
        assert not set(m.iloc[tr]["window_id"]) & set(m.iloc[te]["window_id"])


def test_inner_folds_stay_inside_training():
    m = meta()
    s = make_splits(m, "xu_dev_cv", 5, 1, 0)[0]
    tr, _ = fold_indices(m, s)
    train_meta = m.iloc[tr].reset_index(drop=True)
    for itr, iva in inner_folds(train_meta, 5, 0, ("y",), "y"):
        assert set(train_meta.iloc[iva]["key"]) <= set(s.train_keys)
        assert not set(train_meta.iloc[itr]["key"]) & set(train_meta.iloc[iva]["key"])


def test_setting_subset_is_the_whole_d3_cohort():
    m = meta()
    assert len(setting_subset(m, "xu_dev_cv")) == len(m) == len(setting_subset(m, "xu_holdout"))


def test_subject_table_rejects_inconsistent_labels():
    m = meta()
    m.loc[0, "y"] = 1 - m.loc[0, "y"]
    with pytest.raises(ValueError):
        subject_table(m)


def test_partition_does_not_depend_on_window_counts():
    rng = np.random.default_rng(0)
    m = pd.DataFrame([{"key": k, "dataset": "D3", "y": y, "window_id": f"{k}_w{i}"} for k, y in LABELS.items()
                      for i in range(int(rng.integers(1, 9)))])
    assert num(xu_partition(subject_table(m))[1]) == HOLDOUT
    assert [num(s.test_keys) for s in make_splits(m, "xu_dev_cv", 5, 1, 0)] == FOLDS
