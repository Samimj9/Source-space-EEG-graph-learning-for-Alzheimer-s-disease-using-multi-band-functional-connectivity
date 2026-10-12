import numpy as np
import pandas as pd
import pytest

from adgraph.evaluation.splits import (SETTINGS, Split, fold_indices, inner_folds, make_splits, setting_subset,
                                       subject_holdout, subject_table)


def meta_for(store, setting):
    return store.windows.iloc[setting_subset(store.windows, setting)].reset_index(drop=True)


@pytest.mark.parametrize("setting", ["within_D1", "within_D2", "pooled"])
def test_each_subject_tested_once_per_repeat(store, setting):
    meta = meta_for(store, setting)
    splits = make_splits(meta, setting, k=5, n_repeats=3, seed=0)
    assert len(splits) == 15
    subjects = set(meta["key"])
    for r in range(3):
        tested = [k for s in splits if s.repeat == r for k in s.test_keys]
        assert sorted(tested) == sorted(subjects)
    for s in splits:
        tr, te = fold_indices(meta, s)
        assert not set(meta.iloc[tr]["key"]) & set(meta.iloc[te]["key"])
        assert not set(meta.iloc[tr]["window_id"]) & set(meta.iloc[te]["window_id"])
        assert len(tr) + len(te) == len(meta)                       # every window used, none twice


def test_within_settings_use_one_dataset(store):
    for setting, ds in (("within_D1", "D1"), ("within_D2", "D2")):
        meta = meta_for(store, setting)
        assert set(meta["dataset"]) == {ds}


def test_stratification(store):
    meta = meta_for(store, "within_D1")                                # 10 AD / 10 HC
    for s in make_splits(meta, "within_D1", 5, 2, 0):
        y = subject_table(meta).set_index("key").loc[list(s.test_keys), "y"]
        assert (y == 1).sum() == 2 and (y == 0).sum() == 2
    pooled = meta_for(store, "pooled")
    st = subject_table(pooled).set_index("key")
    for s in make_splits(pooled, "pooled", 5, 1, 0):
        t = st.loc[list(s.test_keys)]
        assert set(t["dataset"]) == {"D1", "D2"} and set(t["y"]) == {0, 1}


def test_repeats_differ_and_are_reproducible(store):
    meta = meta_for(store, "within_D2")
    a = make_splits(meta, "within_D2", 5, 2, 0)
    b = make_splits(meta, "within_D2", 5, 2, 0)
    assert a == b
    assert {s.test_keys for s in a if s.repeat == 0} != {s.test_keys for s in a if s.repeat == 1}


@pytest.mark.parametrize("setting,src,tgt", [("cross_D1_to_D2", "D1", "D2"), ("cross_D2_to_D1", "D2", "D1")])
def test_cross_dataset(store, setting, src, tgt):
    meta = meta_for(store, setting)
    splits = make_splits(meta, setting, 5, 3, 0)
    assert len(splits) == 3 and len({(s.train_keys, s.test_keys) for s in splits}) == 1
    tr, te = fold_indices(meta, splits[0])
    assert set(meta.iloc[tr]["dataset"]) == {src} and set(meta.iloc[te]["dataset"]) == {tgt}


def test_loso(store):
    meta = meta_for(store, "within_D1")
    splits = make_splits(meta, "within_D1", scheme="loso")
    assert len(splits) == 20 and all(len(s.test_keys) == 1 for s in splits)


def test_overlap_is_detected(store):
    meta = meta_for(store, "within_D1")
    keys = sorted(set(meta["key"]))
    with pytest.raises(RuntimeError):
        fold_indices(meta, Split("within_D1", 0, 0, tuple(keys[:12]), tuple(keys[10:])))


def test_overlapping_windows_never_cross_folds(store):
    """Consecutive 20 s / 50 % windows share 10 s of signal: all windows of a subject stay on one side."""
    meta = meta_for(store, "within_D1")
    for s in make_splits(meta, "within_D1", 5, 2, 0):
        tr, te = fold_indices(meta, s)
        side = pd.Series("train", index=meta.index)
        side.iloc[te] = "test"
        assert (meta.assign(side=side).groupby("key")["side"].nunique() == 1).all()


def test_inner_folds_stay_inside_training(store):
    meta = meta_for(store, "pooled")
    s = make_splits(meta, "pooled", 5, 1, 0)[0]
    tr, _ = fold_indices(meta, s)
    train_meta = meta.iloc[tr].reset_index(drop=True)
    for itr, iva in inner_folds(train_meta, 4, 0, ("dataset", "y")):
        assert set(train_meta.iloc[itr]["key"]).isdisjoint(train_meta.iloc[iva]["key"])
        assert set(train_meta.iloc[iva]["key"]) <= set(s.train_keys)
    a, b = subject_holdout(train_meta, 0.2, 0)
    assert set(train_meta.iloc[a]["key"]).isdisjoint(train_meta.iloc[b]["key"])


def test_subject_table_rejects_inconsistent_labels(store):
    meta = store.windows.copy()
    meta.loc[meta.index[0], "y"] = 1 - meta.loc[meta.index[0], "y"]
    with pytest.raises(ValueError):
        subject_table(meta)


def test_settings_registry():
    assert set(SETTINGS) == {"within_D1", "within_D2", "pooled", "cross_D1_to_D2", "cross_D2_to_D1"}
