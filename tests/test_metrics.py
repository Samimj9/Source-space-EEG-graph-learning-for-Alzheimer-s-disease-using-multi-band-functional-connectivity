import numpy as np
import pandas as pd
import pytest

from adgraph.evaluation.metrics import (aggregate_subjects, binary_metrics, bootstrap_ci, corrected_resampled_ttest,
                                        permutation_p_value)


def test_binary_metrics_known_values():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.6, 0.2, 0.1, 0.4])
    m = binary_metrics(y, p)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 3)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(3 / 4)
    assert m["accuracy"] == pytest.approx(5 / 7)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 3 / 4) / 2)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * (2 / 3) / (4 / 3))
    assert m["roc_auc"] == pytest.approx(10 / 12)   # 10 of 12 AD/HC pairs ranked correctly


def test_single_class_auc_is_nan():
    assert np.isnan(binary_metrics(np.ones(4), np.array([0.2, 0.6, 0.7, 0.9]))["roc_auc"])


def test_aggregate_subjects_mean_of_windows():
    pred = pd.DataFrame({"key": ["a", "a", "a", "b", "b"], "dataset": ["D1"] * 5, "y": [1, 1, 1, 0, 0],
                         "prob": [0.9, 0.6, 0.3, 0.2, 0.4]})
    s = aggregate_subjects(pred).set_index("key")
    assert s.loc["a", "prob"] == pytest.approx(0.6) and s.loc["a", "n_windows"] == 3
    assert s.loc["b", "prob"] == pytest.approx(0.3)
    bad = pred.assign(y=[1, 0, 1, 0, 0])
    with pytest.raises(ValueError):
        aggregate_subjects(bad)


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20), np.zeros(20)]
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 40), 0, 1)
    ci = bootstrap_ci(y, p, n_boot=300)
    auc = binary_metrics(y, p)["roc_auc"]
    assert ci["roc_auc"][0] <= auc <= ci["roc_auc"][1]


def test_corrected_ttest_and_permutation_p():
    r = corrected_resampled_ttest(np.array([0.05, 0.02, 0.04, 0.03, 0.06]), n_train=16, n_test=4)
    assert r["t"] > 0 and 0 < r["p"] < 1 and r["df"] == 4
    assert permutation_p_value(0.9, np.array([0.5, 0.6, 0.95])) == pytest.approx(2 / 4)
