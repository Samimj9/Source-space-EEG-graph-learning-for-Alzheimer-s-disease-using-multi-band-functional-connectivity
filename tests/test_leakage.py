"""Leakage rules of the final framework: everything fitted lives in a model whose fit accepts only a training view of
the outer training participants (scalers, selection, PCA, calibration, node scaling, graph-construction thresholds);
participant-balanced training weights; the primary SVM is scored by its margin. Synthetic ds004504 store."""

import numpy as np
import pytest

from adgraph.evaluation.splits import fold_indices, make_splits, setting_subset
from adgraph.graphs.dataset import LeakageError, build_classical, build_graph, training_weights
from adgraph.models.classical import ClassicalModel


def _views(data, setting="xu_dev_cv"):
    sub = setting_subset(data.meta, setting)
    meta = data.meta.iloc[sub].reset_index(drop=True)
    split = make_splits(meta, setting, 5, 1, 0)[0]
    tr, te = fold_indices(meta, split)
    return data.view(sub[tr], "train"), data.view(sub[te], "test")


def test_fit_on_test_view_is_refused(d3_store):
    data = build_classical(d3_store, {"source": "source", "psd_kinds": ["log"]})
    _, test = _views(data)
    with pytest.raises(LeakageError):
        ClassicalModel("logreg").fit(test)


def test_classical_scaler_uses_training_windows_only(d3_store):
    data = build_classical(d3_store, {"source": "source", "psd_kinds": ["log", "rel"]})
    train, test = _views(data)
    m = ClassicalModel("logreg", {"C": 1.0}).fit(train)
    assert np.allclose(m.features.named_steps["scale"].mean_, train.payload["X"].mean(0))
    p1 = m.predict_proba(test)
    # corrupting test features changes nothing that was fitted
    test.payload["X"][:] = 1e6
    assert np.allclose(m.features.named_steps["scale"].mean_, train.payload["X"].mean(0))
    assert p1.shape == (len(test),)


@pytest.mark.parametrize("step", [{"select_k_best": 20}, {"pca_components": 5}])
def test_selection_and_pca_inside_pipeline(d3_store, step):
    data = build_classical(d3_store, {"source": "sensor", "psd_kinds": ["log"]})
    train, test = _views(data)
    m = ClassicalModel("svm", {"C": 1.0}, {"standardize": True, **step}).fit(train)
    assert len(m.predict_score(test)) == len(test)


def test_training_weights_one_vote_per_subject_and_balanced(d3_store):
    data = build_classical(d3_store, {"source": "source", "psd_kinds": ["log"]})
    train, _ = _views(data, "xu_holdout")
    w = training_weights(train.meta)
    per_subj = train.meta.assign(w=w).groupby("key")["w"].sum()
    labels = train.meta.drop_duplicates("key").set_index("key")["y"]
    # all subjects of a class carry equal total weight, and both classes carry equal total weight
    for c in (0, 1):
        assert np.allclose(per_subj[labels == c], per_subj[labels == c].iloc[0])
    assert per_subj[labels == 1].sum() == pytest.approx(per_subj[labels == 0].sum())


def test_graph_scaler_and_construction_fit_on_train_only(d3_store):
    pytest.importorskip("torch")
    from adgraph.graphs.construction import GraphConstruction
    from adgraph.models.trainer import NodeFeatureScaler

    data = build_graph(d3_store, {"connectivity": "plv", "bands": ["theta", "beta"], "band_mode": "multi"},
                       {"components": ["constant", "psd"], "psd_kinds": ["log"], "psd_bands": "matched"})
    train, test = _views(data)
    sc = NodeFeatureScaler("node").fit(train)
    xt = sc.transform(train.payload["x"])
    assert np.allclose(xt[..., 1].mean(0), 0, atol=1e-4)                  # PSD standardized on train
    assert np.allclose(xt[..., 0], 1)                                      # constant feature untouched
    with pytest.raises(LeakageError):
        NodeFeatureScaler().fit(test)
    gc = GraphConstruction("quantile", q=0.9).fit(train)
    iu = np.triu_indices(68, 1)
    assert np.allclose(gc.threshold_, np.quantile(train.payload["adj"][..., iu[0], iu[1]], 0.9, axis=(0, 2)))
    kept = gc.transform(test.payload["adj"])
    assert ((kept == 0) | (kept >= gc.threshold_[None, :, None, None])).all()
    with pytest.raises(LeakageError):
        GraphConstruction("quantile").fit(test)
    knn = GraphConstruction("knn", k=5).transform(test.payload["adj"])     # stateless: nothing to fit
    assert np.allclose(knn, np.swapaxes(knn, -1, -2)) and ((knn > 0).sum(-1) >= 5).all()


def test_gnn_training_uses_the_training_view_only(d3_store):
    pytest.importorskip("torch")
    from adgraph.models.trainer import GNNModel

    data = build_graph(d3_store, {"connectivity": "plv", "bands": ["theta", "beta"], "band_mode": "multi"},
                       {"components": ["psd"], "psd_kinds": ["log"], "psd_bands": "matched"})
    train, test = _views(data)
    m = GNNModel({"encoder": "gcn", "hidden": 8, "n_layers": 1, "fusion": "mean", "band_identity": "onehot"},
                 {"epochs": 1, "batch_size": 16}, seed=0).fit(train)
    before = m.predict_proba(test)
    mean = m.scaler.mean_.copy()
    test.payload["x"][:] = 1e6                                             # corrupting test inputs ...
    assert np.array_equal(m.scaler.mean_, mean)                            # ... changes nothing fitted
    assert before.shape == (len(test),)
    with pytest.raises(LeakageError):
        GNNModel({"encoder": "gcn", "hidden": 8, "n_layers": 1}, {"epochs": 1}).fit(test)


def test_primary_svm_uses_decision_function(d3_store):
    from sklearn.svm import SVC

    data = build_classical(d3_store, {"source": "source", "psd_kinds": ["log"]})
    train, test = _views(data, "xu_holdout")
    m = ClassicalModel("svm", {"C": 1.0}).fit(train)
    assert m.score_type == "decision_function" and isinstance(m.clf, SVC)
    s = m.predict_score(test)
    assert np.allclose(s, m.clf.decision_function(m._design(test)))
    assert (s < 0).any() and (s > 0).any()                               # margins, not probabilities
    with pytest.raises(ValueError, match="grouped_platt"):
        m.predict_proba(test)


def test_svm_calibration_is_subject_grouped(d3_store):
    data = build_classical(d3_store, {"source": "source", "psd_kinds": ["log"]})
    train, test = _views(data, "xu_holdout")
    m = ClassicalModel("svm", {"C": 1.0}, calibration="grouped_platt").fit(train)
    assert m.score_type == "probability"
    folds = m.clf.calibration_folds_
    assert len(folds) == 3
    for fit_subjects, cal_subjects in folds:
        assert set(fit_subjects).isdisjoint(cal_subjects)                 # no subject on both sides
    assert sorted(np.concatenate([c for _, c in folds])) == sorted(set(train.subjects))
    p = m.predict_proba(test)
    assert p.shape == (len(test),) and ((p > 0) & (p < 1)).all()


def test_grouped_platt_requires_groups(d3_store):
    from adgraph.models.classical import GroupedPlattSVC

    with pytest.raises(ValueError, match="groups"):
        GroupedPlattSVC().fit(np.random.rand(20, 3), np.r_[np.zeros(10), np.ones(10)])
