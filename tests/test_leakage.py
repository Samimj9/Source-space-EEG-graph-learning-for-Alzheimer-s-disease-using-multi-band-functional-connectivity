import numpy as np
import pytest

from adgraph.evaluation.splits import fold_indices, make_splits, setting_subset
from adgraph.graphs.dataset import LeakageError, build_classical, build_graph, training_weights
from adgraph.models.classical import ClassicalModel


def _views(data, setting="within_D1"):
    sub = setting_subset(data.meta, setting)
    meta = data.meta.iloc[sub].reset_index(drop=True)
    split = make_splits(meta, setting, 5, 1, 0)[0]
    tr, te = fold_indices(meta, split)
    return data.view(sub[tr], "train"), data.view(sub[te], "test")


def test_fit_on_test_view_is_refused(store):
    data = build_classical(store, {"source": "source", "psd_kinds": ["log"]})
    _, test = _views(data)
    with pytest.raises(LeakageError):
        ClassicalModel("logreg").fit(test)


def test_classical_scaler_uses_training_windows_only(store):
    data = build_classical(store, {"source": "source", "psd_kinds": ["log", "rel"]})
    train, test = _views(data)
    m = ClassicalModel("logreg", {"C": 1.0}).fit(train)
    assert np.allclose(m.pipeline.named_steps["scale"].mean_, train.payload["X"].mean(0))
    p1 = m.predict_proba(test)
    # corrupting test features changes nothing that was fitted
    test.payload["X"][:] = 1e6
    assert np.allclose(m.pipeline.named_steps["scale"].mean_, train.payload["X"].mean(0))
    assert p1.shape == (len(test),)


@pytest.mark.parametrize("step", [{"select_k_best": 20}, {"pca_components": 5}])
def test_selection_and_pca_inside_pipeline(store, step):
    data = build_classical(store, {"source": "sensor", "psd_kinds": ["log"]})
    train, test = _views(data)
    m = ClassicalModel("svm", {"C": 1.0}, {"standardize": True, **step}).fit(train)
    assert len(m.predict_proba(test)) == len(test)


def test_training_weights_one_vote_per_subject_and_balanced(store):
    data = build_classical(store, {"source": "source", "psd_kinds": ["log"]})
    train, _ = _views(data, "within_D2")
    w = training_weights(train.meta)
    per_subj = train.meta.assign(w=w).groupby("key")["w"].sum()
    labels = train.meta.drop_duplicates("key").set_index("key")["y"]
    # all subjects of a class carry equal total weight, and both classes carry equal total weight
    for c in (0, 1):
        assert np.allclose(per_subj[labels == c], per_subj[labels == c].iloc[0])
    assert per_subj[labels == 1].sum() == pytest.approx(per_subj[labels == 0].sum())


def test_graph_scaler_and_sparsifier_fit_on_train_only(store):
    pytest.importorskip("torch")
    from adgraph.models.trainer import GraphSparsifier, NodeFeatureScaler

    data = build_graph(store, {"connectivity": "plv", "bands": ["theta", "beta"], "band_mode": "multi"},
                       {"components": ["constant", "psd"], "psd_kinds": ["log"], "psd_bands": "matched"})
    train, test = _views(data)
    sc = NodeFeatureScaler("node").fit(train)
    xt = sc.transform(train.payload["x"])
    assert np.allclose(xt[..., 1].mean(0), 0, atol=1e-4)                  # PSD standardized on train
    assert np.allclose(xt[..., 0], 1)                                      # constant feature untouched
    with pytest.raises(LeakageError):
        NodeFeatureScaler().fit(test)
    sp = GraphSparsifier("quantile", q=0.9).fit(train)
    iu = np.triu_indices(68, 1)
    assert np.allclose(sp.threshold_, np.quantile(train.payload["adj"][..., iu[0], iu[1]], 0.9, axis=(0, 2)))
    kept = sp.transform(test.payload["adj"])
    assert ((kept == 0) | (kept >= sp.threshold_[None, :, None, None])).all()
    with pytest.raises(LeakageError):
        GraphSparsifier("quantile").fit(test)
    topk = GraphSparsifier("topk", k=5).transform(test.payload["adj"])
    assert np.allclose(topk, np.swapaxes(topk, -1, -2))
    assert ((topk > 0).sum(-1) >= 5).all()
