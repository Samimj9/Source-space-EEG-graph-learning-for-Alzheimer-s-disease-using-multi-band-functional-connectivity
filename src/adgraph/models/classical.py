"""Logistic regression / SVM on flat features (Experiments 0, 1 and the age baseline).

All fitted preprocessing (standardization, optional univariate feature selection,
optional PCA) is inside one sklearn Pipeline owned by the model, so it is fitted
on the training view only and merely applied to validation/test views.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..graphs.dataset import DataView, require_train, training_weights

CLASSIFIERS = ("logreg", "svm")


def make_pipeline(name: str, params: dict, preprocessing: dict, n_features: int, seed: int) -> Pipeline:
    steps = []
    if preprocessing.get("standardize", True):
        steps.append(("scale", StandardScaler()))
    sel = preprocessing.get("select_k_best")
    if sel:
        steps.append(("select", SelectKBest(f_classif, k=min(int(sel), n_features))))
    pca = preprocessing.get("pca_components")
    if pca:
        steps.append(("pca", PCA(n_components=pca, random_state=seed)))
    if name == "logreg":
        clf = LogisticRegression(C=params.get("C", 1.0), solver="lbfgs", max_iter=10_000)   # L2 (default)
    elif name == "svm":
        # Probabilities by Platt scaling, fitted with internal CV on the *training* windows only.
        svc = SVC(C=params.get("C", 1.0), kernel=params.get("kernel", "linear"), gamma=params.get("gamma", "scale"))
        clf = CalibratedClassifierCV(svc, method="sigmoid", cv=3, ensemble=False)
    else:
        raise ValueError(f"unknown classifier {name!r}; choose from {CLASSIFIERS}")
    steps.append(("clf", clf))
    return Pipeline(steps)


class ClassicalModel:
    """Window-level classifier; ``predict_proba`` returns P(class 1) per window.

    Class balance and the one-subject-one-vote rule are applied through sample
    weights (``training_weights``), not through ``class_weight``.
    """

    def __init__(self, name: str, params: dict | None = None, preprocessing: dict | None = None, seed: int = 0):
        self.name, self.params, self.preprocessing, self.seed = name, params or {}, preprocessing or {}, seed
        self.pipeline: Pipeline | None = None

    def fit(self, view: DataView) -> "ClassicalModel":
        require_train(view)
        X, y = view.payload["X"], view.y
        self.pipeline = make_pipeline(self.name, self.params, self.preprocessing, X.shape[1], self.seed)
        self.pipeline.fit(X, y, clf__sample_weight=training_weights(view.meta, view.target))
        return self

    def predict_proba(self, view: DataView) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model not fitted")
        return self.pipeline.predict_proba(view.payload["X"])[:, 1]
