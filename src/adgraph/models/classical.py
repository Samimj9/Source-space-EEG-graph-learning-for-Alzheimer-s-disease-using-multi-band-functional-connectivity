"""Logistic regression / SVM on flat features (Experiments 0, 1, baselines, EEG + age).

Every fitted step is owned by the model and fitted on the training view only:

    X (EEG features) -> [StandardScaler -> SelectKBest -> PCA]   (``features``)
    cov (e.g. age)   -> CovariateScaler (subject-level, training subjects)
    [features(X), cov] -> classifier

Covariates are appended *after* the EEG transforms, so they are never selected away or
mixed into PCA components.

Scores (``predict_score``):
* logreg: P(class 1) (``score_type = "probability"``, threshold 0.5);
* svm, ``calibration: none`` (**primary SVM baseline**): the SVC ``decision_function``
  margin (``score_type = "decision_function"``, threshold 0). ROC-AUC and subject-level
  aggregation (mean margin over windows) use this score directly; no probability
  calibration is required;
* svm, ``calibration: grouped_platt`` (optional, only when calibrated probabilities are
  explicitly needed): Platt scaling fitted on out-of-fold decision values from a
  **subject-grouped** internal CV (``GroupedPlattSVC``); no subject's windows are ever on
  both the fitting and the calibration side. sklearn's window-level
  ``CalibratedClassifierCV`` is never used.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from ..graphs.dataset import DataView, require_train, training_weights
from .covariates import CovariateScaler

CLASSIFIERS = ("logreg", "svm")
CALIBRATIONS = ("none", "grouped_platt")


def score_type(name: str, calibration: str = "none") -> str:
    """'probability' or 'decision_function' for a classical model configuration."""
    if calibration not in CALIBRATIONS:
        raise ValueError(f"calibration must be one of {CALIBRATIONS}")
    if name == "svm" and calibration == "none":
        return "decision_function"
    return "probability"


class GroupedPlattSVC(ClassifierMixin, BaseEstimator):
    """SVC with Platt-scaled probabilities, calibrated by subject-grouped internal CV.

    fit(X, y, sample_weight, groups):
      1. StratifiedGroupKFold over ``groups`` (subjects): for each fold an SVC is fitted on
         the other subjects and its decision values are computed for the held-out subjects;
      2. a 1-D logistic (Platt) map is fitted to these out-of-fold decision values;
      3. the final SVC is fitted on all training windows.
    predict_proba(X) = Platt(final SVC decision value).
    """

    def __init__(self, C=1.0, kernel="linear", gamma="scale", calibration_folds=3, random_state=0):
        self.C, self.kernel, self.gamma = C, kernel, gamma
        self.calibration_folds, self.random_state = calibration_folds, random_state

    def _svc(self):
        return SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)

    def fit(self, X, y, sample_weight=None, groups=None):
        if groups is None:
            raise ValueError("GroupedPlattSVC needs subject groups for calibration")
        y, groups = np.asarray(y), np.asarray(groups)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        subj_y = {g: y[groups == g][0] for g in np.unique(groups)}
        n_min = min(sum(1 for v in subj_y.values() if v == c) for c in (0, 1))
        k = min(self.calibration_folds, n_min)
        if k < 2:
            raise ValueError("too few training subjects per class for grouped calibration")
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        dec = np.empty(len(y))
        self.calibration_folds_ = []
        for tr, te in cv.split(X, y, groups):
            if set(groups[tr]) & set(groups[te]):
                raise RuntimeError("calibration fold shares subjects")      # cannot happen; guarded anyway
            dec[te] = self._svc().fit(X[tr], y[tr], sample_weight=w[tr]).decision_function(X[te])
            self.calibration_folds_.append((np.unique(groups[tr]), np.unique(groups[te])))
        self.platt_ = LogisticRegression(C=1e4).fit(dec[:, None], y, sample_weight=w)
        self.svc_ = self._svc().fit(X, y, sample_weight=w)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(X)

    def predict_proba(self, X):
        return self.platt_.predict_proba(self.decision_function(X)[:, None])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_feature_pipeline(preprocessing: dict, n_features: int, seed: int) -> Pipeline:
    steps = []
    if preprocessing.get("standardize", True):
        steps.append(("scale", StandardScaler()))
    sel = preprocessing.get("select_k_best")
    if sel:
        steps.append(("select", SelectKBest(f_classif, k=min(int(sel), n_features))))
    pca = preprocessing.get("pca_components")
    if pca:
        steps.append(("pca", PCA(n_components=pca, random_state=seed)))
    return Pipeline(steps or [("identity", FunctionTransformer())])


def make_classifier(name: str, params: dict, seed: int, calibration: str = "none", calibration_folds: int = 3):
    if name == "logreg":
        return LogisticRegression(C=params.get("C", 1.0), solver="lbfgs", max_iter=10_000)   # L2 (default)
    if name == "svm":
        if calibration == "grouped_platt":
            return GroupedPlattSVC(C=params.get("C", 1.0), kernel=params.get("kernel", "linear"),
                                   gamma=params.get("gamma", "scale"), calibration_folds=calibration_folds,
                                   random_state=seed)
        return SVC(C=params.get("C", 1.0), kernel=params.get("kernel", "linear"), gamma=params.get("gamma", "scale"))
    raise ValueError(f"unknown classifier {name!r}; choose from {CLASSIFIERS}")


class ClassicalModel:
    """Window-level classifier; ``predict_score`` returns one score per window (see module doc).

    Class balance and the one-subject-one-vote rule are applied through sample
    weights (``training_weights``), not through ``class_weight``.
    """

    def __init__(self, name: str, params: dict | None = None, preprocessing: dict | None = None, seed: int = 0,
                 calibration: str = "none", calibration_folds: int = 3):
        self.name, self.params, self.preprocessing, self.seed = name, params or {}, preprocessing or {}, seed
        self.calibration, self.calibration_folds = calibration, calibration_folds
        self.score_type = score_type(name, calibration)
        self.features = self.cov_scaler = self.clf = None

    def _design(self, view: DataView) -> np.ndarray:
        Z = self.features.transform(view.payload["X"])
        if self.cov_scaler is not None:
            Z = np.hstack([Z, self.cov_scaler.transform(view.payload["cov"])])
        return Z

    def fit(self, view: DataView) -> "ClassicalModel":
        require_train(view)
        X, y = view.payload["X"], view.y
        self.features = make_feature_pipeline(self.preprocessing, X.shape[1], self.seed).fit(X, y)
        self.cov_scaler = CovariateScaler().fit(view) if "cov" in view.payload else None
        Z = self._design(view)
        self.clf = make_classifier(self.name, self.params, self.seed, self.calibration, self.calibration_folds)
        w = training_weights(view.meta, view.target)
        if isinstance(self.clf, GroupedPlattSVC):
            self.clf.fit(Z, y, sample_weight=w, groups=view.subjects)
        else:
            self.clf.fit(Z, y, sample_weight=w)
        return self

    def n_parameters(self) -> int:
        """Fitted parameter count of the classifier (feature transforms excluded).

        LR / linear SVM: coefficients + intercept; non-linear SVM: support-vector entries +
        dual coefficients + intercept; grouped Platt adds its 2 sigmoid parameters."""
        clf = self.clf.svc_ if isinstance(self.clf, GroupedPlattSVC) else self.clf
        if isinstance(clf, SVC) and clf.kernel != "linear":
            n = clf.support_vectors_.size + clf.dual_coef_.size + clf.intercept_.size
        else:
            n = clf.coef_.size + np.size(clf.intercept_)
        return int(n + (2 if isinstance(self.clf, GroupedPlattSVC) else 0))

    def predict_score(self, view: DataView) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("model not fitted")
        Z = self._design(view)
        if self.score_type == "decision_function":
            return self.clf.decision_function(Z)
        return self.clf.predict_proba(Z)[:, 1]

    def predict_proba(self, view: DataView) -> np.ndarray:
        if self.score_type != "probability":
            raise ValueError("uncalibrated SVM has no probabilities; use predict_score (decision_function) or set "
                             "model.calibration: grouped_platt")
        return self.predict_score(view)
