"""Training-fold-only standardization of subject-level covariates (currently: age).

Statistics are computed over the *training subjects* (one value per subject, so
subjects with many windows do not dominate) and applied unchanged to validation
and test views.
"""

from __future__ import annotations

import numpy as np

from ..graphs.dataset import DataView, require_train


class CovariateScaler:
    def __init__(self):
        self.mean_ = self.std_ = None

    def fit(self, view: DataView) -> "CovariateScaler":
        require_train(view, "CovariateScaler.fit")
        cov = view.payload["cov"].astype(np.float64)
        _, first = np.unique(view.subjects, return_index=True)
        per_subject = cov[first]
        self.mean_ = per_subject.mean(axis=0)
        std = per_subject.std(axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, cov: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("covariate scaler not fitted")
        return ((cov - self.mean_) / self.std_).astype(np.float32)
