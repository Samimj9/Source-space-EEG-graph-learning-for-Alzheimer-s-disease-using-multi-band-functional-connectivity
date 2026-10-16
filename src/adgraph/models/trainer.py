"""Fit/predict wrapper for the GNNs, owning every fitted transform.

``GNNModel.fit(train_view)``:
  1. fits ``NodeFeatureScaler`` on training windows only,
  2. fits the graph sparsifier on training windows only (if it has parameters),
  2b. fits the covariate (age) scaler on training subjects only, if covariates are present,
  3. optionally splits *training subjects* into train/validation for early stopping,
  4. trains with subject- and class-balanced sample weights.
``predict_proba(view)`` applies the fitted transforms and returns P(class 1) per window.
"""

from __future__ import annotations

import copy
import random

import numpy as np

from ..graphs.dataset import DataView, require_train, training_weights
from .covariates import CovariateScaler


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NodeFeatureScaler:
    """Z-scores node features with training statistics.

    scope ``node``: one mean/std per (band graph, ROI, feature); ``global``: per
    (band graph, feature) pooled over ROIs; ``none``: identity. Features with zero
    training variance (e.g. the constant feature) are left untouched.
    """

    def __init__(self, scope: str = "node"):
        if scope not in ("node", "global", "none"):
            raise ValueError("scope must be node | global | none")
        self.scope = scope
        self.mean_ = self.std_ = None

    def fit(self, view: DataView) -> "NodeFeatureScaler":
        require_train(view, "NodeFeatureScaler.fit")
        x = view.payload["x"].astype(np.float64)
        axes = (0,) if self.scope == "node" else (0, 2)
        mean, std = x.mean(axis=axes, keepdims=True), x.std(axis=axes, keepdims=True)
        const = std < 1e-12
        self.mean_ = np.where(const, 0.0, mean)[0]
        self.std_ = np.where(const, 1.0, std)[0]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.scope == "none":
            return x
        if self.mean_ is None:
            raise RuntimeError("scaler not fitted")
        return ((x - self.mean_) / self.std_).astype(np.float32)


class GraphSparsifier:
    """Optional edge sparsification of the weighted adjacency.

    ``none``      dense graphs (default)
    ``topk``      per graph, keep each node's k strongest edges (symmetrized by union); stateless
    ``quantile``  keep edges >= the q-quantile of *training* edge weights, per band graph; fitted
    """

    def __init__(self, method: str = "none", k: int = 10, q: float = 0.8):
        if method not in ("none", "topk", "quantile"):
            raise ValueError("method must be none | topk | quantile")
        self.method, self.k, self.q = method, k, q
        self.threshold_ = None

    def fit(self, view: DataView) -> "GraphSparsifier":
        require_train(view, "GraphSparsifier.fit")
        if self.method == "quantile":
            a = view.payload["adj"]
            iu = np.triu_indices(a.shape[-1], 1)
            self.threshold_ = np.quantile(a[..., iu[0], iu[1]], self.q, axis=(0, 2))   # (K,)
        return self

    def transform(self, adj: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return adj
        if self.method == "quantile":
            if self.threshold_ is None:
                raise RuntimeError("sparsifier not fitted")
            return np.where(adj >= self.threshold_[None, :, None, None], adj, 0).astype(np.float32)
        r = adj.shape[-1]
        kth = np.sort(adj, axis=-1)[..., r - self.k][..., None]
        keep = adj >= kth
        keep = keep | np.swapaxes(keep, -1, -2)
        return np.where(keep, adj, 0).astype(np.float32)


class GNNModel:
    """sklearn-like estimator around ``MultiBandGNN``; scores are P(class 1) (threshold 0.5)."""

    score_type = "probability"

    def __init__(self, model: dict, training: dict, node_scaling: str = "node", sparsify: dict | None = None,
                 seed: int = 0, device: str | None = None):
        self.model_cfg, self.train_cfg, self.seed = dict(model), dict(training), seed
        self.scaler = NodeFeatureScaler(node_scaling)
        self.sparsifier = GraphSparsifier(**(sparsify or {"method": "none"}))
        self.device = device
        self.cov_scaler: CovariateScaler | None = None
        self.net = None
        self.history: list[dict] = []

    # -- helpers
    def _device(self):
        import torch

        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _tensors(self, view: DataView):
        import torch

        x = self.scaler.transform(view.payload["x"])
        adj = self.sparsifier.transform(view.payload["adj"])
        cov = None
        if self.cov_scaler is not None:
            cov = torch.from_numpy(self.cov_scaler.transform(view.payload["cov"]))
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(adj)), cov

    def build_net(self, in_dim: int, n_bands: int, n_covariates: int = 0, n_rois: int | None = None):
        cfg = dict(self.model_cfg)
        if cfg.get("architecture") == "earm_gnn":            # Experiment 7 (methodology log entry 43)
            from .earm import EARMGNN

            cfg.pop("architecture")
            cfg.pop("encoder", None)                          # display label only ("earm")
            return EARMGNN(in_dim=in_dim, n_bands=n_bands, n_rois=n_rois, n_covariates=n_covariates, **cfg)
        if cfg.get("architecture") == "xu_dual_gcn":         # Experiment 8, LCMV / Xu suite (entry 52)
            from .xu_dual import XuDualGCN

            cfg.pop("architecture")
            cfg.pop("encoder", None)                          # display label only
            return XuDualGCN(in_dim=in_dim, n_bands=n_bands, n_rois=n_rois, n_covariates=n_covariates, **cfg)
        from .multiband import MultiBandGNN

        return MultiBandGNN(encoder=cfg.pop("encoder"), in_dim=in_dim, n_bands=n_bands, n_covariates=n_covariates,
                            **cfg)

    # -- API
    def fit(self, view: DataView) -> "GNNModel":
        import torch
        from torch.nn import functional as F

        from ..evaluation.splits import subject_holdout

        require_train(view)
        seed_everything(self.seed)
        es = self.train_cfg.get("early_stopping", {}) or {}
        if es.get("enabled", False):
            tr_idx, va_idx = subject_holdout(view.meta, es.get("val_subject_fraction", 0.2), self.seed, view.target)
            fit_view, val_view = view.take(tr_idx, "train"), view.take(va_idx, "val")
        else:
            fit_view, val_view = view, None

        self.scaler.fit(fit_view)
        self.sparsifier.fit(fit_view)
        self.cov_scaler = CovariateScaler().fit(fit_view) if "cov" in fit_view.payload else None
        dev = self._device()
        x, adj, cov = self._tensors(fit_view)
        n_cov = 0 if cov is None else cov.shape[1]
        y = torch.from_numpy(fit_view.y.astype(np.float32))
        w = torch.from_numpy(training_weights(fit_view.meta, fit_view.target).astype(np.float32))
        self.net = self.build_net(x.shape[-1], x.shape[1], n_cov, n_rois=x.shape[2]).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.train_cfg.get("lr", 1e-3),
                               weight_decay=self.train_cfg.get("weight_decay", 1e-4))
        bs, epochs = self.train_cfg.get("batch_size", 32), self.train_cfg.get("epochs", 100)
        gen = torch.Generator().manual_seed(self.seed)
        # whole training set on the device once (not copied per batch); identical batches and results
        x, adj, y, w = x.to(dev), adj.to(dev), y.to(dev), w.to(dev)
        cov = None if cov is None else cov.to(dev)
        best, best_state, patience = np.inf, None, 0
        from .. import progress

        for epoch in range(epochs):
            progress.set(epoch=f"{epoch + 1}/{epochs}")
            self.net.train()
            perm = torch.randperm(len(y), generator=gen)
            total = 0.0
            for i in range(0, len(y), bs):
                idx = perm[i:i + bs].to(dev)
                logits = self.net(x[idx], adj[idx], None if cov is None else cov[idx])
                # weights have mean 1 over the training set: dividing by the batch size (not the batch's
                # weight sum) keeps one vote per subject and class balance across batches
                bce = F.binary_cross_entropy_with_logits(logits, y[idx], reduction="none")
                loss = (bce * w[idx]).sum() / len(idx)
                if hasattr(self.net, "regularization"):          # e.g. EARM-GNN edge-gate penalty
                    loss = loss + self.net.regularization()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            rec = {"epoch": epoch, "train_loss": total / len(y)}
            if val_view is not None:
                rec["val_loss"] = self._weighted_loss(val_view)
                if rec["val_loss"] < best - 1e-6:
                    best, best_state, patience = rec["val_loss"], copy.deepcopy(self.net.state_dict()), 0
                else:
                    patience += 1
            self.history.append(rec)
            if val_view is not None and patience >= es.get("patience", 15):
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        progress.set(epoch=None)
        return self

    def _weighted_loss(self, view: DataView) -> float:
        import torch
        from torch.nn import functional as F

        p = torch.from_numpy(self.predict_proba(view)).clamp(1e-7, 1 - 1e-7)
        y = torch.from_numpy(view.y.astype(np.float32))
        # validation subjects get equal weight; class balance as in training
        w = torch.from_numpy(training_weights(view.meta, view.target).astype(np.float32))
        return float((F.binary_cross_entropy(p.float(), y, reduction="none") * w).sum() / w.sum())

    def predict_proba(self, view: DataView, return_band_weights: bool = False):
        import torch

        if self.net is None:
            raise RuntimeError("model not fitted")
        dev = self._device()
        x, adj, cov = self._tensors(view)
        self.net.eval()
        probs, weights = [], []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                c = None if cov is None else cov[i:i + 256].to(dev)
                logits, bw = self.net(x[i:i + 256].to(dev), adj[i:i + 256].to(dev), c, return_band_weights=True)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                weights.append(bw.cpu().numpy())
        p = np.concatenate(probs) if probs else np.empty(0)
        return (p, np.concatenate(weights)) if return_band_weights else p

    def state_arrays(self) -> dict:
        """Fitted parameters, learned edge gates (if any) and training history, as arrays (for
        ``evaluation.save_model_state``: one ``units/<fold>.model.npz`` per outer fold)."""
        out = {f"param__{k}": v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}
        if hasattr(self.net, "gates"):
            import torch

            with torch.no_grad():
                g = self.net.gates()
            if g is not None:
                out["edge_gates"] = g.cpu().numpy()
        out["history_epoch"] = np.array([h["epoch"] for h in self.history])
        out["history_train_loss"] = np.array([h["train_loss"] for h in self.history])
        if self.scaler.mean_ is not None:
            out["node_scaler_mean"], out["node_scaler_std"] = self.scaler.mean_, self.scaler.std_
        return out

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def embed(self, view: DataView) -> np.ndarray:
        """Fused graph embeddings (N, D) of a view, from this fitted model."""
        import torch

        x, adj, _ = self._tensors(view)
        dev = self._device()
        self.net.eval()
        with torch.no_grad():
            return np.concatenate([self.net.fused_embedding(x[i:i + 256].to(dev), adj[i:i + 256].to(dev)).cpu().numpy()
                                   for i in range(0, len(x), 256)]) if len(x) else np.empty((0, 0))

    def predict_score(self, view: DataView, return_band_weights: bool = False):
        return self.predict_proba(view, return_band_weights)
