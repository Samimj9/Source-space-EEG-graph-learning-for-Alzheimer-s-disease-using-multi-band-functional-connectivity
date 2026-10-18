"""Fit/predict wrapper for the GNNs (Exp 2-8), owning every fitted transform.

``GNNModel.fit(train_view)`` (the final protocol ``gnn_fixed60_v1``; DESIGN.md, docs/final/framework.md):
  1. fits ``NodeFeatureScaler`` on the training windows only;
  2. fits the graph construction (``graphs.construction.GraphConstruction``; only ``quantile`` has parameters) on the
     training windows only and applies it to every view;
  3. trains for exactly ``training.epochs`` epochs (60) with Adam, subject- and class-balanced binary cross-entropy,
     shuffled mini-batches from a generator seeded by the model seed; the model after the last epoch is used (no
     early stopping, no epoch selection, no validation split);
  4. records the training history: per epoch the training loss and the mean gradient norm, then the training-set
     window and participant ROC-AUC of the final model (diagnostics only; never used to select anything).
``predict_proba(view)`` applies the fitted transforms and returns P(AD) = sigmoid(logit) per window (AD = 1).
"""

from __future__ import annotations

import json
import random

import numpy as np

from ..graphs.construction import GraphConstruction
from ..graphs.dataset import DataView, require_train, training_weights


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NodeFeatureScaler:
    """Z-scores node features with training statistics.

    scope ``node``: one mean/std per (band graph, ROI, feature); ``global``: per (band graph, feature) pooled over
    ROIs; ``none``: identity. Features with zero training variance (e.g. the constant feature, a one-hot band
    indicator) are left untouched.
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


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


class GNNModel:
    """sklearn-like estimator around the GNN architectures; scores are P(AD) (threshold 0.5)."""

    score_type = "probability"

    def __init__(self, model: dict, training: dict, node_scaling: str = "node", graph: dict | None = None,
                 seed: int = 0, device: str | None = None, roi_names: list | None = None):
        es = (training or {}).get("early_stopping") or {}
        if es.get("enabled"):
            raise ValueError("early stopping is not part of the final protocol (fixed epochs only)")
        if int(training.get("epochs", 0)) < 1:
            raise ValueError("training.epochs must be a positive, fixed number of epochs")
        self.model_cfg, self.train_cfg, self.seed = dict(model), dict(training), seed
        self.scaler = NodeFeatureScaler(node_scaling)
        self.graph = GraphConstruction.from_config(graph, roi_names)
        self.device = device
        self.net = None
        self.history: list[dict] = []
        self.train_metrics: dict = {}

    # -- helpers
    def _device(self):
        import torch

        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _tensors(self, view: DataView):
        import torch

        x = self.scaler.transform(view.payload["x"])
        adj = self.graph.transform(view.payload["adj"])
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(adj))

    def build_net(self, in_dim: int, n_bands: int, n_rois: int):
        cfg = dict(self.model_cfg)
        arch = cfg.pop("architecture", "multiband_gnn")
        cfg.pop("encoder_label", None)
        if arch == "earm_gnn":                                # Experiment 7 (methodology log entry 43)
            from .earm import EARMGNN

            cfg.pop("encoder", None)
            return EARMGNN(in_dim=in_dim, n_bands=n_bands, n_rois=n_rois, **cfg)
        if arch == "xu_dual_gcn":                             # Experiment 8 (methodology log entry 52)
            from .xu_dual import XuDualGCN

            cfg.pop("encoder", None)
            return XuDualGCN(in_dim=in_dim, n_bands=n_bands, n_rois=n_rois, **cfg)
        if arch != "multiband_gnn":
            raise ValueError(f"unknown architecture {arch!r}")
        from .multiband import MultiBandGNN

        cfg.pop("roi_names", None)
        return MultiBandGNN(encoder=cfg.pop("encoder", "gcn"), in_dim=in_dim, n_bands=n_bands, **cfg)

    # -- API
    def fit(self, view: DataView) -> "GNNModel":
        """Fit on a TRAINING view only, for exactly ``training.epochs`` epochs."""
        import torch
        from torch.nn import functional as F

        from .. import progress

        require_train(view)
        seed_everything(self.seed)
        self.scaler.fit(view)
        self.graph.fit(view)
        dev = self._device()
        x, adj = self._tensors(view)
        y = torch.from_numpy(view.y.astype(np.float32))
        w = torch.from_numpy(training_weights(view.meta, view.target).astype(np.float32))
        self.net = self.build_net(x.shape[-1], x.shape[1], x.shape[2]).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.train_cfg.get("lr", 1e-3),
                               weight_decay=self.train_cfg.get("weight_decay", 1e-4))
        bs = int(self.train_cfg.get("batch_size", 32))
        epochs = int(self.train_cfg["epochs"])
        gen = torch.Generator().manual_seed(self.seed)
        x, adj, y, w = x.to(dev), adj.to(dev), y.to(dev), w.to(dev)   # the training set on the device once
        self.history = []
        for epoch in range(epochs):
            progress.set(epoch=f"{epoch + 1}/{epochs}")
            self.net.train()
            perm = torch.randperm(len(y), generator=gen)
            total, gnorm, nb = 0.0, 0.0, 0
            for i in range(0, len(y), bs):
                idx = perm[i:i + bs].to(dev)
                logits = self.net(x[idx], adj[idx])
                # weights have mean 1 over the training set: dividing by the batch size keeps one vote per
                # participant and class balance across batches
                bce = F.binary_cross_entropy_with_logits(logits, y[idx], reduction="none")
                loss = (bce * w[idx]).sum() / len(idx)
                if hasattr(self.net, "regularization"):          # EARM-GNN edge-gate penalty
                    loss = loss + self.net.regularization()
                opt.zero_grad()
                loss.backward()
                gnorm += float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in self.net.parameters()
                                              if p.grad is not None)))
                opt.step()
                total += loss.item() * len(idx)
                nb += 1
            self.history.append({"epoch": epoch + 1, "train_loss": total / len(y), "grad_norm": gnorm / max(nb, 1)})
        progress.set(epoch=None)
        p = self.predict_proba(view)                       # training-set fit of the final model (diagnostic only)
        subj = view.meta[["key"]].assign(y=view.y, score=p).groupby("key").agg(y=("y", "first"), score=("score", "mean"))
        self.train_metrics = {"train_window_auc": _auc(view.y, p), "train_participant_auc": _auc(subj["y"], subj["score"])}
        return self

    def training_summary(self) -> dict:
        """Per-fit training record (one row of every outer fold's ``training.csv``): epochs, the loss curve, the
        relative loss change over the last 10 epochs (convergence), the gradient norm, the training-set AUC of the
        final model and the structure of the constructed graphs."""
        loss = [h["train_loss"] for h in self.history]
        tail = loss[-11:]
        rel = (tail[0] - tail[-1]) / abs(tail[0]) if len(tail) > 1 and tail[0] else float("nan")
        return {"epochs": len(self.history), "final_train_loss": loss[-1] if loss else float("nan"),
                "loss_rel_change_last10": rel, "final_grad_norm": self.history[-1]["grad_norm"] if loss else float("nan"),
                **self.train_metrics,
                **{f"graph_{k}": v for k, v in self.graph.stats_.items()},
                "history_train_loss": json.dumps([round(v, 6) for v in loss]),
                "history_grad_norm": json.dumps([round(h["grad_norm"], 6) for h in self.history])}

    def predict_proba(self, view: DataView, return_band_weights: bool = False):
        import torch

        if self.net is None:
            raise RuntimeError("model not fitted")
        dev = self._device()
        x, adj = self._tensors(view)
        self.net.eval()
        probs, weights = [], []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                logits, bw = self.net(x[i:i + 256].to(dev), adj[i:i + 256].to(dev), return_band_weights=True)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                weights.append(bw.cpu().numpy())
        p = np.concatenate(probs) if probs else np.empty(0)
        if return_band_weights:
            return p, (np.concatenate(weights) if weights else np.empty((0, x.shape[1])))
        return p

    def state_arrays(self) -> dict:
        """Fitted parameters, learned edge gates (EARM-GNN), training history and scaler statistics, as arrays
        (``evaluation.save_model_state``: one ``units/<fold>.model.npz`` per outer fold)."""
        out = {f"param__{k}": v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}
        if hasattr(self.net, "gates"):
            import torch

            with torch.no_grad():
                g = self.net.gates()
            if g is not None:
                out["edge_gates"] = g.cpu().numpy()
        out["history_epoch"] = np.array([h["epoch"] for h in self.history])
        out["history_train_loss"] = np.array([h["train_loss"] for h in self.history])
        out["history_grad_norm"] = np.array([h["grad_norm"] for h in self.history])
        if self.scaler.mean_ is not None:
            out["node_scaler_mean"], out["node_scaler_std"] = self.scaler.mean_, self.scaler.std_
        if self.graph.threshold_ is not None:
            out["graph_quantile_threshold"] = self.graph.threshold_
        return out

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def predict_score(self, view: DataView, return_band_weights: bool = False):
        return self.predict_proba(view, return_band_weights)
