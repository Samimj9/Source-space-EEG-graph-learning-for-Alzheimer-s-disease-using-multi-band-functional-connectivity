"""Graph construction before GNN classification (docs/final/graph_construction.md).

Input: the weighted band graphs of a window, adj (N, K, R, R), from functional connectivity (PLV or |ciPLV|:
symmetric, zero diagonal, in [0, 1]) or, as an explicitly different alternative, from node-feature similarity
(``feature_similarity_adjacency``). Output: the adjacency the GNN propagates on, same shape and dtype.

Methods (``graph_construction`` block of an experiment run; ``dense`` is the reference):

    dense         the weighted graph as is (reference baseline)
    knn           per node, its k strongest edges; ``symmetrize``:
                    union   i~j if j is among i's k strongest OR i among j's (symmetric; degree >= k)
                    mutual  i~j if both (symmetric; degree <= k); nodes left with fewer than ``min_degree`` edges get
                            their strongest edge(s) back (symmetrically), so no node is isolated
                    none    the directed kNN graph (row i keeps i's k strongest); NOT symmetric, only for analysis
    proportional  the strongest m = round(density * R (R - 1) / 2) edges of each graph (global proportional
                  threshold); ``density`` defaults to k / (R - 1), the mean degree of a k-NN graph
    quantile      edges >= the q-quantile of TRAINING edge weights per band graph (the only fitted method; the
                  threshold is fitted on the training view in ``fit``)

``anatomical`` (optional, any method): the weights are first restricted to an anatomical mask
    contiguity        regions sharing a cortical-surface edge on fsaverage (``source.atlas.contiguity``)
    lobe_hemisphere   regions of the same lobe and hemisphere (``source.atlas.lobe_of``)

k is given directly (``k``) or relative to the number of ROIs (``k_fraction``: k = max(1, round(f (R - 1))), so the
same pre-specified range applies to every atlas). Retained edges keep their original weights (never binarized),
the diagonal stays zero (self-loops are added by the GCN normalization D^-1/2 (A + I) D^-1/2, or not at all by
EARM-GNN, as before), and no weight becomes negative. Band order and ROI order are unchanged (every operation is per
band graph and per window). Stateless methods depend on each window alone, so they cannot leak information between
participants; ``quantile`` is fitted on training windows only (``require_train``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dataset import DataView, require_train

METHODS = ("dense", "knn", "proportional", "quantile")
SYMMETRIZE = ("union", "mutual", "none")
ANATOMICAL = ("none", "contiguity", "lobe_hemisphere")
BLOCK = 256                                    # windows processed at a time (bounds temporaries)


@dataclass
class GraphConstruction:
    method: str = "dense"
    k: int | None = None
    k_fraction: float | None = None
    symmetrize: str = "union"
    min_degree: int = 1
    density: float | None = None
    q: float = 0.8
    anatomical: str = "none"
    roi_names: list | None = None
    threshold_: np.ndarray | None = field(default=None, repr=False)
    stats_: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"graph construction method must be one of {METHODS}, got {self.method!r}")
        if self.symmetrize not in SYMMETRIZE:
            raise ValueError(f"symmetrize must be one of {SYMMETRIZE}")
        if self.anatomical not in ANATOMICAL:
            raise ValueError(f"anatomical must be one of {ANATOMICAL}")
        if self.method in ("knn", "proportional") and self.k is None and self.k_fraction is None \
                and self.density is None:
            raise ValueError(f"{self.method}: give k, k_fraction or density")
        if self.anatomical != "none" and not self.roi_names:
            raise ValueError("an anatomical mask needs the store's ROI names")

    @classmethod
    def from_config(cls, cfg: dict | None, roi_names: list | None = None) -> "GraphConstruction":
        cfg = dict(cfg or {"method": "dense"})
        cfg.pop("name", None)
        return cls(**cfg, roi_names=list(roi_names) if roi_names is not None else None)

    # ------------------------------------------------------------------------------------------------- parameters
    def k_for(self, r: int) -> int:
        if self.k is not None:
            k = int(self.k)
        elif self.k_fraction is not None:
            k = max(1, int(round(self.k_fraction * (r - 1))))
        else:
            k = max(1, int(round(self.density * (r - 1))))
        if not 1 <= k <= r - 1:
            raise ValueError(f"k = {k} outside [1, {r - 1}] for {r} ROIs")
        return k

    def mask(self, r: int) -> np.ndarray | None:
        if self.anatomical == "none":
            return None
        from ..source import atlas

        if len(self.roi_names) != r:
            raise ValueError(f"{len(self.roi_names)} ROI names for {r} ROIs")
        if self.anatomical == "contiguity":
            return atlas.contiguity(list(self.roi_names)).astype(bool)
        lobes = atlas.lobe_of(atlas.atlas_of(list(self.roi_names)))
        g = np.array([f"{lobes[n]}-{n.rsplit('-', 1)[1]}" for n in self.roi_names])
        m = g[:, None] == g[None, :]
        np.fill_diagonal(m, False)
        return m

    @property
    def symmetric(self) -> bool:
        return not (self.method == "knn" and self.symmetrize == "none")

    # ------------------------------------------------------------------------------------------------- fit / apply
    def fit(self, view: DataView) -> "GraphConstruction":
        require_train(view, "GraphConstruction.fit")
        if self.method == "quantile":
            a = self._masked(view.payload["adj"])
            iu = np.triu_indices(a.shape[-1], 1)
            self.threshold_ = np.quantile(a[..., iu[0], iu[1]], self.q, axis=(0, 2))       # (K,)
        return self

    def _masked(self, a: np.ndarray) -> np.ndarray:
        m = self.mask(a.shape[-1])
        return a if m is None else a * m

    def transform(self, adj: np.ndarray) -> np.ndarray:
        """Constructed adjacency (N, K, R, R) float32; records structural statistics in ``stats_``."""
        if adj.ndim != 4 or adj.shape[-1] != adj.shape[-2]:
            raise ValueError(f"adjacency must be (N, K, R, R), got {adj.shape}")
        if self.method == "dense" and self.anatomical == "none":
            self.stats_ = structure_stats(adj[: min(len(adj), BLOCK)])
            return adj
        if self.method == "quantile" and self.threshold_ is None:
            raise RuntimeError("quantile graph construction not fitted")
        out = np.empty(adj.shape, np.float32)
        for i in range(0, len(adj), BLOCK):
            out[i:i + BLOCK] = self._block(np.asarray(adj[i:i + BLOCK], np.float32))
        self.stats_ = structure_stats(out[: min(len(out), BLOCK)])
        return out

    def _block(self, a: np.ndarray) -> np.ndarray:
        r = a.shape[-1]
        a = self._masked(a)
        if self.method == "dense":
            return a.astype(np.float32)
        if self.method == "quantile":
            return np.where(a >= self.threshold_[None, :, None, None], a, 0).astype(np.float32)
        if self.method == "proportional":
            dens = self.density if self.density is not None else self.k_for(r) / (r - 1)
            keep = proportional_mask(a, dens)
        else:
            keep = knn_mask(a, self.k_for(r), self.symmetrize, self.min_degree)
        return np.where(keep, a, 0).astype(np.float32)


# ------------------------------------------------------------------------------------------------------ primitives
def _offdiag_scores(a: np.ndarray) -> np.ndarray:
    s = a.astype(np.float64, copy=True)
    r = a.shape[-1]
    s[..., np.arange(r), np.arange(r)] = -np.inf
    return s


def knn_mask(a: np.ndarray, k: int, symmetrize: str = "union", min_degree: int = 1) -> np.ndarray:
    """Boolean kNN mask of (..., R, R) weights (strongest k off-diagonal edges per row; zero-weight edges are never
    kept). Ties are broken deterministically by ROI index (stable sort)."""
    s = _offdiag_scores(a)
    order = np.argsort(-s, axis=-1, kind="stable")[..., :k]
    directed = np.zeros(a.shape, bool)
    np.put_along_axis(directed, order, True, axis=-1)
    directed &= a > 0
    if symmetrize == "none":
        return directed
    t = np.swapaxes(directed, -1, -2)
    if symmetrize == "union":
        return directed | t
    keep = directed & t
    if min_degree > 0:                          # mutual kNN: give isolated / under-connected nodes their best edges
        need = keep.sum(-1) < min_degree
        if need.any():
            top = np.argsort(-s, axis=-1, kind="stable")[..., :min_degree]
            add = np.zeros(a.shape, bool)
            np.put_along_axis(add, top, True, axis=-1)
            add &= need[..., None] & (a > 0)
            keep = keep | add | np.swapaxes(add, -1, -2)
    return keep


def proportional_mask(a: np.ndarray, density: float) -> np.ndarray:
    """The round(density * R (R - 1) / 2) strongest edges of each (R, R) graph (symmetric, positive weights only)."""
    r = a.shape[-1]
    iu = np.triu_indices(r, 1)
    m = int(round(density * len(iu[0])))
    if not 1 <= m <= len(iu[0]):
        raise ValueError(f"density {density} keeps {m} of {len(iu[0])} edges")
    w = a[..., iu[0], iu[1]]
    top = np.argsort(-w, axis=-1, kind="stable")[..., :m]
    sel = np.zeros(w.shape, bool)
    np.put_along_axis(sel, top, True, axis=-1)
    sel &= w > 0
    keep = np.zeros(a.shape, bool)
    keep[..., iu[0], iu[1]] = sel
    return keep | np.swapaxes(keep, -1, -2)


def structure_stats(adj: np.ndarray, eps: float = 0.0) -> dict:
    """Structural summary of (N, K, R, R) graphs: density, degree, isolated nodes, symmetry, negativity."""
    a = np.asarray(adj)
    r = a.shape[-1]
    e = a > eps
    e[..., np.arange(r), np.arange(r)] = False
    deg = e.sum(-1)
    return {"density": float(e.sum((-1, -2)).mean() / (r * (r - 1))), "mean_degree": float(deg.mean()),
            "min_degree": int(deg.min()) if deg.size else 0,
            "isolated_nodes_per_graph": float((deg == 0).sum(-1).mean()),
            "graphs_with_isolated_nodes": float(((deg == 0).sum(-1) > 0).mean()),
            "symmetric": bool(np.allclose(a, np.swapaxes(a, -1, -2))), "min_weight": float(a.min()),
            "zero_diagonal": bool(np.abs(np.diagonal(a, axis1=-2, axis2=-1)).max() == 0) if a.size else True}


def n_components(adj2d: np.ndarray) -> int:
    """Connected components of one (R, R) graph (edges = positive weights, either direction)."""
    from scipy.sparse.csgraph import connected_components

    return int(connected_components((adj2d > 0) | (adj2d.T > 0), directed=False)[0])


# ------------------------------------------------------------------------------------------- feature similarity
FS_KINDS = ("log", "rel", "de")


def feature_similarity_adjacency(features: np.ndarray, k_bands: int) -> np.ndarray:
    """Node-feature similarity graphs, an explicitly DIFFERENT graph definition from functional connectivity.

    features (N, R, D): per window, each ROI's spectral feature vector (e.g. log / relative PSD and DE of all five
    bands). Per window (stateless): the D features are z-scored across ROIs; S_ij = exp(-||f_i - f_j||^2 / (2 s^2))
    with s the median off-diagonal distance of that window; zero diagonal. The same similarity graph is used for
    every band graph (returned as (N, K, R, R)), because it is not band-specific connectivity."""
    n, r, _ = features.shape
    iu = np.triu_indices(r, 1)
    out = np.empty((n, k_bands, r, r), np.float32)
    for i in range(0, n, BLOCK):
        f = np.asarray(features[i:i + BLOCK], np.float64)
        mu, sd = f.mean(1, keepdims=True), f.std(1, keepdims=True)
        z = (f - mu) / np.where(sd < 1e-12, 1.0, sd)
        sq = (z ** 2).sum(-1)
        d2 = np.maximum(sq[:, :, None] + sq[:, None, :] - 2 * z @ np.swapaxes(z, 1, 2), 0.0)
        s2 = np.median(d2[:, iu[0], iu[1]], axis=1)[:, None, None]
        sim = np.exp(-d2 / (2 * np.where(s2 < 1e-12, 1.0, s2)))
        sim = (sim + np.swapaxes(sim, 1, 2)) / 2
        sim[:, np.arange(r), np.arange(r)] = 0.0
        out[i:i + BLOCK] = sim[:, None].astype(np.float32)
    return out
