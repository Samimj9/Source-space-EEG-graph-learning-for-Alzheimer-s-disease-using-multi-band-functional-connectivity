"""Experiment inputs built from a ``GraphStore``.

Two kinds of transformation are kept strictly apart:

* **Stateless, per-window transforms** live here: selecting PLV or ciPLV,
  selecting bands, band averaging, node strength, PSD node features. Each
  window's output depends only on that window, so they can be computed once for
  all windows without leaking information between subjects.
* **Fitted transforms** (standardization, feature selection, PCA, quantile
  sparsification, hyperparameters) live inside the estimators and only ever see
  a ``DataView`` whose role is ``"train"`` (see ``require_train``).

Every sample carries its metadata row (``window_id``, ``key``, ``dataset``,
``subject_id``, ``y``, ...). Views are created only through ``Dataset.view``,
which keeps arrays and metadata aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .store import CONNECTIVITY, PSD_KINDS, GraphStore

ROLES = ("train", "val", "test")


class LeakageError(RuntimeError):
    """Raised when a fitted transform would see non-training data."""


@dataclass
class DataView:
    """Aligned subset of a dataset: ``meta`` rows <-> first axis of every payload array."""

    meta: pd.DataFrame
    payload: dict[str, np.ndarray]
    role: str
    target: str = "y"

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        for k, v in self.payload.items():
            if len(v) != len(self.meta):
                raise ValueError(f"payload {k} has {len(v)} rows, meta has {len(self.meta)}")

    def __len__(self) -> int:
        return len(self.meta)

    @property
    def y(self) -> np.ndarray:
        return self.meta[self.target].to_numpy().astype(int)

    @property
    def subjects(self) -> np.ndarray:
        return self.meta["key"].to_numpy()

    def take(self, idx: np.ndarray, role: str) -> "DataView":
        idx = np.asarray(idx)
        return DataView(self.meta.iloc[idx].reset_index(drop=True), {k: v[idx] for k, v in self.payload.items()},
                        role, self.target)


def require_train(view: DataView, who: str = "fit") -> None:
    if view.role != "train":
        raise LeakageError(f"{who} called on a '{view.role}' view; fitted transforms may only see training data")


def training_weights(meta: pd.DataFrame, target: str = "y") -> np.ndarray:
    """Per-window loss weights: each subject sums to its class weight, classes balanced by subject count.

    w = 1 / n_windows(subject) * n_subjects / (2 * n_subjects(class)), normalized to mean 1.
    Computed from the training view only.
    """
    n_win = meta.groupby("key")["key"].transform("size").to_numpy()
    subj = meta.drop_duplicates("key")
    counts = subj[target].value_counts()
    cls_w = meta[target].map(lambda c: len(subj) / (len(counts) * counts[c])).to_numpy()
    w = cls_w / n_win
    return w / w.mean()


@dataclass
class Dataset:
    """All windows of an experiment with stateless features already built."""

    meta: pd.DataFrame
    payload: dict[str, np.ndarray]
    kind: str                       # "classical" | "graph"
    info: dict = field(default_factory=dict)
    target: str = "y"

    def view(self, idx: np.ndarray, role: str) -> DataView:
        idx = np.asarray(idx)
        return DataView(self.meta.iloc[idx].reset_index(drop=True), {k: v[idx] for k, v in self.payload.items()},
                        role, self.target)


# ---------------------------------------------------------------- metadata
def _meta(store: GraphStore, target: str) -> pd.DataFrame:
    meta = store.windows.copy()
    if target == "dataset":           # dataset-identity probe (DESIGN.md §6)
        meta["y_dataset"] = (meta["dataset"] == "D2").astype(int)
    if not store.subjects.empty and "age" in store.subjects:
        meta = meta.merge(store.subjects[["key", "age"]], on="key", how="left", validate="many_to_one")
    return meta


def target_column(target: str) -> str:
    return {"label": "y", "y": "y", "dataset": "y_dataset"}[target]


# ---------------------------------------------------------------- classical
def build_classical(store: GraphStore, features: dict, target: str = "label") -> Dataset:
    """Flat feature matrix for LR / SVM.

    features.source: ``sensor`` (electrode PSD, Experiment 0), ``source`` (ROI PSD,
    Experiment 1), ``plv`` / ``ciplv`` (vectorized upper-triangle connectivity,
    DESIGN.md §5 comparator) or ``age`` (age-only baseline). ``psd_kinds`` from
    {abs, log, rel}; ``bands`` subset. Feature order: kind -> band -> channel/ROI (or edge).
    """
    meta = _meta(store, target)
    src = features["source"]
    if src in CONNECTIVITY:
        bands = features.get("bands") or store.bands
        A = store.arrays[src][:, store.band_index(bands)]
        iu = np.triu_indices(A.shape[-1], 1)
        X = A[..., iu[0], iu[1]].reshape(len(A), -1).astype(np.float64)
        names = [f"{src}:{b}:{store.roi_names[i]}-{store.roi_names[j]}" for b in bands for i, j in zip(*iu)]
    elif src == "age":
        if "age" not in meta or meta["age"].isna().any():
            raise ValueError("age-only baseline needs an age for every subject")
        X = meta[["age"]].to_numpy(float)
        names = ["age"]
    else:
        prefix = {"sensor": "sensor_psd", "source": "psd"}[src]
        nodes = store.channels if src == "sensor" else store.roi_names
        bands = features.get("bands") or store.bands
        b_idx = store.band_index(bands)
        blocks, names = [], []
        for kind in features["psd_kinds"]:
            if kind not in PSD_KINDS:
                raise ValueError(f"psd kind {kind!r} not in {PSD_KINDS}")
            arr = store.arrays[f"{prefix}_{kind}"][:, b_idx, :]           # (N, n_bands, n_nodes)
            blocks.append(arr.reshape(len(arr), -1))
            names += [f"{kind}:{b}:{n}" for b in bands for n in nodes]
        X = np.concatenate(blocks, axis=1).astype(np.float64)
    if not np.isfinite(X).all():
        raise ValueError("non-finite classical features (e.g. log of zero power)")
    tcol = target_column(target)
    return Dataset(meta, {"X": X}, "classical", {"feature_names": names, "source": src}, tcol)


# ---------------------------------------------------------------- graphs
def node_strength(adj: np.ndarray) -> np.ndarray:
    """Weighted node strength: mean connectivity of each node to all others. adj (..., R, R) -> (..., R)."""
    r = adj.shape[-1]
    off = adj.sum(-1) - np.diagonal(adj, axis1=-2, axis2=-1)
    return off / (r - 1)


def build_graph(store: GraphStore, graph: dict, node_features: dict, target: str = "label") -> Dataset:
    """Adjacency + node-feature tensors for the GNN experiments.

    graph:
      connectivity: plv | ciplv            (never mixed)
      bands: [...]                          subset, in this order
      band_mode: single | mean | multi      single: exactly one band, K = 1
                                            mean:   A_mean = mean over bands, K = 1
                                            multi:  one graph per band, K = n_bands
    node_features:
      components: subset of [constant, psd, strength, identity]
      psd_kinds: [log, rel, abs]
      psd_bands: matched | all              matched: graph k gets PSD of band k
                                            (K = 1 graphs: PSD of the graph's own band(s))
                                            all: every graph gets PSD of all store bands

    Returns payload ``x`` (N, K, R, F) float32 and ``adj`` (N, K, R, R) float32.
    """
    meta = _meta(store, target)
    conn = graph["connectivity"]
    if conn not in CONNECTIVITY:
        raise ValueError(f"connectivity must be one of {CONNECTIVITY}")
    bands = list(graph["bands"])
    mode = graph["band_mode"]
    b_idx = store.band_index(bands)
    A = store.arrays[conn][:, b_idx]                                   # (N, B, R, R)
    if mode == "single":
        if len(bands) != 1:
            raise ValueError("band_mode 'single' needs exactly one band")
    elif mode == "mean":
        A = A.mean(axis=1, keepdims=True)
    elif mode != "multi":
        raise ValueError(f"unknown band_mode {mode!r}")
    n, k, r, _ = A.shape
    graph_bands = bands if mode == "multi" else ["+".join(bands)]

    comps = list(node_features["components"])
    unknown = set(comps) - {"constant", "psd", "strength", "identity"}
    if unknown or not comps:
        raise ValueError(f"node feature components must be a non-empty subset of constant/psd/strength/identity, "
                         f"got {comps}")
    blocks, names = [], []
    if "constant" in comps:
        blocks.append(np.ones((n, k, r, 1)))
        names.append("constant")
    if "psd" in comps:
        kinds = node_features.get("psd_kinds", ["log", "rel"])
        matched = node_features.get("psd_bands", "matched") == "matched" and mode == "multi"
        psd_bands = bands if node_features.get("psd_bands", "matched") == "matched" else list(store.bands)
        p_idx = store.band_index(psd_bands)
        for kind in kinds:
            p = store.arrays[f"psd_{kind}"][:, p_idx, :]               # (N, B, R)
            if matched:
                blocks.append(p[..., None])                            # (N, K=B, R, 1)
                names.append(f"psd_{kind}:band")
            else:
                blocks.append(np.broadcast_to(np.moveaxis(p, 1, -1)[:, None], (n, k, r, len(psd_bands))))
                names += [f"psd_{kind}:{b}" for b in psd_bands]
    if "strength" in comps:
        blocks.append(node_strength(A)[..., None])
        names.append("strength")
    if "identity" in comps:
        blocks.append(np.broadcast_to(np.eye(r), (n, k, r, r)))
        names += [f"roi:{x}" for x in store.roi_names]
    X = np.concatenate([np.asarray(b, dtype=np.float32) for b in blocks], axis=-1)
    if not np.isfinite(X).all() or not np.isfinite(A).all():
        raise ValueError("non-finite graph inputs")
    info = {"connectivity": conn, "graph_bands": graph_bands, "band_mode": mode, "feature_names": names,
            "shape_x": X.shape, "shape_adj": A.shape}
    return Dataset(meta, {"x": X, "adj": np.ascontiguousarray(A, dtype=np.float32)}, "graph", info,
                   target_column(target))


def build_dataset(store: GraphStore, cfg: dict) -> Dataset:
    """Dispatch on ``cfg['input']``: ``classical`` or ``graph``."""
    target = cfg.get("target", "label")
    if cfg["input"] == "classical":
        return build_classical(store, cfg["features"], target)
    if cfg["input"] == "graph":
        return build_graph(store, cfg["graph"], cfg["node_features"], target)
    raise ValueError(f"unknown input {cfg['input']!r}")
