"""Structural diagnostics of the graph constructions (docs/final/graph_construction.md §4), label-blind.

For every construction of ``configs/final/graphs.yaml``, connectivity measure (PLV, |ciPLV|) and band, on a fixed
sample of windows of the DEVELOPMENT participants only (the holdout is never read; the diagnosis is never read):

* structure: edge density, mean / SD / min / max node degree, isolated nodes, graphs with an isolated node,
  connected components (mean, max) and the fraction of connected graphs, the fraction of the dense graph's total
  connectivity weight that the construction retains, symmetry, zero diagonal, minimum weight;
* stability: mean Jaccard overlap of the retained edge sets between windows of the SAME participant and between
  windows of DIFFERENT participants (a construction whose edges change from window to window as much as from
  participant to participant carries little participant-specific structure);
* cost: retained edges per graph relative to the dense graph (dense GCN cost is unchanged, but sparse message
  passing, memory of sparse formats and EARM-GNN gate penalties scale with it).

Nothing here selects a construction; selection uses development-CV results only (``experiments.selection``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .construction import GraphConstruction, feature_similarity_adjacency, n_components, structure_stats
from .store import GraphStore


def sample_windows(store: GraphStore, keys: list[str], per_participant: int = 10, seed: int = 0) -> np.ndarray:
    """Indices of up to ``per_participant`` windows of each participant in ``keys`` (fixed seed, sorted)."""
    rng = np.random.default_rng(seed)
    idx = []
    for k, rows in store.windows.groupby("key").indices.items():
        if k in set(keys):
            take = rows if len(rows) <= per_participant else rng.choice(rows, per_participant, replace=False)
            idx += sorted(int(i) for i in take)
    return np.array(sorted(idx), int)


def _jaccard(e: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    out = []
    for i, j in pairs:
        inter = np.logical_and(e[i], e[j]).sum()
        union = np.logical_or(e[i], e[j]).sum()
        out.append(inter / union if union else 1.0)
    return float(np.mean(out))


def stability(edges: np.ndarray, keys: np.ndarray, n_pairs: int = 400, seed: int = 0) -> dict:
    """Mean Jaccard of edge sets (G, R, R bool) within and between participants (random pairs, fixed seed)."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(edges.shape[-1], 1)
    e = edges[:, iu[0], iu[1]]
    groups = pd.Series(range(len(keys))).groupby(keys).apply(list).to_dict()
    within, between = [], []
    multi = [g for g in groups.values() if len(g) > 1]
    for _ in range(n_pairs):
        if multi:
            g = multi[rng.integers(len(multi))]
            i, j = rng.choice(g, 2, replace=False)
            within.append((int(i), int(j)))
        if len(groups) > 1:
            a, b = rng.choice(list(groups), 2, replace=False)
            between.append((int(rng.choice(groups[a])), int(rng.choice(groups[b]))))
    jw, jb = _jaccard(e, within), _jaccard(e, between)
    return {"jaccard_within_participant": jw, "jaccard_between_participants": jb,
            "jaccard_ratio_within_between": jw / jb if jb else float("nan")}


def construction_diagnostics(store: GraphStore, constructions: dict[str, dict], development_keys: list[str],
                             connectivity=("plv", "ciplv"), per_participant: int = 10, seed: int = 0) -> pd.DataFrame:
    """One row per (construction, connectivity, band): the diagnostics of the module docstring."""
    idx = sample_windows(store, development_keys, per_participant, seed)
    if not len(idx):
        raise ValueError("no development-participant windows in the store")
    keys = store.windows["key"].to_numpy()[idx]
    rows = []
    for conn in connectivity:
        dense = np.asarray(store.arrays[conn][idx], np.float32)                     # (n, K, R, R)
        r = dense.shape[-1]
        fs = None
        for name, cfg in constructions.items():
            cfg = dict(cfg)
            similarity = cfg.pop("adjacency", "connectivity") == "feature_similarity"
            if similarity:
                if fs is None:
                    f = np.concatenate([np.moveaxis(store.arrays[f"psd_{k}"][idx], 1, -1) for k in
                                        ("log", "rel", "de")], axis=-1)
                    fs = feature_similarity_adjacency(f, dense.shape[1])
                base = fs
            else:
                base = dense
            gc = GraphConstruction.from_config(cfg, store.roi_names)
            if gc.method == "quantile":
                raise ValueError("quantile constructions are fitted on training data; not part of the diagnostics")
            adj = gc.transform(base)
            for b, band in enumerate(store.bands):
                a = adj[:, b]
                st = structure_stats(a[:, None])
                e = a > 0
                e[:, np.arange(r), np.arange(r)] = False
                deg = e.sum(-1)
                comp = np.array([n_components(g) for g in a])
                total = dense[:, b].sum((-1, -2))
                rows.append({"construction": name, "adjacency": "feature_similarity" if similarity else conn,
                             "connectivity": conn, "band": band, "n_windows": int(len(idx)),
                             "n_participants": int(len(set(keys))), "k": gc.k_for(r) if gc.method in ("knn",) else None,
                             "density": st["density"], "mean_degree": float(deg.mean()),
                             "sd_degree": float(deg.std(axis=-1).mean()), "min_degree": int(deg.min()),
                             "max_degree": int(deg.max()),
                             "isolated_nodes_per_graph": st["isolated_nodes_per_graph"],
                             "graphs_with_isolated_nodes": st["graphs_with_isolated_nodes"],
                             "components_mean": float(comp.mean()), "components_max": int(comp.max()),
                             "fraction_connected": float((comp == 1).mean()),
                             "retained_weight_fraction": (float(np.mean(np.where(total > 0, a.sum((-1, -2)) / total,
                                                                                 np.nan))) if not similarity
                                                          else float("nan")),
                             "edges_per_graph": float(e.sum((-1, -2)).mean() / 2),
                             "relative_edges": float(e.sum((-1, -2)).mean() / (r * (r - 1))),
                             "symmetric": st["symmetric"], "zero_diagonal": st["zero_diagonal"],
                             "min_weight": st["min_weight"], **stability(e, keys, seed=seed)})
    return pd.DataFrame(rows)


def summarize(diag: pd.DataFrame) -> pd.DataFrame:
    """Construction x adjacency means over bands (the table of the graph-construction notebook)."""
    cols = ["density", "mean_degree", "sd_degree", "min_degree", "isolated_nodes_per_graph", "fraction_connected",
            "components_max", "retained_weight_fraction", "relative_edges", "jaccard_within_participant",
            "jaccard_between_participants", "jaccard_ratio_within_between"]
    agg = {c: ("min" if c == "min_degree" else "max" if c == "components_max" else "mean") for c in cols}
    return diag.groupby(["construction", "adjacency"], sort=False).agg(agg).reset_index()
