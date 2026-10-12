"""Exploratory dataset-shift views (family 13): PCA (or optional UMAP) of
subject-level source PSD and of saved graph embeddings.

These figures are **descriptive and exploratory**. Projections are unsupervised (labels
are never used to fit them); apparent clusters are not statistical evidence of group or
dataset differences. Colour = diagnosis, marker shape = dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from .anatomy import subject_level
from .results import anonymize, select_unique
from .style import DATASET_MARKER, GROUP_STYLE, figure_size

DISCLAIMER = "Exploratory, unsupervised projection; not statistical evidence"


def project(X: np.ndarray, method: str, seed: int = 0):
    Xz = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    if method == "pca":
        p = PCA(n_components=2, random_state=seed).fit(Xz)
        return p.transform(Xz), [f"PC{i + 1} ({v:.0%} var.)" for i, v in enumerate(p.explained_variance_ratio_)]
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("method 'umap' requires the optional package umap-learn") from exc
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(Xz), ["UMAP 1", "UMAP 2"]
    raise ValueError("method must be 'pca' or 'umap'")


def _scatter(ax, df, style, fill_col=None):
    for (ds, grp), d in df.groupby(["dataset", "group"]):
        g = GROUP_STYLE[grp]
        filled = d[fill_col] if fill_col else pd.Series(True, index=d.index)
        for f, dd in d.groupby(filled):
            ax.scatter(dd["c1"], dd["c2"], marker=DATASET_MARKER.get(ds, "o"), s=style["marker_size"] ** 2 * 1.5,
                       facecolors=g["color"] if f else "none", edgecolors=g["color"], linewidths=0.7,
                       label=f"{grp} / {ds}" + ("" if not fill_col else (" (train)" if f else " (test)")))
    ax.legend(fontsize=5, loc="best")


def fig_psd_embedding(store, spec, style, stats=None):
    kinds, method = spec.get("kinds", ["log", "rel"]), spec.get("method", "pca")
    feats, subj = [], None
    for k in kinds:
        subj, a = subject_level(store, f"psd_{k}", spec.get("dataset"))
        feats.append(a.reshape(len(a), -1))
    emb, axes_lab = project(np.concatenate(feats, axis=1), method)
    anon = anonymize(subj["key"])
    df = pd.DataFrame({"subject": subj["key"].map(anon), "dataset": subj["dataset"],
                       "group": np.where(subj["y"] == 1, "AD", "HC"), "c1": emb[:, 0], "c2": emb[:, 1]})
    fig, ax = plt.subplots(figsize=figure_size(style, spec.get("width", "single"), height_ratio=0.9))
    _scatter(ax, df, style)
    ax.set_xlabel(axes_lab[0])
    ax.set_ylabel(axes_lab[1])
    ax.set_title(f"Source PSD ({'+'.join(kinds)}), {method.upper()}\n{DISCLAIMER}", fontsize=6)
    fig.tight_layout()
    meta = {"kind": "descriptive (exploratory)", "statistical_unit": "subject (PSD averaged over windows)",
            "notes": "features z-scored over subjects; projection fitted without labels", "axes": axes_lab}
    return fig, df, meta, [str(store.root)]


def fig_graph_embedding(index, spec, style, stats=None):
    rows = select_unique(index, spec["entry"], [spec["setting"]])
    if len(rows) != 1:
        raise ValueError("graph-embedding figure needs exactly one result (one condition)")
    r = rows.iloc[0]
    path = Path(r["path"]) / "embeddings.npz"
    if not path.exists():
        raise ValueError(f"{path} missing: rerun with evaluation.save_embeddings: true")
    if not isinstance(spec.get("repeat"), int) or not isinstance(spec.get("fold"), int):
        raise ValueError("graph-embedding figures need one explicit fitted model: integer 'repeat' and 'fold'; "
                         "embeddings of independently trained folds are never merged")
    z = np.load(path, allow_pickle=True)
    sel = (z["repeat"] == spec["repeat"]) & (z["fold"] == spec["fold"])
    if not sel.any():
        raise ValueError("requested repeat/fold not in embeddings.npz")
    e = pd.DataFrame(z["emb"][sel])
    dims = list(e.columns)
    e = e.assign(key=z["key"][sel], side=z["side"][sel], y=z["y"][sel], dataset=z["dataset"][sel])
    per = e.groupby(["key", "side"]).agg({**{d: "mean" for d in dims}, "y": "first", "dataset": "first"}).reset_index()
    emb, axes_lab = project(per[dims].to_numpy(), spec.get("method", "pca"))   # one embedding per subject
    anon = anonymize(per["key"])
    df = pd.DataFrame({"subject": per["key"].map(anon), "side": per["side"], "dataset": per["dataset"],
                       "group": np.where(per["y"] == 1, "AD", "HC"), "c1": emb[:, 0], "c2": emb[:, 1]})
    df["is_train"] = df["side"] == "train"
    fig, ax = plt.subplots(figsize=figure_size(style, spec.get("width", "single"), height_ratio=0.9))
    _scatter(ax, df, style, fill_col="is_train")
    ax.set_xlabel(axes_lab[0])
    ax.set_ylabel(axes_lab[1])
    ax.set_title(f"{spec['entry']['label']}: fused graph embeddings\none fitted model (repeat "
                 f"{spec['repeat']}, fold {spec['fold']}); {spec['setting']}\n{DISCLAIMER}", fontsize=5.5)
    fig.tight_layout()
    meta = {"kind": "descriptive (exploratory)", "statistical_unit": "subject (window embeddings averaged)",
            "notes": "embeddings from a single fitted model (spaces of different folds are not comparable); "
                     "filled = training subjects, open = test subjects", "axes": axes_lab,
            "repeat": spec["repeat"], "fold": spec["fold"], "single_fitted_model": True}
    return fig, df, meta, [r["path"]]
