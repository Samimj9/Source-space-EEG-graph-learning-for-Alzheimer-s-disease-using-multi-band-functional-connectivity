"""Connectivity figures (family 6): group matrices, AD − HC differences, cortical network view.

Unit: subject (window matrices averaged within subject, then over subjects). PLV and
|ciPLV| are drawn in separate figures (``spec['connectivity']``). The signed ciPLV is
never used here. Network views show only a **display-only** edge selection (top-N
by |AD − HC| or by group mean); it is marked as such in the figure, its data file and
its metadata, and is not a statistical threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from .anatomy import roi_table, subject_level
from .results import MissingInputs
from .style import OKABE_ITO, figure_size

LABEL = {"plv": "PLV", "ciplv": "|ciPLV|"}


def _group_mats(store, conn, dataset):
    if conn not in LABEL:
        raise ValueError("connectivity must be 'plv' or 'ciplv' (|ciPLV|)")
    subj, mats = subject_level(store, conn, dataset)                       # (S, B, R, R)
    y = subj["y"].to_numpy()
    return subj, {"AD": mats[y == 1], "HC": mats[y == 0]}


def fig_connectivity_matrices(store, spec, style, stats=None):
    conn, dataset = spec["connectivity"], spec.get("dataset")
    bands = spec.get("bands", store.bands)
    subj, groups = _group_mats(store, conn, dataset)
    order = roi_table(store.roi_names)["index"].to_numpy()
    fig, axes = plt.subplots(len(bands), 3, squeeze=False,
                             figsize=figure_size(style, spec.get("width", "double"),
                                                 height_mm=spec.get("height_mm", 55 * len(bands))))
    rows = []
    for i, band in enumerate(bands):
        b = store.band_index([band])[0]
        m = {g: a[:, b].mean(axis=0)[np.ix_(order, order)] for g, a in groups.items()}
        d = m["AD"] - m["HC"]
        off = ~np.eye(len(order), dtype=bool)
        vmax = max(m["AD"][off].max(), m["HC"][off].max())
        for j, (g, mat) in enumerate((("AD", m["AD"]), ("HC", m["HC"]))):
            im = axes[i, j].imshow(np.where(off, mat, np.nan), cmap=style["sequential_cmap"], vmin=0, vmax=vmax)
            axes[i, j].set_title(f"{band}: {g} mean (n={len(groups[g])})", fontsize=6)
        fig.colorbar(im, ax=axes[i, :2], shrink=0.7, label=LABEL[conn])
        lim = np.nanmax(np.abs(d[off])) or 1.0
        im2 = axes[i, 2].imshow(np.where(off, d, np.nan), cmap=style["diverging_cmap"], vmin=-lim, vmax=lim)
        axes[i, 2].set_title(f"{band}: AD − HC (descriptive)", fontsize=6)
        fig.colorbar(im2, ax=axes[i, 2], shrink=0.7)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
        iu = np.triu_indices(len(order), 1)
        rows += [{"band": band, "roi_i": store.roi_names[order[a]], "roi_j": store.roi_names[order[c]],
                  "AD_mean": m["AD"][a, c], "HC_mean": m["HC"][a, c], "AD_minus_HC": d[a, c]}
                 for a, c in zip(*iu)]
    fig.suptitle(f"{LABEL[conn]} group matrices (ROIs ordered lh→rh, by lobe)" + (f", {dataset}" if dataset else ""))
    meta = {"kind": "descriptive", "statistical_unit": "subject (windows averaged within subject)",
            "notes": "no thresholding; difference maps descriptive; no significance shown",
            "n_AD": len(groups["AD"]), "n_HC": len(groups["HC"])}
    return fig, pd.DataFrame(rows), meta, [str(store.root)]


def fig_connectivity_network(store, spec, style, stats=None):
    if store.roi_coords is None:
        raise MissingInputs("graph run has no roi_coords (ROI centroids); cannot draw the network view")
    conn, dataset, top_n = spec["connectivity"], spec.get("dataset"), int(spec.get("display_top_n", 40))
    quantity = spec.get("quantity", "AD_minus_HC")                          # AD_minus_HC | AD_mean | HC_mean
    bands = spec.get("bands", store.bands)
    subj, groups = _group_mats(store, conn, dataset)
    xy = np.asarray(store.roi_coords)[:, :2]                                # axial projection (x: L-R, y: P-A)
    fig, axes = plt.subplots(1, len(bands), squeeze=False,
                             figsize=figure_size(style, spec.get("width", "double"), height_mm=spec.get("height_mm", 60)))
    rows = []
    iu = np.triu_indices(len(store.roi_names), 1)
    for k, band in enumerate(bands):
        b = store.band_index([band])[0]
        mats = {g: a[:, b].mean(axis=0) for g, a in groups.items()}
        q = {"AD_minus_HC": mats["AD"] - mats["HC"], "AD_mean": mats["AD"], "HC_mean": mats["HC"]}[quantity]
        vals = q[iu]
        keep = np.argsort(-np.abs(vals))[:top_n]                             # DISPLAY-ONLY selection
        ax = axes[0, k]
        segs = [[xy[iu[0][e]], xy[iu[1][e]]] for e in keep]
        lim = np.abs(vals[keep]).max() or 1.0
        cmap = plt.get_cmap(style["diverging_cmap"] if quantity == "AD_minus_HC" else style["sequential_cmap"])
        colors = [cmap(0.5 + 0.5 * vals[e] / lim) if quantity == "AD_minus_HC" else cmap(vals[e] / lim) for e in keep]
        widths = 0.3 + 1.5 * np.abs(vals[keep]) / lim
        styles = ["-" if vals[e] >= 0 else "--" for e in keep]               # sign also by line style (grayscale)
        ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, linestyles=styles, zorder=1))
        ax.scatter(xy[:, 0], xy[:, 1], s=4, color=OKABE_ITO["black"], zorder=2)
        ax.set_aspect("equal")
        ax.set_title(band)
        ax.axis("off")
        rows += [{"band": band, "roi_i": store.roi_names[iu[0][e]], "roi_j": store.roi_names[iu[1][e]],
                  quantity: vals[e], "display_rank": r + 1} for r, e in enumerate(keep)]
    fig.suptitle(f"{LABEL[conn]} {quantity.replace('_', ' ')}: top {top_n} edges by magnitude — "
                 "DISPLAY-ONLY selection, not a statistical threshold", fontsize=6)
    meta = {"kind": "descriptive", "statistical_unit": "subject (windows averaged within subject)",
            "display_only_threshold": {"rule": f"top {top_n} |{quantity}| per band", "statistical": False},
            "notes": "axial projection of ROI centroids; solid = positive, dashed = negative"}
    return fig, pd.DataFrame(rows), meta, [str(store.root)]
