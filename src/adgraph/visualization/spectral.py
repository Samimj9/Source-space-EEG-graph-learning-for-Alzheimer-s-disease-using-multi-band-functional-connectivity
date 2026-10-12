"""Source-space PSD figures (family 5), from a saved graph run.

Unit: subject. Each subject's band power is first averaged over its windows; group
summaries (mean, SD across subjects) are then computed over subjects. The AD − HC
difference is descriptive. Significance is drawn only from a valid, corrected
statistics file (comparison id ``psd:<kind>:<band>:<roi>``), never computed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import stats as st
from .anatomy import LOBE_ORDER, roi_table, subject_level
from .style import GROUP_STYLE, figure_size, marker_kwargs

KIND_LABEL = {"log": "log10 absolute power", "rel": "relative power (of 1–45 Hz)"}


def _psd_subjects(store, kind, dataset):
    if kind not in ("log", "rel"):
        raise ValueError("PSD figures support kind 'log' (log absolute) or 'rel' (relative)")
    return subject_level(store, f"psd_{kind}", dataset)


def fig_psd_heatmap(store, spec, style, stats=None):
    kind, dataset = spec.get("kind", "rel"), spec.get("dataset")
    subj, psd = _psd_subjects(store, kind, dataset)                        # (S, B, R)
    rt = roi_table(store.roi_names)
    order = rt["index"].to_numpy()
    groups = {"AD": psd[subj["y"].to_numpy() == 1], "HC": psd[subj["y"].to_numpy() == 0]}
    means = {g: a.mean(axis=0)[:, order].T for g, a in groups.items()}       # (R, B)
    diff = means["AD"] - means["HC"]
    fig, axes = plt.subplots(1, 3, figsize=figure_size(style, spec.get("width", "double"),
                                                       height_mm=spec.get("height_mm", 150)), sharey=True)
    vmin, vmax = min(m.min() for m in means.values()), max(m.max() for m in means.values())
    for ax, g in zip(axes[:2], ("AD", "HC")):
        im = ax.imshow(means[g], aspect="auto", cmap=style["sequential_cmap"], vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"{g} mean (n={len(groups[g])} subjects)")
    fig.colorbar(im, ax=axes[:2], shrink=0.5, label=KIND_LABEL[kind])
    lim = np.abs(diff).max() or 1.0
    im2 = axes[2].imshow(diff, aspect="auto", cmap=style["diverging_cmap"], vmin=-lim, vmax=lim,
                         interpolation="nearest")
    axes[2].set_title("AD − HC (descriptive)")
    fig.colorbar(im2, ax=axes[2], shrink=0.5, label="difference")
    marked = []
    for bi, band in enumerate(store.bands):
        for ri, roi_idx in enumerate(order):
            row = st.lookup(stats, f"psd:{kind}:{band}:{store.roi_names[roi_idx]}")
            if row is not None and row["p_corrected"] < spec.get("alpha", 0.05):
                axes[2].plot(bi, ri, marker="*", color="k", markersize=3)
                marked.append(row["comparison"])
    for ax in axes:
        ax.set_xticks(range(len(store.bands)), store.bands, rotation=45)
    axes[0].set_yticks(range(len(order)), [f"{r.roi}" for r in rt.itertuples()], fontsize=3.5)
    title = f"Source PSD ({KIND_LABEL[kind]})" + (f", {dataset}" if dataset else ", D1+D2")
    fig.suptitle(title)
    data = pd.DataFrame([{"roi": store.roi_names[idx], "lobe": rt.loc[ri, "lobe"], "band": b,
                          "AD_mean": means["AD"][ri, bi], "HC_mean": means["HC"][ri, bi],
                          "AD_sd": groups["AD"][:, bi, idx].std(ddof=1), "HC_sd": groups["HC"][:, bi, idx].std(ddof=1),
                          "AD_minus_HC": diff[ri, bi], "n_AD": len(groups["AD"]), "n_HC": len(groups["HC"])}
                         for ri, idx in enumerate(order) for bi, b in enumerate(store.bands)])
    meta = {"kind": "inferential" if marked else "descriptive",
            "statistical_unit": "subject (windows averaged within subject first)",
            "notes": "group means over subjects; difference map is descriptive" +
                     ("; * = corrected p from statistics file" if marked else "; no significance shown"),
            "significance_from_stats_file": marked}
    return fig, data, meta, [str(store.root)]


def fig_psd_lobes(store, spec, style, stats=None):
    kind, dataset = spec.get("kind", "rel"), spec.get("dataset")
    subj, psd = _psd_subjects(store, kind, dataset)
    rt = roi_table(store.roi_names)
    lobes = [lb for lb in LOBE_ORDER if lb in set(rt["lobe"])]
    fig, axes = plt.subplots(1, len(store.bands), squeeze=False, sharey=True,
                             figsize=figure_size(style, spec.get("width", "double"), height_mm=spec.get("height_mm", 60)))
    rng = np.random.default_rng(0)
    rows = []
    for bi, band in enumerate(store.bands):
        ax = axes[0, bi]
        for li, lobe in enumerate(lobes):
            idx = rt.loc[rt["lobe"] == lobe, "index"].to_numpy()
            per_subject = psd[:, bi, idx].mean(axis=1)                       # one value per subject
            for gi, (g, yv) in enumerate((("HC", 0), ("AD", 1))):
                v = per_subject[subj["y"].to_numpy() == yv]
                x = li + (gi - 0.5) * 0.35
                spec_g = GROUP_STYLE[g]
                ax.plot(x + rng.uniform(-0.06, 0.06, len(v)), v, linestyle="none", alpha=style["point_alpha"],
                        **marker_kwargs(spec_g, style["point_size"]))
                ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else None, capsize=1.5,
                            **marker_kwargs(spec_g, style["marker_size"] * 1.3))
                rows.append({"band": band, "lobe": lobe, "group": g, "mean": v.mean(),
                             "sd": v.std(ddof=1) if len(v) > 1 else np.nan, "n_subjects": len(v)})
        ax.set_xticks(range(len(lobes)), lobes, rotation=45, ha="right")
        ax.set_title(band)
    axes[0, 0].set_ylabel(KIND_LABEL[kind])
    handles = [plt.Line2D([], [], linestyle="none", label=g, **marker_kwargs(GROUP_STYLE[g])) for g in ("HC", "AD")]
    fig.legend(handles=handles, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    meta = {"kind": "descriptive", "statistical_unit": "subject (mean over windows, then over lobe ROIs)",
            "notes": "points = subjects; marker = group mean; bars = SD across subjects; no significance"}
    return fig, pd.DataFrame(rows), meta, [str(store.root)]
