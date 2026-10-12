"""Learned multi-band fusion weights (family 7), from saved test-window band weights.

Unit: subject. Window weights (``band_weight_<band>`` in predictions_windows.csv, test
windows only) are averaged per subject and repeat, then over repeats, giving one weight
vector per subject. Fusion weights describe how the trained model combined band
embeddings; they are **not** a measure of biological importance of a band.
An AD vs HC split is drawn only when a valid statistics file supplies every comparison
(``fusion:<label>:<band>``); otherwise it is refused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import stats as st
from .results import read_csv, select_unique
from .style import BAND_STYLE, GROUP_STYLE, condition_style, figure_size, marker_kwargs


def subject_band_weights(row) -> pd.DataFrame:
    w = read_csv(row, "predictions_windows.csv")
    cols = [c for c in w if c.startswith("band_weight_")]
    if not cols:
        raise ValueError(f"{row['path']}: no band weights saved")
    per = w.groupby(["repeat", "key"]).agg({**{c: "mean" for c in cols}, "y": "first", "dataset": "first"})
    per = per.reset_index().groupby("key").agg({**{c: "mean" for c in cols}, "y": "first", "dataset": "first"})
    return per.reset_index().rename(columns={c: c.removeprefix("band_weight_") for c in cols})


def fig_fusion_weights(index, spec, style, stats=None):
    rows = []
    for e in spec["entries"]:
        for _, r in select_unique(index, e, [spec["setting"]]).iterrows():
            fusion = r["config"]["model"].get("fusion")
            if fusion not in ("gate", "attention"):
                raise ValueError(f"{e['label']}: fusion '{fusion}' has no learned band weights")
            rows.append((f"{e['label']} [{condition_style(r['condition'])['label']}]", r))
    split = spec.get("split_by_diagnosis", False)
    fig, axes = plt.subplots(1, len(rows), squeeze=False, sharey=True,
                             figsize=figure_size(style, spec.get("width", "double"), height_mm=spec.get("height_mm", 60)))
    data, rng = [], np.random.default_rng(0)
    for j, (label, r) in enumerate(rows):
        sw = subject_band_weights(r)
        bands = [b for b in r["config"]["graph"]["bands"] if b in sw]
        if split:
            missing = [b for b in bands if st.lookup(stats, f"fusion:{label}:{b}") is None]
            if missing:
                raise st.InvalidStatistics(f"AD vs HC fusion-weight split for {label} needs valid corrected "
                                           f"statistics for bands {missing}")
        ax = axes[0, j]
        for bi, b in enumerate(bands):
            groups = (("HC", 0), ("AD", 1)) if split else (("all", None),)
            for gi, (g, yv) in enumerate(groups):
                v = sw[b] if yv is None else sw.loc[sw["y"] == yv, b]
                x = bi + ((gi - 0.5) * 0.3 if split else 0)
                spec_g = GROUP_STYLE.get(g, {"color": BAND_STYLE[b]["color"], "marker": BAND_STYLE[b]["marker"],
                                             "fill": True})
                ax.boxplot(v, positions=[x], widths=0.25 if split else 0.5, showfliers=False,
                           medianprops={"color": "k"}, boxprops={"linewidth": 0.5}, whiskerprops={"linewidth": 0.5},
                           capprops={"linewidth": 0.5})
                ax.plot(x + rng.uniform(-0.08, 0.08, len(v)), v, linestyle="none", alpha=style["point_alpha"],
                        **marker_kwargs(spec_g, style["point_size"]))
                data += [{"model": label, "band": b, "group": g, "subject_index": i, "weight": float(val)}
                         for i, val in enumerate(v)]
                if split:
                    row = st.lookup(stats, f"fusion:{label}:{b}")
                    ax.text(bi, ax.get_ylim()[1] * 0.98, st.stars(row["p_corrected"]) or "n.s.", ha="center",
                            fontsize=6)
        ax.axhline(1 / len(bands), color="0.5", lw=0.6, ls=":")              # uniform weighting reference
        ax.set_xticks(range(len(bands)), bands)
        ax.set_title(label, fontsize=6)
        if j == 0:
            ax.set_ylabel("Learned fusion weight (subject mean)")
    fig.tight_layout()
    meta = {"kind": "inferential" if split else "descriptive",
            "statistical_unit": "subject (test-window weights averaged per subject, then over repeats)",
            "notes": "dotted line = uniform weight 1/K; weights describe the fitted model, not biological importance"}
    return fig, pd.DataFrame(data), meta, [r["path"] for _, r in rows]
