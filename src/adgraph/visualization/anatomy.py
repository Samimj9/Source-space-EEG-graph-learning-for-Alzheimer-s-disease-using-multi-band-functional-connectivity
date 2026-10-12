"""ROI grouping/ordering for Desikan-Killiany (aparc) labels and subject-level averaging.

Lobe assignment follows the common Desikan-Killiany lobar grouping (frontal,
parietal, temporal, occipital, cingulate, insula). It is used only to order and
summarize ROIs in figures; labels not in the table (e.g. synthetic ROIs) go to
``other``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOBES = {
    "frontal": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
                "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal", "precentral",
                "paracentral", "frontalpole"],
    "parietal": ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"],
    "temporal": ["superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
                 "transversetemporal", "entorhinal", "temporalpole", "parahippocampal"],
    "occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
    "cingulate": ["rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate", "isthmuscingulate"],
    "insula": ["insula"],
}
LOBE_ORDER = list(LOBES) + ["other"]
_LOBE_OF = {roi: lobe for lobe, rois in LOBES.items() for roi in rois}


def parse_roi(name: str) -> tuple[str, str]:
    """('superiorfrontal-lh') -> ('superiorfrontal', 'lh'); hemisphere '' if absent."""
    base, _, hemi = name.rpartition("-")
    return (base, hemi) if hemi in ("lh", "rh") else (name, "")


def roi_table(roi_names: list[str]) -> pd.DataFrame:
    rows = []
    for i, n in enumerate(roi_names):
        base, hemi = parse_roi(n)
        rows.append({"index": i, "roi": n, "base": base, "hemi": hemi, "lobe": _LOBE_OF.get(base, "other")})
    t = pd.DataFrame(rows)
    t["order"] = t.apply(lambda r: (r["hemi"] != "lh", LOBE_ORDER.index(r["lobe"]), r["base"]), axis=1)
    return t.sort_values("order").drop(columns="order").reset_index(drop=True)


def subject_level(store, array: str, dataset: str | None = None):
    """Average a per-window array over each subject's windows -> (subjects table, (n_subj, ...) array).

    The subject, not the window, is the unit of every group summary built on top of this.
    """
    w = store.windows
    mask = np.ones(len(w), bool) if dataset is None else (w["dataset"] == dataset).to_numpy()
    keys = w.loc[mask, "key"].to_numpy()
    arr = store.arrays[array][mask]
    uniq = list(dict.fromkeys(keys))
    out = np.stack([arr[keys == k].mean(axis=0) for k in uniq])
    subj = (w.loc[mask].drop_duplicates("key").set_index("key").loc[uniq, ["dataset", "y", "label"]]
            .reset_index())
    subj["n_windows"] = [int((keys == k).sum()) for k in uniq]
    return subj, out
