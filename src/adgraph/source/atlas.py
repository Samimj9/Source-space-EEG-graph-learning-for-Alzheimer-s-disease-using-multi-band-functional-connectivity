"""Cortical atlases of the final framework: the 68 Desikan-Killiany labels and three coarser aggregations of them
(docs/final/parcellation.md).

Every coarser region is a UNION of whole Desikan-Killiany labels of ONE hemisphere (contiguous gyral / lobar
groups). Signals of a coarse region are never formed by averaging 68-ROI signals or connectivity: the merged
``mne.Label`` goes through the same ``mean_flip`` extraction as a Desikan-Killiany label (sign flip from the merged
label's own cortical normals, then the mean over all of its sources), so a coarse store is built by the unchanged
store pipeline with ``source.atlas`` set (``sourcegraphs.config.resolve``).

    dk68        68  the Desikan-Killiany atlas (the existing stores)
    dk34_gyral  34  17 gyral groups per hemisphere (adjacent labels of one gyrus / functional strip merged)
    r18         18   9 large regions per hemisphere (prefrontal, orbitofrontal, sensorimotor, parietal, lateral and
                    medial temporal, occipital, cingulate, insula)
    lobes12     12   6 lobes per hemisphere (frontal, parietal, temporal, occipital, cingulate, insula; the grouping
                    of ``models.xu_dual.DK_LOBES``)

Hemispheres are never merged: homologous left / right regions are spatially distant, their mean_flip signs are
unrelated, and averaging them would cancel rather than summarize activity.
"""

from __future__ import annotations

import json
from functools import reduce
from pathlib import Path

import numpy as np

HEMIS = ("lh", "rh")
DK_REGIONS = ("bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus", "entorhinal", "frontalpole",
              "fusiform", "inferiorparietal", "inferiortemporal", "insula", "isthmuscingulate", "lateraloccipital",
              "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal", "paracentral",
              "parahippocampal", "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
              "postcentral", "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate",
              "rostralmiddlefrontal", "superiorfrontal", "superiorparietal", "superiortemporal", "supramarginal",
              "temporalpole", "transversetemporal")          # the 34 aparc labels per hemisphere

# lobe of each Desikan-Killiany label (conventional FreeSurfer lobar assignment; cingulate and insula separate)
DK_LOBE = {
    **{r: "frontal" for r in ("superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
                              "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal",
                              "precentral", "paracentral", "frontalpole")},
    **{r: "parietal" for r in ("superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus")},
    **{r: "temporal" for r in ("superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
                               "transversetemporal", "entorhinal", "temporalpole", "parahippocampal")},
    **{r: "occipital" for r in ("lateraloccipital", "lingual", "cuneus", "pericalcarine")},
    **{r: "cingulate" for r in ("rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate",
                                "isthmuscingulate")},
    "insula": "insula",
}

# region -> member Desikan-Killiany labels (per hemisphere); the lobe of a merged region is given explicitly
GROUPINGS = {
    "dk34_gyral": {
        "superiorfrontalgyrus": (("superiorfrontal", "frontalpole"), "frontal"),
        "middlefrontalgyrus": (("rostralmiddlefrontal", "caudalmiddlefrontal"), "frontal"),
        "inferiorfrontalgyrus": (("parsopercularis", "parstriangularis", "parsorbitalis"), "frontal"),
        "orbitofrontal": (("lateralorbitofrontal", "medialorbitofrontal"), "frontal"),
        "sensorimotor": (("precentral", "postcentral", "paracentral"), "frontal"),
        "superiorparietal": (("superiorparietal",), "parietal"),
        "inferiorparietal": (("inferiorparietal", "supramarginal"), "parietal"),
        "precuneus": (("precuneus",), "parietal"),
        "anteriorcingulate": (("rostralanteriorcingulate", "caudalanteriorcingulate"), "cingulate"),
        "posteriorcingulate": (("posteriorcingulate", "isthmuscingulate"), "cingulate"),
        "superiortemporalgyrus": (("superiortemporal", "transversetemporal", "bankssts"), "temporal"),
        "middletemporalgyrus": (("middletemporal",), "temporal"),
        "inferiortemporalgyrus": (("inferiortemporal", "fusiform"), "temporal"),
        "medialtemporal": (("entorhinal", "parahippocampal", "temporalpole"), "temporal"),
        "lateraloccipital": (("lateraloccipital",), "occipital"),
        "medialoccipital": (("lingual", "cuneus", "pericalcarine"), "occipital"),
        "insula": (("insula",), "insula"),
    },
    "r18": {
        "prefrontal": (("superiorfrontal", "frontalpole", "rostralmiddlefrontal", "caudalmiddlefrontal",
                        "parsopercularis", "parstriangularis", "parsorbitalis"), "frontal"),
        "orbitofrontal": (("lateralorbitofrontal", "medialorbitofrontal"), "frontal"),
        "sensorimotor": (("precentral", "postcentral", "paracentral"), "frontal"),
        "parietal": (("superiorparietal", "inferiorparietal", "supramarginal", "precuneus"), "parietal"),
        "lateraltemporal": (("superiortemporal", "transversetemporal", "bankssts", "middletemporal",
                             "inferiortemporal"), "temporal"),
        "medialtemporal": (("entorhinal", "parahippocampal", "temporalpole", "fusiform"), "temporal"),
        "occipital": (("lateraloccipital", "lingual", "cuneus", "pericalcarine"), "occipital"),
        "cingulate": (("rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate",
                       "isthmuscingulate"), "cingulate"),
        "insula": (("insula",), "insula"),
    },
    "lobes12": {lobe: (tuple(r for r in DK_REGIONS if DK_LOBE[r] == lobe), lobe)
                for lobe in ("frontal", "parietal", "temporal", "occipital", "cingulate", "insula")},
}
ATLASES = ("dk68", "dk34_gyral", "r18", "lobes12")
N_REGIONS = {"dk68": 68, "dk34_gyral": 34, "r18": 18, "lobes12": 12}


def _check() -> None:
    for name, g in GROUPINGS.items():
        members = [m for regions, _ in g.values() for m in regions]
        if sorted(members) != sorted(DK_REGIONS):
            raise AssertionError(f"atlas {name}: groups do not partition the 34 Desikan-Killiany labels")


_check()


def regions(atlas: str) -> dict[str, tuple[tuple[str, ...], str]]:
    """{region base name: (member Desikan-Killiany labels, lobe)} for one hemisphere."""
    if atlas == "dk68":
        return {r: ((r,), DK_LOBE[r]) for r in DK_REGIONS}
    if atlas not in GROUPINGS:
        raise ValueError(f"unknown atlas {atlas!r}; choose from {ATLASES}")
    return GROUPINGS[atlas]


def region_names(atlas: str) -> list[str]:
    """Region names ``<region>-<hemi>`` sorted by name (the store order convention of the Desikan-Killiany ROIs)."""
    return sorted(f"{r}-{h}" for r in regions(atlas) for h in HEMIS)


def members(atlas: str) -> dict[str, list[str]]:
    """{region name: [member Desikan-Killiany label names]} (``<label>-<hemi>``)."""
    out = {}
    for r, (labs, _) in regions(atlas).items():
        for h in HEMIS:
            out[f"{r}-{h}"] = [f"{x}-{h}" for x in labs]
    return {k: out[k] for k in sorted(out)}


def lobe_of(atlas: str) -> dict[str, str]:
    """{region name: lobe}; works for Desikan-Killiany names and every coarser atlas."""
    return {f"{r}-{h}": lobe for r, (_, lobe) in regions(atlas).items() for h in HEMIS}


def atlas_of(roi_names: list[str]) -> str:
    """The atlas whose region names (in store order) are exactly ``roi_names``."""
    for a in ATLASES:
        if list(roi_names) == region_names(a):
            return a
    raise ValueError(f"ROI names match no atlas of {ATLASES} ({len(roi_names)} names, first {list(roi_names)[:3]})")


def aggregation_matrix(atlas: str) -> np.ndarray:
    """(n_regions, 68) membership matrix M (M[i, j] = 1 if DK label j belongs to region i); for bookkeeping and
    tests only. It is NOT used to average 68-ROI signals or connectivity into coarse ones (see module docstring)."""
    dk = region_names("dk68")
    m = members(atlas)
    return np.array([[1.0 if lab in m[r] else 0.0 for lab in dk] for r in m])


def merged_labels(dk_labels: list, atlas: str) -> list:
    """MNE labels of ``atlas`` from the 68 Desikan-Killiany ``mne.Label`` objects (``source.inverse.load_labels``):
    each region is the union of its member labels (same hemisphere), named ``<region>-<hemi>``, in region_names
    order. For ``dk68`` the input labels are returned unchanged (identical objects, identical order)."""
    by_name = {lab.name: lab for lab in dk_labels}
    if atlas == "dk68":
        if [lab.name for lab in dk_labels] != region_names("dk68"):
            raise ValueError("Desikan-Killiany labels not in the expected sorted order")
        return list(dk_labels)
    out = []
    for name, labs in members(atlas).items():
        missing = [x for x in labs if x not in by_name]
        if missing:
            raise ValueError(f"atlas {atlas}: labels {missing} not found")
        merged = reduce(lambda a, b: a + b, [by_name[x] for x in labs])
        merged.name = name
        out.append(merged)
    return out


# ------------------------------------------------------------------------------------------- static atlas tables
TABLES = Path(__file__).with_name("atlas_tables.json")


def tables() -> dict:
    """Precomputed fsaverage geometry of every atlas (``scripts/make_atlas_tables.py``): region centroids (fsaverage
    white-surface coordinates, mm), vertex counts, and cortical-surface contiguity (two regions share a mesh edge)."""
    if not TABLES.exists():
        raise FileNotFoundError(f"{TABLES} missing: run scripts/make_atlas_tables.py once (needs fsaverage)")
    return json.loads(TABLES.read_text())


def contiguity(roi_names: list[str]) -> np.ndarray:
    """(R, R) binary matrix: 1 if two regions of the store's atlas share a cortical-surface edge (zero diagonal)."""
    atlas = atlas_of(roi_names)
    t = tables()[atlas]
    idx = {n: i for i, n in enumerate(t["regions"])}
    a = np.zeros((len(roi_names),) * 2, np.float32)
    for x, y in t["contiguous_pairs"]:
        i, j = roi_names.index(x), roi_names.index(y)
        a[i, j] = a[j, i] = 1.0
    if set(idx) != set(roi_names):
        raise ValueError("atlas table regions differ from the store ROIs")
    return a


def centroids(roi_names: list[str]) -> np.ndarray:
    """(R, 3) region centroids (mm, fsaverage surface coordinates) in store order."""
    t = tables()[atlas_of(roi_names)]
    c = dict(zip(t["regions"], t["centroids_mm"]))
    return np.array([c[n] for n in roi_names], float)
