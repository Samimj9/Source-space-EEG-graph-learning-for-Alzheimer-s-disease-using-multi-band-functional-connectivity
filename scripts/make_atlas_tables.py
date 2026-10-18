"""Precompute the static fsaverage geometry of every atlas -> src/adgraph/source/atlas_tables.json (committed).

    python scripts/make_atlas_tables.py --subjects-dir <dir containing fsaverage/>

Needs fsaverage's ``label/{lh,rh}.aparc.annot``, ``surf/{lh,rh}.white`` and ``bem/fsaverage-ico-5-src.fif`` (MNE's
fsaverage download; on Colab ``<cache_dir>/subjects``). No EEG is read. Per atlas and region: member labels, lobe,
number of surface vertices and of ico-5 source-space vertices, centroid (mean white-surface coordinate, mm), and
the cortical-surface contiguity (two regions are contiguous if a triangle of the white mesh joins them).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adgraph.source import atlas  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--subjects-dir", required=True)
args = parser.parse_args()

import mne  # noqa: E402

sd = Path(args.subjects_dir)
labels = mne.read_labels_from_annot("fsaverage", "aparc", subjects_dir=sd, verbose="error")
labels = sorted([lab for lab in labels if not lab.name.startswith(("unknown", "corpuscallosum"))],
                key=lambda lab: lab.name)
src = mne.read_source_spaces(sd / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif", verbose="error")
surf = {h: mne.read_surface(sd / "fsaverage" / "surf" / f"{h}.white") for h in atlas.HEMIS}
vert_label = {}
for h in atlas.HEMIS:
    lab_of = -np.ones(len(surf[h][0]), int)
    for i, lab in enumerate(labels):
        if lab.hemi == h:
            lab_of[lab.vertices] = i
    vert_label[h] = lab_of

out = {"source": "fsaverage white surface + aparc annotation + ico-5 source space (mne.datasets.fetch_fsaverage)",
       "mne": mne.__version__}
for name in atlas.ATLASES:
    mem = atlas.members(name)
    dk_to_region = {dk: r for r, dks in mem.items() for dk in dks}
    regions = list(mem)
    rows, pairs = [], set()
    for h_i, h in enumerate(atlas.HEMIS):
        tris = surf[h][1]
        lab = vert_label[h]
        region_of_label = np.array([regions.index(dk_to_region[lb.name]) if lb.hemi == h else -1 for lb in labels])
        reg = np.where(lab >= 0, region_of_label[np.maximum(lab, 0)], -1)
        for a, b in ((0, 1), (1, 2), (0, 2)):
            ra, rb = reg[tris[:, a]], reg[tris[:, b]]
            m = (ra >= 0) & (rb >= 0) & (ra != rb)
            for x, y in set(zip(ra[m].tolist(), rb[m].tolist())):
                pairs.add(tuple(sorted((regions[x], regions[y]))))
    for r in regions:
        h = r.rsplit("-", 1)[1]
        dks = [lb for lb in labels if lb.name in mem[r]]
        verts = np.concatenate([lb.vertices for lb in dks])
        s = src[atlas.HEMIS.index(h)]
        rows.append({"region": r, "members": mem[r], "lobe": atlas.lobe_of(name)[r], "n_vertices": int(len(verts)),
                     "n_source_vertices": int(np.isin(s["vertno"], verts).sum()),
                     "centroid_mm": (surf[h][0][verts].mean(0)).round(2).tolist()})
    out[name] = {"regions": regions, "members": {r["region"]: r["members"] for r in rows},
                 "lobes": {r["region"]: r["lobe"] for r in rows},
                 "n_vertices": {r["region"]: r["n_vertices"] for r in rows},
                 "n_source_vertices": {r["region"]: r["n_source_vertices"] for r in rows},
                 "centroids_mm": [r["centroid_mm"] for r in rows],
                 "contiguous_pairs": sorted([list(p) for p in pairs])}
    print(f"{name}: {len(regions)} regions, {len(pairs)} contiguous pairs, source vertices per region "
          f"{min(out[name]['n_source_vertices'].values())}-{max(out[name]['n_source_vertices'].values())}")
path = Path(atlas.__file__).with_name("atlas_tables.json")
path.write_text(json.dumps(out, indent=1))
print(f"written {path}")
