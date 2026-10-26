"""Synthetic stores for tests and the cross-dataset smoke run (no EEG; results on them are never results).

* the D3 stores: ``adgraph.synthetic.write_release_stores`` at the release paths (dk68 for Exp 0, lobes12), with
  synthetic ages / sex / MMSE in the participant tables (similar ages for AD and HC, as in ds004504);
* the D2 stores: 23 participants (10 AD, 13 HC) in the cross-dataset namespace (``d2store.store_dir``), the same
  schema, a planted dataset shift (gain of all band powers) and an AD-older age structure, as in D2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..checkpoint import write_marker
from ..config import config_hash
from ..graphs.store import GraphStore, window_id
from ..synthetic import make_store, write_release_stores
from . import d2store


def _demographics(subj: pd.DataFrame, ad_age: tuple, hc_age: tuple, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = subj.copy()
    y = s["y"].to_numpy()
    s["age"] = np.where(y == 1, rng.normal(*ad_age, len(s)), rng.normal(*hc_age, len(s))).round()
    s["sex"] = rng.choice(["F", "M"], len(s))
    s["mmse"] = np.where(y == 1, rng.integers(12, 25, len(s)), 30)
    return s


def write_d3(processed, n_windows: int = 6, seed: int = 0, **effects) -> dict:
    out = {}
    for atlas in ("dk68", "lobes12"):
        out.update({(inv, mode, atlas): d for (inv, mode), d in
                    write_release_stores(processed, atlas=atlas, n_windows=n_windows, seed=seed, **effects).items()})
    for d in out.values():
        s = pd.read_csv(d / "subjects.csv")
        _demographics(s, (66.4, 7.9), (67.9, 5.4), seed).to_csv(d / "subjects.csv", index=False)
    return out


def d2_store(n_windows: int, inverse: str, mode: str, seed: int = 100, shift: float = 0.3, **effects) -> GraphStore:
    st = make_store(n_windows, atlas="lobes12", seed=seed, n_ad=10, n_hc=13, mode=mode, inverse=inverse, **effects)
    rename = {k: k.replace("D3-", "D2-") for k in st.subjects["key"]}
    w = st.windows.copy()
    w["key"] = w["key"].map(rename)
    w["dataset"] = "D2"
    w["window_id"] = [window_id(k, i) for k, i in zip(w["key"], w["window_index"])]
    s = st.subjects.copy()
    s["key"], s["dataset"] = s["key"].map(rename), "D2"
    s["path"] = [f"/synthetic/{k}.edf" for k in s["key"]]
    s = _demographics(s, (80.0, 6.0), (62.0, 4.0), seed)
    a = dict(st.arrays)
    for p in ("psd", "sensor_psd"):                             # dataset shift: a gain of every band power
        a[f"{p}_abs"] = a[f"{p}_abs"] * np.exp(shift)
        a[f"{p}_log"] = a[f"{p}_log"] + shift / np.log(10)
        a[f"{p}_de"] = a[f"{p}_de"] + 0.5 * shift
    a = {k: v.astype(np.float32) for k, v in a.items()}
    cfg = d2store.resolve(inverse)
    man = {**st.manifest, "graph_run_id": f"synthetic-D2-{inverse}-{mode}-{seed}", "config": cfg,
           "config_hash": config_hash(cfg), "cross_dataset": {"dataset": "D2", "synthetic": True}}
    return GraphStore(a, w, st.bands, st.band_edges, st.roi_names, st.channels, s, man)


def write_d2(processed, n_windows: int = 6, seed: int = 100, shift: float = 0.3, **effects) -> dict:
    out = {}
    for inv in ("lcmv", "wmne"):
        for mode in ("B", "A"):
            st = d2_store(min(n_windows, 10) if mode == "A" else n_windows, inv, mode, seed + (17 if inv == "wmne"
                                                                                               else 0), shift,
                          **effects)
            d = d2store.store_dir(processed, inv, mode)
            st.save(d)
            (d / "validation.json").write_text(json.dumps({"passed": True, "synthetic": True,
                                                           "graph_run_id": st.manifest["graph_run_id"],
                                                           "checks": {}}))
            write_marker(d, "assembly", {"kind": "synthetic_d2"}, ["graphs.npz", "windows.csv", "subjects.csv",
                                                                   "manifest.json"])
            out[(inv, mode)] = d
    return out


def write_all(processed, n_windows: int = 6, **effects) -> dict:
    """Synthetic D3 (release paths) and D2 (namespace) stores with the same planted disease effects."""
    return {"D3": write_d3(processed, n_windows, **effects), "D2": write_d2(processed, n_windows, **effects)}
