"""Synthetic stores with the schema of the real ds004504 source-graph stores, for tests and CPU smoke runs only
(no EEG involved; results on them are never results).

``make_store``: 65 participants D3-1 ... D3-65 (36 AD, 29 HC, AD = keys 1-36 as in ds004504), the five Xu bands,
the ROIs of an atlas (default the 68 Desikan-Killiany labels in store order), 19 sensors, ``n_windows`` windows per
participant, every array of the real stores (PLV, |ciPLV|, PSD abs / log / rel / DE, sensor PSD and DE) and the
participants table. Planted, label-dependent effects (all optional):
* ``psd_effect``: AD participants get higher relative theta and lower relative alpha power;
* ``conn_effect``: AD participants get weaker alpha PLV / |ciPLV| inside a block of ROIs;
* ``band_direction``: AD log power up in theta and down in alpha by the same amount (invisible to a band-invariant
  model, the band-identity check).
``save_store`` writes it like an assembled store (graphs.npz, windows.csv, subjects.csv, manifest.json,
xu_split.json, validation.json, assembly marker) under the directory the store registry expects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .graphs.store import GraphStore, window_id

BANDS = {"delta": [0.5, 4.0], "theta": [4.0, 8.0], "alpha": [8.0, 13.0], "beta": [13.0, 25.0],
         "gamma": [25.0, 45.0]}
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz",
            "Cz", "Pz"]


def make_store(n_windows: int = 6, atlas: str = "dk68", psd_effect: float = 0.0, conn_effect: float = 0.0,
               band_direction: float = 0.0, seed: int = 0, n_ad: int = 36, n_hc: int = 29, mode: str = "B",
               inverse: str = "lcmv", noise: float = 1.0) -> GraphStore:
    from .source.atlas import region_names

    rng = np.random.default_rng(seed)
    rois = region_names(atlas)
    r, k, c = len(rois), len(BANDS), len(CHANNELS)
    keys = [f"D3-{i}" for i in range(1, n_ad + n_hc + 1)]
    labels = {kk: int(i < n_ad) for i, kk in enumerate(keys)}
    block = np.arange(min(10, r))
    rows, subj = [], []
    arr = {name: [] for name in ("plv", "ciplv", "psd_abs", "psd_de", "sensor_psd_abs", "sensor_psd_de")}
    for sid, key in enumerate(keys, start=1):
        y = labels[key]
        base_p = rng.normal(0, 0.3, (k, r))                         # participant-specific spectral fingerprint
        base_c = rng.uniform(0.2, 0.5, (k, r, r))
        for w in range(n_windows):
            p = np.exp(base_p + rng.normal(0, 0.2 * noise, (k, r))) * np.array([4, 3, 5, 2, 1.0])[:, None]
            if y:
                p[1] *= np.exp(psd_effect)                          # theta up
                p[2] *= np.exp(-psd_effect)                         # alpha down
                p[1] *= np.exp(band_direction)
                p[2] *= np.exp(-band_direction)
            a = np.clip(base_c + rng.normal(0, 0.05 * noise, (k, r, r)), 0, 1)
            if y and conn_effect:
                a[2][np.ix_(block, block)] *= max(0.0, 1 - 0.3 * conn_effect)
            a = (a + np.swapaxes(a, -1, -2)) / 2
            a[:, np.arange(r), np.arange(r)] = 0
            ci = np.clip(a * rng.uniform(0.3, 0.7, (k, r, r)), 0, 1)
            ci = (ci + np.swapaxes(ci, -1, -2)) / 2
            ci[:, np.arange(r), np.arange(r)] = 0
            sp = np.exp(rng.normal(0, 0.3, (k, c))) * np.array([4, 3, 5, 2, 1.0])[:, None]
            arr["plv"].append(a)
            arr["ciplv"].append(ci)
            arr["psd_abs"].append(p * 1e-12)
            arr["psd_de"].append(0.5 * np.log(2 * np.pi * np.e * p * 1e-12))
            arr["sensor_psd_abs"].append(sp * 1e-11)
            arr["sensor_psd_de"].append(0.5 * np.log(2 * np.pi * np.e * sp * 1e-11))
            start = w * 250
            rows.append({"window_id": window_id(key, w), "key": key, "dataset": "D3", "subject_id": sid,
                         "label": "AD" if y else "HC", "y": y, "window_index": w, "start_sample": start,
                         "stop_sample": start + 2500, "tmin": start / 250, "tmax": (start + 2500) / 250,
                         "sample_weight": 1.0 / n_windows, "segment_index": 0, "piece": 0,
                         "xu_tenfold": w < 10})
        subj.append({"key": key, "dataset": "D3", "subject_id": sid, "label": "AD" if y else "HC", "y": y,
                     "status": "ok", "reason": "", "n_windows_candidate": n_windows + 2,
                     "n_windows_accepted": n_windows, "n_windows_rejected": 2, "n_windows_kept": n_windows,
                     "n_tenfold": min(10, n_windows), "rejection_threshold_uv": float(rng.uniform(60, 120)),
                     "median_std_uv": float(rng.uniform(8, 20)), "selected_duration_s": 300.0})
    out = {k_: np.asarray(v, np.float32) for k_, v in arr.items()}
    tot = out["psd_abs"].sum(1, keepdims=True)
    out["psd_log"] = np.log10(out["psd_abs"])
    out["psd_rel"] = out["psd_abs"] / tot
    stot = out["sensor_psd_abs"].sum(1, keepdims=True)
    out["sensor_psd_log"] = np.log10(out["sensor_psd_abs"])
    out["sensor_psd_rel"] = out["sensor_psd_abs"] / stot
    out = {k_: v.astype(np.float32) for k_, v in out.items()}
    windows = pd.DataFrame(rows)
    manifest = {"synthetic": True, "mode": {"B": "B_all_windows", "A": "A_xu_tenfold"}[mode],
                "graph_run_id": f"synthetic-{inverse}-{mode}-{atlas}-{seed}",
                "source_operator": {"method": inverse, "roi_names": rois, "channels": CHANNELS}}
    return GraphStore(out, windows, list(BANDS), np.array(list(BANDS.values()), float), rois, CHANNELS,
                      pd.DataFrame(subj), manifest)


def write_release_stores(processed: str | Path, inverses=("lcmv", "wmne"), modes=("B", "A"), atlas: str = "dk68",
                         n_windows: int = 4, seed: int = 0, **effects) -> dict[tuple[str, str], Path]:
    """Synthetic stores at exactly the directories the release registry resolves (``stores.store_dir``), with the
    resolved configuration hash, ``validation.json`` and, for wMNE, a passed ``match.json`` against the LCMV store of
    the same mode: so smoke runs and end-to-end tests go through the real verification, lock and QC gate. Mode A
    keeps the first 10 windows. The wMNE store has the same windows and sensor features as the LCMV one and
    different source features, like the real matched stores."""
    from .config import config_hash
    from .sourcegraphs.config import resolve
    from .stores import store_dir

    out = {}
    for mode in modes:
        nw = min(n_windows, 10) if mode == "A" else n_windows
        ref = None
        for inv in ("lcmv", "wmne"):
            if inv not in inverses and not (inv == "lcmv" and "wmne" in inverses):
                continue
            st = make_store(nw, atlas=atlas, seed=seed + (17 if inv == "wmne" else 0), mode=mode, inverse=inv,
                            **effects)
            if ref is not None:                          # matched windows / participants / sensor features
                st.windows = ref.windows.copy()
                for k in [k for k in st.arrays if k.startswith("sensor_")]:
                    st.arrays[k] = ref.arrays[k].copy()
            d = store_dir(processed, inv, mode, atlas)
            save_store(st, d, config_hash(resolve(inv, atlas)))
            if inv == "wmne":
                rid = json.loads((store_dir(processed, "lcmv", mode, atlas) / "manifest.json").read_text())
                (d / "match.json").write_text(json.dumps({"passed": True, "graph_run_id": st.manifest["graph_run_id"],
                                                          "reference_graph_run_id": rid["graph_run_id"],
                                                          "checks": {}}))
            ref = st if inv == "lcmv" else ref
            out[(inv, mode)] = d
    return out


def save_store(store: GraphStore, directory: str | Path, config_hash: str | None = None) -> Path:
    """Write ``store`` as an assembled, validated store (for registry and end-to-end tests)."""
    from .checkpoint import write_marker
    from .evaluation.splits import subject_table, xu_partition

    d = Path(directory)
    store.manifest = {**store.manifest, **({"config_hash": config_hash} if config_hash else {})}
    store.save(d)
    dev, test = xu_partition(subject_table(store.windows))
    (d / "xu_split.json").write_text(json.dumps({"development": dev, "test": test}))
    (d / "validation.json").write_text(json.dumps({"passed": True, "graph_run_id": store.manifest["graph_run_id"],
                                                   "checks": {}}))
    write_marker(d, "assembly", {"kind": "synthetic"}, ["graphs.npz", "windows.csv", "subjects.csv", "manifest.json"])
    return d
