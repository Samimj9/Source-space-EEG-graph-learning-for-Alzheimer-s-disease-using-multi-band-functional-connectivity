"""Synthetic graph runs with the real schema, for tests and smoke runs (no EEG involved).

Group effects can be planted so that pipelines can be checked end to end:
* ``psd_effect``: AD subjects get higher relative theta and lower relative alpha power;
* ``conn_effect``: AD subjects get weaker alpha1 PLV/ciPLV within a block of ROIs;
* ``dataset_shift``: additive offset of D2 log power (a site effect).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graphs.store import GraphStore, window_id

BANDS = {"theta": [4.0, 8.0], "alpha1": [8.0, 10.0], "alpha2": [10.0, 13.0], "beta": [13.0, 30.0]}
DEFAULT_COHORT = {"D1": (10, 10), "D2": (10, 13)}       # (n_AD, n_HC), as in the real cohort


def make_synthetic_store(cohort: dict | None = None, windows_per_subject: tuple[int, int] = (5, 9), n_rois: int = 68,
                         n_channels: int = 30, psd_effect: float = 0.0, conn_effect: float = 0.0,
                         dataset_shift: float = 0.0, sfreq: float = 250.0, window_s: float = 20.0,
                         step_s: float = 10.0, seed: int = 0) -> GraphStore:
    rng = np.random.default_rng(seed)
    cohort = cohort or DEFAULT_COHORT
    k = len(BANDS)
    plv, ciplv, psd_abs, s_abs, rows, subj_rows = [], [], [], [], [], []
    block = np.arange(min(10, n_rois))
    for ds, (n_ad, n_hc) in cohort.items():
        for sid, y in enumerate([1] * n_ad + [0] * n_hc, start=1):
            key = f"{ds}-{sid}"
            nw = int(rng.integers(windows_per_subject[0], windows_per_subject[1] + 1))
            subj_level = rng.normal(0, 0.3)
            for w in range(nw):
                base = rng.beta(2, 5, size=(k, n_rois, n_rois))
                base = (base + base.transpose(0, 2, 1)) / 2
                if y == 1 and conn_effect:
                    base[1][np.ix_(block, block)] *= max(0.0, 1 - conn_effect)
                for b in range(k):
                    np.fill_diagonal(base[b], 0.0)
                plv.append(base)
                ciplv.append(base * rng.uniform(0.2, 0.6, size=(k, 1, 1)))
                spec = np.array([1.0, 1.4, 1.0, 0.6])[:, None] * np.exp(rng.normal(subj_level, 0.2, (k, n_rois)))
                if y == 1 and psd_effect:
                    spec[0] *= 1 + psd_effect
                    spec[1:3] *= max(0.05, 1 - psd_effect)
                if ds == "D2" and dataset_shift:
                    spec *= 10 ** dataset_shift
                psd_abs.append(spec * 1e-18)
                s_abs.append(spec[:, :n_channels] if n_channels <= n_rois else
                             np.resize(spec, (k, n_channels)))
                start = int((10 + w * step_s) * sfreq)
                stop = start + int(window_s * sfreq)
                rows.append({"window_id": window_id(key, w), "key": key, "dataset": ds, "subject_id": sid,
                             "label": "AD" if y else "HC", "y": y, "window_index": w, "start_sample": start,
                             "stop_sample": stop, "tmin": start / sfreq, "tmax": stop / sfreq,
                             "sample_weight": 1.0 / nw, "qc_frac_channels_over_threshold": 0.0,
                             "qc_max_ptp_uv": 50.0})
            subj_rows.append({"key": key, "dataset": ds, "subject_id": sid, "label": "AD" if y else "HC", "y": y,
                              "age": float(rng.normal(80 if y else 65, 6)), "status": "ok",
                              "median_std_uv": float(rng.uniform(8, 20)), "n_windows_kept": nw})
    psd_abs = np.array(psd_abs, np.float64)
    s_abs = np.array(s_abs, np.float64) * 1e12
    total = lambda a: a.sum(1, keepdims=True) * 1.3                       # noqa: E731 (power outside 4-30 Hz)
    arrays = {
        "plv": np.array(plv, np.float32), "ciplv": np.array(ciplv, np.float32),
        "psd_abs": psd_abs.astype(np.float32), "psd_log": np.log10(psd_abs).astype(np.float32),
        "psd_rel": (psd_abs / total(psd_abs)).astype(np.float32),
        "sensor_psd_abs": s_abs.astype(np.float32), "sensor_psd_log": np.log10(s_abs).astype(np.float32),
        "sensor_psd_rel": (s_abs / total(s_abs)).astype(np.float32),
    }
    manifest = {"synthetic": True, "config_hash": "synthetic", "registry": {"subset_limit": None},
                "source_operator": {"shared_across_datasets": True, "verification_rel_error": 0.0,
                                    "channels": [f"C{i}" for i in range(n_channels)]}}
    return GraphStore(arrays, pd.DataFrame(rows), list(BANDS), np.array(list(BANDS.values())),
                      [f"roi{i:02d}" for i in range(n_rois)], [f"C{i}" for i in range(n_channels)],
                      pd.DataFrame(subj_rows), manifest)
