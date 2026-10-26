"""D3 / D2 compatibility report and harmonization manifest (design sections 4-5).

``compatibility`` classifies every property that affects comparability as

* **identical**     verified equal in code or data (configuration key, tensor axis, names, order, definition);
* **approximate**   made as similar as the data allow, with the remaining difference stated;
* **different**     a known difference that harmonization cannot remove (stated, never hidden).

``shift_diagnostics`` describes the two datasets LABEL-BLIND (participant-level summaries per dataset, never per
diagnosis): windows per participant, rejection, amplitude, relative band power, mean PLV / |ciPLV| per band and the
standardized difference between the datasets. Descriptive only; nothing is fitted or selected on it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import config_hash
from ..graphs.store import CONNECTIVITY, GraphStore, check_adjacency, graph_run_id
from ..sourcegraphs.match import _config_diff
from . import channels

# configuration keys a D2 store may differ in (everything else must equal the D3 store configuration)
EXPECTED_CONFIG_DIFFERENCES = {
    "channels.channel_map_hash", "channels.coordinates.accepted_lookups", "channels.coordinates.compare_template",
    "channels.coordinates.head_radius_m", "channels.coordinates.source", "channels.coordinates.units",
    "channels.montage", "channels.selection", "channels.verify_layout", "cross_dataset", "datasets",
    "discontinuities.source", "input.format", "input.recordings", "output_subdir", "preprocessing.bandpass",
    "preprocessing.edge_crop_s", "preprocessing.settle_s", "split", "store_names", "units.scale.D2", "units.scale.D3"}

KNOWN = [  # (property, status, D3, D2, note)
    ("artifact cleaning", "different", "provider ASR + ICA (ICLabel eye / jaw components removed)", "none",
     "not reproduced on D2 without a validated implementation (design 4.3); the adaptive window rejection is identical"),
    ("acquisition site / hardware", "different", "AHEPA Thessaloniki, Nihon Kohden EEG 2100, 500 Hz",
     "Lebanese 2019 cohort, 32-channel cap, 1024 Hz (amplifier: see EDF header audit)", "dataset shift, not disease"),
    ("recording condition", "different", "eyes closed, sitting", "resting state (eyes condition to be confirmed by "
     "the author)", "must be confirmed before interpretation"),
    ("population", "different", "AD 66.4 +/- 7.9 y, HC 67.9 +/- 5.4 y (similar ages)",
     "AD about 80 y, HC about 62 y (strong age imbalance)", "age analyses, design section 9"),
    ("recording continuity", "different", "provider removed data (EEGLAB boundaries in 56 / 65 recordings)",
     "continuous", "no window crosses a boundary in either dataset"),
    ("band-pass", "approximate", "provider 0.5-45 Hz Butterworth (order / phase not documented)",
     "0.5-45 Hz Butterworth, order 4, zero phase (applied here)", "same pass band and family; order may differ"),
    ("line noise (50 Hz)", "approximate", "no notch (provider low-pass only)", "no notch (same low-pass)",
     "measured per recording in the audit; all band filters end at 45 Hz"),
    ("reference", "identical", "A1-A2 (provider), then common average of the 19 electrodes",
     "recording reference, then common average of the same 19 electrodes",
     "the common average of the same electrodes removes any common reference signal exactly"),
    ("original sampling rate", "approximate", "500 Hz", "1024 Hz", "each continuous segment resampled to 250 Hz"),
    ("edge handling", "approximate", "none added (provider filtered the whole recording)",
     "10 s crop before and 5 s after the band-pass at each end", "D2 band-pass transients removed"),
    ("electrode positions", "identical", "colin27 10-05 template (provider lookup), MRI frame",
     "the same positions (D3 geometry), verified against the D2 .loc layout", "neither dataset has digitized positions"),
    ("head model / source space", "identical", "fsaverage ico-5, 3-layer BEM, mindist 5 mm", "same",
     "template anatomy for every participant (source-localization uncertainty applies to both)"),
]


def _status(ok: bool) -> str:
    return "identical" if ok else "DIFFERENT (unexpected)"


def compatibility(d3: GraphStore, d2: GraphStore, d3_cfg: dict, d2_cfg: dict, table: dict) -> pd.DataFrame:
    """One row per property: status, D3, D2, note. Raises nothing: an unexpected difference is a row whose status
    starts with 'DIFFERENT'; ``passed`` = no such row."""
    rows = []

    def add(prop, ok, a, b, note=""):
        rows.append({"property": prop, "status": _status(ok), "D3": str(a), "D2": str(b), "note": note})

    diff = set(_config_diff(d3_cfg, d2_cfg))
    add("store configuration (all other keys)", diff <= EXPECTED_CONFIG_DIFFERENCES,
        config_hash(d3_cfg), config_hash(d2_cfg), f"differing keys: {sorted(diff)}; expected only the D2 overrides")
    add("ROI names and order (lobes12)", d3.roi_names == d2.roi_names, len(d3.roi_names), len(d2.roi_names))
    add("bands, order and edges", d3.bands == d2.bands and np.allclose(d3.band_edges, d2.band_edges),
        dict(zip(d3.bands, d3.band_edges.tolist())), dict(zip(d2.bands, d2.band_edges.tolist())))
    add("sensor channels and order", d3.channels == d2.channels == list(table["canonical_order"]), d3.channels,
        d2.channels, f"T3=T7, T4=T8, T5=P7, T6=P8 (table {channels.table_hash(table)})")
    same_arrays = sorted(d3.arrays) == sorted(d2.arrays)
    shapes = all(d3.arrays[k].shape[1:] == d2.arrays[k].shape[1:] for k in d3.arrays if k in d2.arrays)
    add("feature arrays and tensor dimensions", same_arrays and shapes,
        {k: list(v.shape[1:]) for k, v in sorted(d3.arrays.items())},
        {k: list(v.shape[1:]) for k, v in sorted(d2.arrays.items())})
    for k in ("bands", "filters", "features", "psd", "windowing", "source"):
        add(f"definition: {k}", d3_cfg.get(k) == d2_cfg.get(k), "D3 configuration", "same key in D2",
            "PLV / |ciPLV| / PSD / DE / analytic FIR / windows / inverse are computed by the same code")
    add("sampling rate after resampling", d3_cfg["preprocessing"]["sfreq"] == d2_cfg["preprocessing"]["sfreq"],
        d3_cfg["preprocessing"]["sfreq"], d2_cfg["preprocessing"]["sfreq"])
    for name, s in (("D3", d3), ("D2", d2)):
        fin = all(np.isfinite(v).all() for v in s.arrays.values())
        try:
            for c in CONNECTIVITY:
                if c in s.arrays:
                    check_adjacency(s.arrays[c], c)
            adj = True
        except ValueError:
            adj = False
        add(f"{name}: finite features, valid PLV / |ciPLV| adjacencies", fin and adj, "", "")
    for name, s in (("D3", d3), ("D2", d2)):
        w = s.windows
        add(f"{name}: one diagnosis per participant, AD = 1", bool((w.groupby("key")["y"].nunique() == 1).all()
                                                                  and ((w["label"] == "AD") == (w["y"] == 1)).all()),
            "", "")
        add(f"{name}: unique participants and windows", not w["window_id"].duplicated().any()
            and not s.subjects["key"].duplicated().any(), "", "")
    add("no participant in both datasets", not set(d3.windows["key"]) & set(d2.windows["key"]),
        sorted({k.split('-')[0] for k in d3.windows['key']}), sorted({k.split('-')[0] for k in d2.windows['key']}))
    add("graph construction", True, "dense (recorded)", "dense (same code, stateless per window)")
    add("normalization convention", True, "node features z-scored with SOURCE-training statistics inside the model",
        "same object; the target is never used to fit it", "no target normalization (zero-shot)")
    for prop, status, a, b, note in KNOWN:
        rows.append({"property": prop, "status": status, "D3": a, "D2": b, "note": note})
    return pd.DataFrame(rows)


def passed(table: pd.DataFrame) -> bool:
    return not table["status"].str.startswith("DIFFERENT").any()


def shift_diagnostics(stores: dict[str, GraphStore]) -> pd.DataFrame:
    """Label-blind participant summaries per dataset (median, IQR) and the standardized difference D2 vs D3."""
    rows = {}
    for ds, s in stores.items():
        w = s.windows
        per = pd.DataFrame({"key": w["key"].to_numpy()})
        for b, band in enumerate(s.bands):
            per[f"sensor_rel_power_{band}"] = s.arrays["sensor_psd_rel"][:, b].mean(1)
            for c in CONNECTIVITY:
                a = s.arrays[c][:, b]
                r = a.shape[-1]
                per[f"mean_{c}_{band}"] = (a.sum((-1, -2)) / (r * (r - 1)))
        p = per.groupby("key").mean()
        subj = s.subjects.set_index("key")
        for col in ("n_windows_kept", "median_std_uv", "n_windows_rejected", "n_windows_candidate",
                    "selected_duration_s"):
            if col in subj:
                p[col] = pd.to_numeric(subj[col], errors="coerce").reindex(p.index)
        rows[ds] = p
    out = []
    a, b = rows.get("D3"), rows.get("D2")
    for col in sorted(set(a.columns) & set(b.columns)):
        x, y = a[col].dropna(), b[col].dropna()
        sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2) if len(x) > 1 and len(y) > 1 else np.nan
        out.append({"feature": col, "D3_median": x.median(), "D3_iqr": x.quantile(.75) - x.quantile(.25),
                    "D2_median": y.median(), "D2_iqr": y.quantile(.75) - y.quantile(.25),
                    "standardized_difference_D2_minus_D3": (y.mean() - x.mean()) / sd if sd and sd > 0 else np.nan})
    return pd.DataFrame(out)


def harmonization_manifest(d3: GraphStore, d2: GraphStore, d3_cfg: dict, d2_cfg: dict, table: dict,
                           compat: pd.DataFrame, audit: dict | None = None) -> dict:
    from ..provenance import git_commit, versions

    return {"created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(), "versions": versions(),
            "channel_map": {"hash": channels.table_hash(table), "layout_sha256": table["layout_sha256"],
                            "canonical_order": table["canonical_order"], "equivalences": table["equivalences"],
                            "audit": audit},
            "stores": {"D3": {"root": str(d3.root), "graph_run_id": graph_run_id(d3),
                              "config_hash": config_hash(d3_cfg)},
                       "D2": {"root": str(d2.root), "graph_run_id": graph_run_id(d2),
                              "config_hash": config_hash(d2_cfg),
                              "base_d3_config_hash": d2_cfg.get("cross_dataset", {}).get("base_d3_config_hash")}},
            "d2_preprocessing": d2_cfg["preprocessing"],
            "compatibility": compat.to_dict("records"), "compatibility_passed": passed(compat),
            "claims_not_made": ["preprocessing is identical", "electrode matching removes dataset shift",
                                "D2 performance is disease-specific without the age analyses"]}
