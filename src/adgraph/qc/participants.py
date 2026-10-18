"""Participant-level quality control of a store, before any classification (docs/final/qc.md).

Three categories are kept strictly apart and NOTHING is excluded automatically:

A. **Technical data corruption** (``technical_problems``): non-finite values, negative or zero power, connectivity
   outside [0, 1], a non-zero adjacency diagonal or asymmetry, constant (flat) features over all windows of a
   participant, degenerate connectivity (mean off-diagonal PLV ~ 1: identical ROI signals), duplicate windows, or
   window counts that disagree with ``subjects.csv``, a source operator that does not reproduce MNE's inverse. An
   experiment refuses to start while any category A finding
   has no recorded decision (``decisions``).
B. **Statistically unusual but valid EEG** (``robust_flags``): a participant whose summary feature lies more than
   3.5 robust standard deviations from the reference participants' median (modified z-score of Iglewicz & Hoaglin,
   z = 0.6745 (x - median) / MAD), or who is a multivariate outlier of the robustly standardized summaries
   (Minimum Covariance Determinant on the leading principal components, chi-square 0.999 cut-off). Flagged for
   review and kept.
C. **Difficult to classify** (``qc.difficult``): from out-of-fold predictions only, after the experiments. Kept.

All summaries are label-blind (the diagnosis is attached only to describe how flags are distributed). The cohort
report uses the 52 development participants as the reference distribution by default; the fold-level QC
(``qc.fold``) refits the reference on each fold's training participants only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..graphs.store import GraphStore

Z_THRESHOLD = 3.5                 # modified z-score (Iglewicz & Hoaglin 1993)
MCD_QUANTILE = 0.999              # chi-square quantile of the robust Mahalanobis distance
PLV_DEGENERATE = 0.99             # mean off-diagonal PLV of a window above this: ROI signals are (nearly) identical
VERIFICATION_TOL = 1e-6           # the store builder's own tolerance for the operator vs MNE
BLOCK = 512


def _offdiag_mean(a: np.ndarray) -> np.ndarray:
    """Mean off-diagonal value of (..., R, R) -> (..., R) (node strength) computed blockwise."""
    r = a.shape[-1]
    return (a.sum(-1) - np.diagonal(a, axis1=-2, axis2=-1)) / (r - 1)


def window_summaries(store: GraphStore) -> pd.DataFrame:
    """Per-window, label-free summaries: per band the ROI-median log / relative PSD and DE (source), the sensor
    log PSD, the mean PLV and |ciPLV| node strength, and the across-ROI SD of the PLV strength."""
    rows = {"window_id": store.windows["window_id"].to_numpy(), "key": store.windows["key"].to_numpy()}
    for kind in ("log", "rel", "de"):
        arr = store.arrays.get(f"psd_{kind}")
        if arr is not None:
            med = np.median(arr, axis=-1)
            for b, name in enumerate(store.bands):
                rows[f"psd_{kind}_{name}"] = med[:, b]
    if "psd_log" in store.arrays:                       # source estimate: spread of log power over ROIs (a
        spread = np.std(store.arrays["psd_log"], axis=-1)   # degenerate operator makes every ROI alike)
        for b, name in enumerate(store.bands):
            rows[f"source_log_power_roi_sd_{name}"] = spread[:, b]
    s = store.arrays.get("sensor_psd_log")
    if s is not None:
        med = np.median(s, axis=-1)
        for b, name in enumerate(store.bands):
            rows[f"sensor_psd_log_{name}"] = med[:, b]
    for conn in ("plv", "ciplv"):
        arr = store.arrays.get(conn)
        if arr is None:
            continue
        mean = np.empty((len(arr), len(store.bands)))
        sd = np.empty_like(mean)
        for i in range(0, len(arr), BLOCK):
            st = _offdiag_mean(np.asarray(arr[i:i + BLOCK], np.float64))
            mean[i:i + BLOCK], sd[i:i + BLOCK] = st.mean(-1), st.std(-1)
        for b, name in enumerate(store.bands):
            rows[f"{conn}_strength_{name}"] = mean[:, b]
            rows[f"{conn}_strength_sd_{name}"] = sd[:, b]
    if "psd_log" in store.arrays and "sensor_psd_log" in store.arrays:
        rows["source_minus_sensor_log_power"] = (np.median(store.arrays["psd_log"], axis=(1, 2))
                                                 - np.median(store.arrays["sensor_psd_log"], axis=(1, 2)))
    return pd.DataFrame(rows)


def participant_summaries(store: GraphStore, win: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per participant: medians of the window summaries, their across-window variability (median absolute
    deviation of the window values, a stability measure) and the recording / rejection statistics of
    ``subjects.csv``."""
    win = window_summaries(store) if win is None else win
    feats = [c for c in win.columns if c not in ("window_id", "key")]
    g = win.groupby("key")[feats]
    med = g.median()
    mad = g.agg(lambda x: float(np.median(np.abs(x - np.median(x)))))
    mad.columns = [f"{c}__window_mad" for c in mad.columns]
    out = pd.concat([med, mad[[f"{c}__window_mad" for c in feats if c.startswith(("plv_strength_", "psd_log_"))]]],
                    axis=1)
    out["n_windows"] = win.groupby("key").size()
    subj = store.subjects.set_index("key") if not store.subjects.empty else pd.DataFrame(index=out.index)
    for col in ("n_windows_candidate", "n_windows_rejected", "rejection_threshold_uv", "median_std_uv",
                "selected_duration_s", "n_boundary_events", "n_segments", "n_pieces", "lcmv_rank"):
        if col in subj:
            out[col] = pd.to_numeric(subj[col], errors="coerce").reindex(out.index)
    if {"n_windows_candidate", "n_windows_rejected"} <= set(out.columns):
        out["rejected_fraction"] = out["n_windows_rejected"] / out["n_windows_candidate"].where(
            out["n_windows_candidate"] > 0)
    return out.sort_index()


def technical_problems(store: GraphStore) -> pd.DataFrame:
    """Category A findings, one row per (participant, check) that failed; empty when the store is clean."""
    rows = []
    w = store.windows
    keys = w["key"].to_numpy()
    if w["window_id"].duplicated().any():
        for k in sorted(set(w.loc[w["window_id"].duplicated(), "key"])):
            rows.append({"key": k, "check": "duplicate window ids", "detail": ""})
    for name, arr in store.arrays.items():
        bad_nf, bad_range, bad_flat = set(), set(), set()
        for i in range(0, len(arr), BLOCK):
            blk = np.asarray(arr[i:i + BLOCK])
            nf = ~np.isfinite(blk).reshape(len(blk), -1).all(1)
            bad_nf |= set(keys[i:i + BLOCK][nf])
            if name in ("plv", "ciplv"):
                out = (blk < -1e-5).reshape(len(blk), -1).any(1) | (blk > 1 + 1e-5).reshape(len(blk), -1).any(1)
                diag = np.abs(np.diagonal(blk, axis1=-2, axis2=-1)).reshape(len(blk), -1).max(axis=1) > 1e-5
                asym = np.abs(blk - np.swapaxes(blk, -1, -2)).reshape(len(blk), -1).max(axis=1) > 1e-5
                bad_range |= set(keys[i:i + BLOCK][out | diag | asym])
                if name == "plv":
                    deg = _offdiag_mean(blk.astype(np.float64)).mean(-1).max(-1) > PLV_DEGENERATE
                    bad_flat |= set(keys[i:i + BLOCK][deg])
            if name.endswith("psd_abs"):
                bad_range |= set(keys[i:i + BLOCK][(blk <= 0).reshape(len(blk), -1).any(1)])
        for k in sorted(bad_nf):
            rows.append({"key": k, "check": f"non-finite values in {name}", "detail": ""})
        for k in sorted(bad_range):
            rows.append({"key": k, "check": f"{name} outside its valid range / not a valid adjacency", "detail": ""})
        for k in sorted(bad_flat):
            rows.append({"key": k, "check": "degenerate connectivity (mean PLV ~ 1: identical ROI signals)",
                         "detail": ""})
        if name.startswith(("psd_", "sensor_psd_")) and arr.ndim == 3:
            for k, idx in w.groupby("key").indices.items():
                if len(idx) > 1 and (np.ptp(arr[idx], axis=0) == 0).any():
                    rows.append({"key": k, "check": f"{name}: a feature is constant over all windows (flat)",
                                 "detail": ""})
    sub = store.subjects
    for col in ("lcmv_verification_rel_error", "wmne_verification_rel_error"):
        if not sub.empty and col in sub:                # the store's operator vs MNE's own inverse on real data
            err = pd.to_numeric(sub[col], errors="coerce")
            for k in sub.loc[err > VERIFICATION_TOL, "key"]:
                rows.append({"key": k, "check": f"source operator differs from MNE ({col} > {VERIFICATION_TOL})",
                             "detail": str(float(err[sub["key"] == k].iloc[0]))})
    if not store.subjects.empty and "n_windows_kept" in store.subjects:
        col = "n_tenfold" if (store.manifest.get("mode") == "A_xu_tenfold") else "n_windows_kept"
        n = w.groupby("key").size()
        sub = store.subjects.set_index("key")
        for k, cnt in n.items():
            if k in sub.index and col in sub and int(sub.loc[k, col]) != int(cnt):
                rows.append({"key": k, "check": f"window count differs from subjects.csv {col}",
                             "detail": f"{cnt} vs {sub.loc[k, col]}"})
    return pd.DataFrame(rows, columns=["key", "check", "detail"])


def modified_z(values: pd.DataFrame, reference: list[str]) -> pd.DataFrame:
    """Modified z-scores of every participant relative to the ``reference`` participants (median / MAD fitted on
    them only). A feature with zero reference MAD uses 1.253314 x mean absolute deviation; if that is zero too the
    feature is constant in the reference and gets z = 0 where equal to the median, else +-inf."""
    ref = values.loc[[k for k in reference if k in values.index]]
    med = ref.median()
    mad = (ref - med).abs().median()
    meanad = (ref - med).abs().mean() * 1.253314 / 0.6745
    scale = mad.where(mad > 0, meanad)
    z = 0.6745 * (values - med) / scale.replace(0, np.nan)
    const = scale.fillna(0) == 0
    for c in values.columns[const.to_numpy()]:
        z[c] = np.where(values[c] == med[c], 0.0, np.sign(values[c] - med[c]) * np.inf)
    return z


def robust_flags(values: pd.DataFrame, reference: list[str], z_threshold: float = Z_THRESHOLD,
                 n_components: int = 5, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Category B flags. Returns (per-participant table, long table of the flagged features).

    Univariate: |modified z| > ``z_threshold`` for any summary. Multivariate: robust distance of the robustly
    standardized summaries projected on the first ``n_components`` principal components of the reference
    participants, with an MCD covariance fitted on the reference participants; flagged above the chi-square
    ``MCD_QUANTILE`` quantile. Also returns the 2-D PCA coordinates for the exploratory plot."""
    from scipy import stats
    from sklearn.covariance import MinCovDet
    from sklearn.decomposition import PCA

    values = values.astype(float)
    z = modified_z(values, reference)
    zf = z.replace([np.inf, -np.inf], np.nan)
    long = (z.abs() > z_threshold).stack()
    long = long[long].reset_index()
    long.columns = ["key", "feature", "flag"]
    long["modified_z"] = [z.loc[k, f] for k, f in zip(long["key"], long["feature"])]
    long["value"] = [values.loc[k, f] for k, f in zip(long["key"], long["feature"])]
    long = long.drop(columns="flag")
    usable = [c for c in zf.columns if zf.loc[[k for k in reference if k in zf.index], c].notna().all()
              and np.isfinite(zf[c]).all()]
    table = pd.DataFrame(index=values.index)
    table["n_features_flagged"] = (z.abs() > z_threshold).sum(axis=1)
    table["max_abs_modified_z"] = zf.abs().max(axis=1)
    ref = [k for k in reference if k in values.index]
    if len(usable) >= 2 and len(ref) > n_components + 2:
        X = zf[usable].clip(-50, 50)
        pca = PCA(n_components=min(n_components, len(usable)), random_state=seed).fit(X.loc[ref])
        P = pca.transform(X)
        mcd = MinCovDet(random_state=seed).fit(P[[values.index.get_loc(k) for k in ref]])
        d2 = mcd.mahalanobis(P)
        cut = stats.chi2.ppf(MCD_QUANTILE, P.shape[1])
        table["robust_distance2"] = d2
        table["multivariate_outlier"] = d2 > cut
        table["pc1"], table["pc2"] = P[:, 0], P[:, 1] if P.shape[1] > 1 else 0.0
        table.attrs["pca_explained"] = pca.explained_variance_ratio_.tolist()
        table.attrs["mcd_cutoff"] = float(cut)
    else:
        table["robust_distance2"], table["multivariate_outlier"] = np.nan, False
    table["category_B"] = (table["n_features_flagged"] > 0) | table["multivariate_outlier"]
    return table, long.sort_values(["key", "feature"]).reset_index(drop=True)
