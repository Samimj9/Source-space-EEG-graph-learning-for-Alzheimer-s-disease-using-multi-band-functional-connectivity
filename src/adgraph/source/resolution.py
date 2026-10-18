"""Spatial resolution of 19-channel EEG source estimates per atlas (docs/final/parcellation.md); no EEG, no labels.

The question: how many cortical regions can 19 average-referenced electrodes distinguish, and how strongly do the
ROI signals of each atlas (68 Desikan-Killiany labels, 34 / 18 / 12 merged regions; ``source.atlas``) leak into one
another? Everything is derived from the forward model and the two inverse operators of the stores, exactly as the
stores build them (an identity "recording" through the MNE inverse and ``mean_flip`` label extraction):

* **spatial rank** of the average-referenced normal-orientation leadfield G (19 x n_sources): numerical rank (<= 18
  after the average reference), the entropy effective rank (Roy & Vetterli 2007), the number of singular values
  holding 95 / 99 % of the leadfield energy, and the number above the regularization level of the wMNE inverse;
* **ROI operator rank**: rank(W) <= 18 for the (R x 19) ROI operator, so with R > 18 at least R - 18 ROI signals are
  exact linear combinations of the others;
* **region resolution matrix** R = W G_regions (R x R): column j is the response of all ROI estimates to unit
  activity of region j (sources of region j with the sign pattern of ``mean_flip``, every other source silent);
  per ROI the cross-talk (leakage) fraction sum_{j != i} R_ij^2 / sum_j R_ij^2 under independent equal-variance
  region activity, and whether its own region dominates it (argmax_j |R_ij| = i);
* **leakage correlation**: correlations of the ROI estimates when every region is active independently with unit
  variance plus sensor noise at amplitude SNR 3 (the wMNE regularization 1/9): the zero-lag correlation floor that
  volume conduction alone puts into PLV-type connectivity (|ciPLV| discards the zero-lag part), and the
  participation ratio (effective number of independent ROI signals) of that correlation matrix.

The LCMV filters depend on the data covariance; here they use the model covariance of the same scenario
(independent unit-variance cortical sources + noise), so they describe the beamformer's resolution for spatially
white activity, not for any participant. The analysis uses the forward model it is given: the fsaverage 3-layer BEM
of the stores (Colab, ``scripts/final.py parcellation``), or a three-shell sphere fitted to the electrodes when the
BEM is not available (an approximation, recorded in the output).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import atlas as atl

CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz",
            "Cz", "Pz"]
SNR = 3.0                           # amplitude SNR; lambda2 = 1 / SNR^2 = the wMNE regularization of the stores


def sensor_info(channels=CHANNELS, montage: str = "colin27_1005", sfreq: float = 250.0):
    """19-channel Info with template positions and the applied average-reference projection (as the stores)."""
    import mne

    info = mne.create_info(list(channels), sfreq, "eeg")
    info.set_montage(mne.channels.make_standard_montage(montage))
    raw = mne.io.RawArray(np.zeros((len(channels), 1)), info, verbose="error")
    raw.set_eeg_reference("average", projection=True, verbose="error")
    raw.apply_proj(verbose="error")
    return raw.info


def forward_model(info, subjects_dir: str | Path, bem: str | None = None, spacing: str = "ico5",
                  mindist_mm: float = 5.0):
    """Free-orientation forward on the fsaverage source space: the given BEM solution file (the stores'
    ``fsaverage-5120-5120-5120-bem-sol.fif``) or, with ``bem=None``, a three-shell sphere fitted to the electrodes.
    Returns (forward, description)."""
    import mne

    fs = Path(subjects_dir) / "fsaverage"
    src = mne.read_source_spaces(fs / "bem" / f"fsaverage-{spacing.replace('ico', 'ico-')}-src.fif", verbose="error")
    trans = fs / "bem" / "fsaverage-trans.fif"
    if bem is None:
        # centre fitted to the electrodes; radius such that every cortical source lies inside the inner-skull shell
        # (relative radius 0.90) with 5 mm to spare, so no region loses its sources
        fit = mne.make_sphere_model("auto", "auto", info, verbose="error")
        r0 = np.asarray(fit["r0"])
        t = mne.read_trans(trans) if trans.exists() else mne.transforms._get_trans("fsaverage")[0]
        t = mne.transforms._ensure_trans(t, "mri", "head")                 # sources are in MRI coordinates
        # MNE tests "inside the sphere" in MRI coordinates against a head-frame centre; express the sources in head
        # coordinates (positions and normals) and use the identity transform so both frames coincide
        src = src.copy()
        for s in src:
            s["rr"] = mne.transforms.apply_trans(t, s["rr"])
            s["nn"] = mne.transforms.apply_trans(t, s["nn"], move=False)
        trans = mne.transforms.Transform("head", "mri")
        pts = np.concatenate([s["rr"][s["vertno"]] for s in src])
        radius = (np.linalg.norm(pts - r0, axis=1).max() + mindist_mm / 1000) / 0.90
        model = mne.make_sphere_model(r0, radius, info, verbose="error")
        desc = (f"three-shell sphere (centre fitted to the electrodes, radius {radius * 1000:.0f} mm enclosing the "
                "cortex; an approximation, the stores use the fsaverage BEM)")
    else:
        model = str(bem)
        desc = (f"approximate 3-layer fsaverage BEM {Path(bem).name} (realistic scalp and inner skull, derived outer "
                "skull; the stores use the fsaverage 5120 BEM)" if "approx" in Path(bem).name
                else f"fsaverage BEM {Path(bem).name} (as the stores)")
    if isinstance(trans, Path):
        trans = str(trans) if trans.exists() else "fsaverage"
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=model,
                                    eeg=True, meg=False, mindist=mindist_mm, verbose="error")
    return fwd, desc


def approximate_bem(path: str | Path, skull_mm: float = 5.0) -> Path:
    """A 3-layer fsaverage BEM solution from the two realistic surfaces shipped with MNE (``mne/data/fsaverage``:
    scalp ``fsaverage-head.fif``, inner skull ``fsaverage-inner_skull-bem.fif``, ico-4), with the outer skull derived
    by a radial expansion of the inner skull of ``skull_mm`` (at most half the local inner-skull-to-scalp gap, so the
    layers nest); conductivities 0.3 / 0.006 / 0.3 S/m. For environments without the stores' fsaverage BEM; the
    result is labelled as an approximation."""
    import os

    import mne
    from mne.bem import _ico_downsample, _surfaces_to_bem
    from mne.io.constants import FIFF
    from scipy.spatial import cKDTree

    path = Path(path)
    if path.exists():
        return path
    d = Path(os.path.dirname(mne.__file__)) / "data" / "fsaverage"
    inner = _ico_downsample(mne.read_bem_surfaces(d / "fsaverage-inner_skull-bem.fif", verbose="error")[0], 4)
    head = mne.read_bem_surfaces(d / "fsaverage-head.fif", verbose="error")[0]
    c = inner["rr"].mean(0)
    v = inner["rr"] - c
    gap = cKDTree(head["rr"]).query(inner["rr"])[0]
    disp = np.minimum(skull_mm / 1000, 0.5 * gap)[:, None]
    outer = c + v * (1 + disp / np.linalg.norm(v, axis=1, keepdims=True))
    surfs = [{"rr": inner["rr"].copy(), "tris": inner["tris"]}, {"rr": outer, "tris": inner["tris"]},
             {"rr": head["rr"].copy(), "tris": head["tris"]}]
    bem = _surfaces_to_bem(surfs, [FIFF.FIFFV_BEM_SURF_ID_BRAIN, FIFF.FIFFV_BEM_SURF_ID_SKULL,
                                   FIFF.FIFFV_BEM_SURF_ID_HEAD], (0.3, 0.006, 0.3), ico=None, rescale=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    mne.write_bem_solution(path, mne.make_bem_solution(bem, verbose="error"), overwrite=True, verbose="error")
    return path


def _avg_ref(n: int) -> np.ndarray:
    return np.eye(n) - 1.0 / n


def normal_leadfield(fwd) -> np.ndarray:
    """Average-referenced leadfield along the cortical normals (n_channels x n_sources)."""
    import mne

    fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose="error")
    g = fixed["sol"]["data"]
    return _avg_ref(g.shape[0]) @ g


def spatial_rank(g: np.ndarray, snr: float = SNR) -> dict:
    s = np.linalg.svd(g, compute_uv=False)
    tol = s.max() * max(g.shape) * np.finfo(float).eps * 1e3
    e = s ** 2 / (s ** 2).sum()
    cum = np.cumsum(e)
    p = s[s > tol] / s[s > tol].sum()
    # the wMNE regularization: singular values below sqrt(lambda2) x the largest are effectively suppressed
    lam = (1.0 / snr ** 2)
    return {"n_channels": int(g.shape[0]), "numerical_rank": int((s > tol).sum()),
            "effective_rank_entropy": float(np.exp(-(p * np.log(p)).sum())),
            "n_sv_95pct_energy": int(np.searchsorted(cum, 0.95) + 1),
            "n_sv_99pct_energy": int(np.searchsorted(cum, 0.99) + 1),
            "n_sv_above_regularization": int((s / s.max() > np.sqrt(lam)).sum()),
            "singular_values_rel": (s / s.max()).round(5).tolist()}


def labels(subjects_dir: str | Path, atlas: str) -> list:
    from .inverse import load_labels

    return atl.merged_labels(load_labels(subjects_dir, "aparc"), atlas)


def _vertex_rows(src) -> dict[tuple[int, int], int]:
    out, off = {}, 0
    for h, s in enumerate(src):
        for i, v in enumerate(s["vertno"]):
            out[(h, int(v))] = off + i
        off += len(s["vertno"])
    return out


def region_leadfield(g: np.ndarray, labs: list, src) -> np.ndarray:
    """(n_channels x R): field of unit activity of each region, sources with the sign pattern of ``mean_flip``
    (so that mean_flip extraction of the true sources returns +1)."""
    import mne

    rows = _vertex_rows(src)
    out = np.zeros((g.shape[0], len(labs)))
    for j, lab in enumerate(labs):
        h = 0 if lab.hemi == "lh" else 1
        verts = np.intersect1d(lab.vertices, src[h]["vertno"])
        flip = mne.label_sign_flip(lab, src)
        idx = [rows[(h, int(v))] for v in verts]
        out[:, j] = g[:, idx] @ flip / len(idx)
    return out


def _probe_stc(info, apply):
    import mne

    probe = mne.EvokedArray(np.eye(len(info["ch_names"])), info, tmin=0.0, nave=1, verbose="error")
    return apply(probe)


def roi_operators(info, fwd, labs: list, cfg: dict, g: np.ndarray, snr: float = SNR) -> dict[str, np.ndarray]:
    """{inverse: (R x 19) ROI operator} for the wMNE and LCMV settings of ``cfg`` (``sourcegraphs.config.resolve``),
    built like the stores: identity probe -> inverse -> ``mean_flip``."""
    import mne
    from mne.beamformer import apply_lcmv, make_lcmv
    from mne.minimum_norm import apply_inverse, make_inverse_operator

    out = {}
    w = cfg["wmne"]["source"]["wmne"]
    inv = make_inverse_operator(info, fwd, mne.make_ad_hoc_cov(info, verbose="error"), loose=w["loose"],
                                depth=w["depth"], verbose="error")
    stc = _probe_stc(info, lambda p: apply_inverse(p, inv, lambda2=w["lambda2"], method=w["method"],
                                                   pick_ori=w["pick_ori"], verbose="error"))
    out["wmne"] = np.asarray(mne.extract_label_time_course(stc, labs, inv["src"], mode="mean_flip", verbose="error"))
    lc = cfg["lcmv"]["source"]["lcmv"]
    fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose="error")
    sig = g @ g.T
    noise = np.trace(sig) / (snr ** 2) / g.shape[0]
    c = sig + noise * _avg_ref(g.shape[0])
    cov = mne.Covariance(c, list(info["ch_names"]), [], info["projs"], 10_000, verbose="error")
    rank = mne.compute_rank(cov, rank=None, info=info, verbose="error")
    filt = make_lcmv(info, fixed, cov, reg=lc["reg"], noise_cov=mne.make_ad_hoc_cov(info, verbose="error"),
                     pick_ori=None, weight_norm=lc["weight_norm"], rank=rank, reduce_rank=False, depth=None,
                     verbose="error")
    stc = _probe_stc(info, lambda p: apply_lcmv(p, filt, verbose="error"))
    out["lcmv"] = np.asarray(mne.extract_label_time_course(stc, labs, fixed["src"], mode="mean_flip",
                                                           verbose="error"))
    return out


def resolution_metrics(W: np.ndarray, g_regions: np.ndarray, snr: float = SNR) -> dict:
    """Metrics of one ROI operator W (R x 19) against the region leadfield (19 x R)."""
    r = W.shape[0]
    R = W @ g_regions
    p2 = R ** 2
    own = np.diag(p2)
    leak = 1.0 - own / p2.sum(1)
    peak_own = (np.abs(R).argmax(1) == np.arange(r))
    sig = g_regions @ g_regions.T
    noise = np.trace(sig) / (snr ** 2) / g_regions.shape[0]
    cov = W @ (sig + noise * _avg_ref(g_regions.shape[0])) @ W.T
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    iu = np.triu_indices(r, 1)
    ac = np.abs(corr[iu])
    ev = np.clip(np.linalg.eigvalsh(corr), 0, None)
    s = np.linalg.svd(W, compute_uv=False)
    return {"n_regions": r, "operator_rank": int((s > s.max() * 1e-8).sum()),
            "redundant_signals": int(r - (s > s.max() * 1e-8).sum()),
            "crosstalk_fraction_median": float(np.median(leak)), "crosstalk_fraction_mean": float(leak.mean()),
            "crosstalk_fraction_q90": float(np.quantile(leak, 0.9)),
            "rois_dominated_by_own_region": float(peak_own.mean()),
            "leakage_corr_abs_median": float(np.median(ac)), "leakage_corr_abs_q90": float(np.quantile(ac, 0.9)),
            "pairs_leakage_corr_above_0.5": float((ac > 0.5).mean()),
            "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "participation_ratio_per_region": float(ev.sum() ** 2 / (ev ** 2).sum() / r)}


def analyse(subjects_dir: str | Path, cfg: dict, atlases=atl.ATLASES, bem: str | None = None,
            montage: str = "colin27_1005") -> tuple[pd.DataFrame, dict]:
    """Every atlas x inverse: the resolution metrics (table) and the spatial rank of the leadfield (dict).
    ``cfg``: {"lcmv": resolve('lcmv'), "wmne": resolve('wmne')}."""
    info = sensor_info(montage=montage)
    fwd, desc = forward_model(info, subjects_dir, bem)
    g = normal_leadfield(fwd)
    rank = {**spatial_rank(g), "forward_model": desc, "montage": montage, "n_sources": int(g.shape[1])}
    rows = []
    for a in atlases:
        labs = labels(subjects_dir, a)
        gr = region_leadfield(g, labs, fwd["src"])
        for inv, W in roi_operators(info, fwd, labs, cfg, g).items():
            rows.append({"atlas": a, "inverse": inv, **resolution_metrics(W, gr)})
    return pd.DataFrame(rows), rank


def recommend_atlas(table: pd.DataFrame, rule: dict) -> dict:
    """The finest atlas of ``rule['candidates']`` meeting every criterion for every inverse
    (``configs/final/selection.yaml`` ``atlas_resolution``); with the per-atlas verdicts."""
    verdicts = []
    for a in rule["candidates"]:
        t = table[table["atlas"] == a] if "atlas" in table else table.iloc[0:0]
        if t.empty:
            verdicts.append({"atlas": a, "eligible": False, "reasons": ["not analysed"]})
            continue
        reasons = []
        if rule.get("require_no_forced_redundancy", True) and (t["redundant_signals"] > 0).any():
            reasons.append(f"{int(t['redundant_signals'].max())} ROI signals are linear combinations of the others")
        if (t["crosstalk_fraction_median"] > rule["max_median_crosstalk"]).any():
            reasons.append(f"median cross-talk {t['crosstalk_fraction_median'].max():.2f} > "
                           f"{rule['max_median_crosstalk']}")
        if (t["rois_dominated_by_own_region"] < rule["min_own_region_dominance"]).any():
            reasons.append(f"own-region dominance {t['rois_dominated_by_own_region'].min():.2f} < "
                           f"{rule['min_own_region_dominance']}")
        verdicts.append({"atlas": a, "eligible": not reasons, "reasons": reasons,
                         "n_regions": int(t["n_regions"].iloc[0]),
                         "crosstalk_median_worst_inverse": float(t["crosstalk_fraction_median"].max()),
                         "own_region_dominance_worst_inverse": float(t["rois_dominated_by_own_region"].min())})
    ok = [v["atlas"] for v in verdicts if v["eligible"]]
    return {"recommended": ok[0] if ok else None, "verdicts": verdicts, "rule": rule,
            "note": "finest atlas meeting every criterion for both inverses; chosen without EEG or labels"}


# ------------------------------------------------------------------------------- stores (development participants)
def split_half_reliability(values: pd.DataFrame, key: np.ndarray) -> pd.Series:
    """Spearman-Brown corrected correlation, across participants, between the participant means of the even and
    the odd windows, per feature column (label-blind)."""
    df = values.copy()
    df["_key"] = key
    df["_half"] = df.groupby("_key").cumcount() % 2
    a = df[df["_half"] == 0].groupby("_key").mean(numeric_only=True).drop(columns="_half")
    b = df[df["_half"] == 1].groupby("_key").mean(numeric_only=True).drop(columns="_half")
    common = a.index.intersection(b.index)
    r = a.loc[common].corrwith(b.loc[common])
    return 2 * r / (1 + r)


def store_reliability(store, development_keys: list[str]) -> pd.DataFrame:
    """Per feature family and band of a store (development participants only): median split-half reliability over
    ROIs (PSD log / rel, DE, PLV and |ciPLV| node strength) and over edges (PLV, |ciPLV|)."""
    keep = store.windows["key"].isin(set(development_keys)).to_numpy()
    idx = np.flatnonzero(keep)
    key = store.windows["key"].to_numpy()[idx]
    rows = []
    for b, band in enumerate(store.bands):
        for kind in ("log", "rel", "de"):
            arr = store.arrays.get(f"psd_{kind}")
            if arr is not None:
                rel = split_half_reliability(pd.DataFrame(np.asarray(arr[idx, b], float)), key)
                rows.append({"feature": f"psd_{kind}", "band": band, "unit": "ROI", "median": float(rel.median()),
                             "q10": float(rel.quantile(0.1)), "n": int(rel.size)})
        for conn in ("plv", "ciplv"):
            a = np.asarray(store.arrays[conn][idx, b], np.float32)
            r_ = a.shape[-1]
            strength = (a.sum(-1)) / (r_ - 1)
            rel = split_half_reliability(pd.DataFrame(strength.astype(float)), key)
            rows.append({"feature": f"{conn}_strength", "band": band, "unit": "ROI", "median": float(rel.median()),
                         "q10": float(rel.quantile(0.1)), "n": int(rel.size)})
            iu = np.triu_indices(r_, 1)
            rel = split_half_reliability(pd.DataFrame(a[:, iu[0], iu[1]].astype(float)), key)
            rows.append({"feature": f"{conn}_edge", "band": band, "unit": "edge", "median": float(rel.median()),
                         "q10": float(rel.quantile(0.1)), "n": int(rel.size)})
    return pd.DataFrame(rows)


def sensor_signals(W: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Average-referenced sensor signals x recovered from the saved ROI signals roi = W x.

    W (R x C) annihilates the constant vector (average reference) and has rank C - 1, so its row space IS the
    average-referenced sensor space and x = pinv(W) roi exactly (up to the float32 storage of roi)."""
    return np.linalg.pinv(W.astype(np.float64)) @ roi.astype(np.float64)


def reconstruction_check(store_root: str | Path, keys: list[str] | None = None, margin: int = 825,
                         window: int = 2500) -> pd.DataFrame:
    """Can coarser-atlas ROI signals be rebuilt from a store's saved intermediates (no recordings)? Read-only.

    Per participant (``subjects/<key>/operator.npz`` W, ``roi_timeseries.npz`` pieces, ``windows.csv``):
    ``rank`` of W and its constant-vector response (average reference); ``rel_residual`` = |W pinv(W) roi - roi| /
    |roi| (0 up to float32: the sensor signals are exactly recoverable); and the fraction of the participant's windows
    whose filter support [start - margin, stop + margin) lies inside the saved piece (the store saved each piece
    without the analytic-filter margins, so windows within one margin of a piece end cannot be recomputed exactly).
    ``margin`` = ``sourcegraphs.recording.margin_samples`` (825 at 250 Hz), ``window`` = 10 s."""
    rows = []
    for unit in sorted(Path(store_root).glob("subjects/*")):
        k = unit.name
        if keys is not None and k not in set(keys):
            continue
        if not (unit / "operator.npz").exists() or not (unit / "roi_timeseries.npz").exists():
            rows.append({"key": k, "available": False})
            continue
        with np.load(unit / "operator.npz") as z:
            W = z["W"]
        s = np.linalg.svd(W, compute_uv=False)
        res, tot, ok, n = 0.0, 0.0, 0, 0
        win = pd.read_csv(unit / "windows.csv") if (unit / "windows.csv").exists() else pd.DataFrame()
        with np.load(unit / "roi_timeseries.npz") as z:
            for name in [f for f in z.files if not f.endswith("_start_sample")]:
                roi = z[name]
                x = sensor_signals(W, roi)
                res += float(np.linalg.norm(W @ x - roi) ** 2)
                tot += float(np.linalg.norm(roi) ** 2)
                if len(win) and "piece" in win:
                    start = int(z[f"{name}_start_sample"])
                    w = win[win["piece"] == int(name.replace("piece", ""))]
                    off = w["start_sample"].to_numpy() - start
                    ok += int(((off >= margin) & (off + window + margin <= roi.shape[1])).sum())
                    n += len(w)
        rows.append({"key": k, "available": True, "n_rois": W.shape[0], "rank": int((s > s[0] * 1e-8).sum()),
                     "constant_response": float(np.abs(W.sum(1)).max() / np.abs(W).max()),
                     "rel_residual": float(np.sqrt(res / tot)) if tot else float("nan"), "n_windows": n,
                     "windows_reconstructable": ok, "fraction_reconstructable": ok / n if n else float("nan")})
    return pd.DataFrame(rows)


def operator_ranks(store_root: str | Path, keys: list[str] | None = None) -> pd.DataFrame:
    """Rank and singular-value spread of each participant's saved ROI operator (``subjects/<key>/operator.npz``)."""
    rows = []
    for f in sorted(Path(store_root).glob("subjects/*/operator.npz")):
        k = f.parent.name
        if keys is not None and k not in set(keys):
            continue
        with np.load(f) as z:
            W = z["W"]
        s = np.linalg.svd(W, compute_uv=False)
        rows.append({"key": k, "n_rois": W.shape[0], "rank": int((s > s.max() * 1e-8).sum()),
                     "sv_ratio_10": float(s[min(9, len(s) - 1)] / s[0])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- the stores' own operators (real data)
def operator_resolution(operators: dict[str, np.ndarray], g_regions: np.ndarray) -> pd.DataFrame:
    """``resolution_metrics`` of each participant's saved ROI operator ({key: W}) against one region leadfield."""
    return pd.DataFrame([{"key": k, **resolution_metrics(W.astype(np.float64), g_regions)}
                         for k, W in sorted(operators.items())])


def store_resolution(store_root: str | Path, subjects_dir: str | Path, atlas: str,
                     keys: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Resolution of the ROI operators a store ACTUALLY used (read-only): each participant's saved W
    (``subjects/<key>/operator.npz``; LCMV from that participant's own data covariance, wMNE the data-independent
    operator of the electrode geometry) against the region leadfield of the store's OWN cached forward model
    (``<store parent>/cache/lx_forward_<montage_hash>_<spacing>-fwd.fif``: the stores' fsaverage BEM and electrode
    geometry). Returns (one row per participant, description). Remaining approximations: fsaverage template anatomy,
    template (Colin27 lookup) electrode positions, independent equal-variance region activity for the cross-talk."""
    import mne

    root = Path(store_root)
    subj = pd.read_csv(root / "subjects.csv")
    if keys is not None:
        subj = subj[subj["key"].isin(set(keys))]
    labs = labels(subjects_dir, atlas)
    by_geom, used = {}, {}
    for geom, g in subj.groupby(subj["montage_hash"].astype(str) if "montage_hash" in subj else
                                pd.Series("", index=subj.index)):
        files = sorted((root.parent / "cache").glob(f"lx_forward_{geom}_*-fwd.fif" if geom else "lx_forward_*-fwd.fif"))
        if len(files) != 1:
            raise FileNotFoundError(f"forward model of geometry {geom!r} not found uniquely in {root.parent / 'cache'}")
        fwd = mne.read_forward_solution(files[0], verbose="error")
        gl = normal_leadfield(fwd)
        gr = region_leadfield(gl, labs, fwd["src"])
        ops = {}
        for k in g["key"]:
            f = root / "subjects" / k / "operator.npz"
            if f.exists():
                with np.load(f) as z:
                    ops[k] = z["W"]
        if ops and next(iter(ops.values())).shape != (len(labs), gl.shape[0]):
            raise ValueError(f"saved operators {next(iter(ops.values())).shape} do not fit atlas {atlas} "
                             f"({len(labs)} regions x {gl.shape[0]} channels)")
        by_geom[geom] = operator_resolution(ops, gr).assign(montage_hash=geom)
        used[geom] = files[0].name
    t = pd.concat(by_geom.values(), ignore_index=True) if by_geom else pd.DataFrame()
    return t, {"store": root.name, "atlas": atlas, "forward_models": used, "n_participants": int(len(t)),
               "head_model": "the store's cached fsaverage BEM forward (as used to build the store)",
               "approximations": ["fsaverage template anatomy", "template (Colin27 lookup) electrode positions",
                                  "cross-talk under independent equal-variance region activity"]}
