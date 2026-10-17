"""Time-domain LCMV beamformer collapsed to a per-participant 68 x 19 ROI operator (methodology log entry 50).

* **Forward model** (built once per electrode geometry, cached): fsaverage ico-5 surface source space, the
  3-layer fsaverage BEM, the 19 electrode positions stored in the .set files (entry 46), converted to fixed
  cortical-normal orientation (``surf_ori=True, force_fixed=True``). With fixed orientation every source has
  one scalar filter, so the ROI sign-flip by the cortical normal (``mean_flip``) is well defined, as in the
  existing wMNE pipeline (``pick_ori='normal'``).
* **Data covariance** (per participant, never labels or other participants): the participant's own accepted,
  average-referenced, broadband (0.5-45 Hz) samples of the selected five minutes. The average reference
  removes one dimension and the provider's ICA removed further components, so the covariance rank is
  estimated from the data (``mne.compute_rank``) and passed to ``make_lcmv``, which inverts it in the reduced
  space. Tikhonov loading ``reg`` = 5 % of the mean eigenvalue.
* **Filters**: ``mne.beamformer.make_lcmv`` with ``weight_norm='unit-noise-gain-invariant'`` (rotation- and
  rank-robust noise normalization), an ad-hoc diagonal noise covariance (whitening of one channel type only).
  One broadband filter per participant is used for all five bands, so band powers of one ROI stay comparable.
* **ROI operator**: an identity "recording" (one column per channel) is pushed through ``apply_lcmv`` and then
  through ``mne.extract_label_time_course(mode='mean_flip')``; both are linear, so the composition is the
  (68, 19) matrix W with ROI signals = W @ x. ``verify`` compares W @ x with MNE's own ``apply_lcmv_raw`` +
  label extraction on real data of the participant.

Limitation (stated in docs/lcmv_xu.md): 19 electrodes (rank <= 18) cannot resolve 68 independent cortical
time courses; the ROI signals are spatially smoothed, partly mixed estimates (leakage), and |ciPLV| is the
leakage-robust connectivity measure. The beamformer also suppresses perfectly correlated sources.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def sensor_info(cfg: dict, geometry: dict):
    """Info of the 19 channels at 250 Hz: stored electrode positions + applied average-reference projection."""
    from ..pipeline import canonical_info

    return canonical_info(cfg, geometry)


def forward(info, cfg: dict, subjects_dir: str | Path, cache_dir: str | Path, geometry_hash: str,
            fixed: bool = True):
    """fsaverage forward model for this geometry (cached as a .fif file): fixed cortical-normal orientation
    (LCMV), or the free-orientation solution itself (``fixed=False``; the wMNE inverse, ``lcmvxu.wmne``)."""
    import mne

    from ..source.inverse import fetch_fsaverage

    scfg = cfg["source"]
    path = Path(cache_dir) / f"lx_forward_{geometry_hash}_{scfg['spacing']}-fwd.fif"
    if path.exists():
        fwd = mne.read_forward_solution(path, verbose="error")
    else:
        fs_dir = fetch_fsaverage(subjects_dir)
        src = mne.read_source_spaces(fs_dir / "bem" / f"fsaverage-{scfg['spacing'].replace('ico', 'ico-')}-src.fif",
                                     verbose="error")
        fwd = mne.make_forward_solution(info, trans="fsaverage", src=src, bem=str(fs_dir / "bem" / scfg["bem"]),
                                        eeg=True, meg=False, mindist=scfg["mindist_mm"], verbose="error")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name.replace("-fwd.fif", ".tmp-fwd.fif"))
        mne.write_forward_solution(tmp, fwd, overwrite=True, verbose="error")
        tmp.replace(path)
    if fixed:
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose="error")
    if list(fwd["info"]["ch_names"]) != list(info["ch_names"]):
        raise RuntimeError("forward channel order differs from the data")
    return fwd


def data_covariance(info, samples: np.ndarray):
    """Sample covariance (n_ch, n_ch) of the given average-referenced samples, as an mne.Covariance."""
    import mne

    x = samples - samples.mean(axis=1, keepdims=True)
    c = x @ x.T / max(x.shape[1] - 1, 1)
    return mne.Covariance(c, list(info["ch_names"]), [], info["projs"], x.shape[1], verbose="error")


def roi_operator(info, fwd, data_cov, labels: list, cfg: dict) -> dict:
    """LCMV filters -> (68, 19) ROI operator. Returns W, rank and the filters' settings."""
    import mne
    from mne.beamformer import apply_lcmv, make_lcmv

    lc = cfg["source"]["lcmv"]
    rank = mne.compute_rank(data_cov, rank=None, info=info, verbose="error")
    noise = mne.make_ad_hoc_cov(info, verbose="error")
    filters = make_lcmv(info, fwd, data_cov, reg=lc["reg"], noise_cov=noise, pick_ori=None,
                        weight_norm=lc["weight_norm"], rank=rank, reduce_rank=False, depth=None, verbose="error")
    probe = mne.EvokedArray(np.eye(len(info["ch_names"])), info, tmin=0.0, nave=1, verbose="error")
    stc = apply_lcmv(probe, filters, verbose="error")
    W = np.asarray(mne.extract_label_time_course(stc, labels, fwd["src"], mode=cfg["source"]["roi_mode"],
                                                 verbose="error"))
    if not np.isfinite(W).all():
        raise RuntimeError("non-finite LCMV ROI operator")
    return {"W": W, "rank": int(rank["eeg"]), "filters": filters}


def verify(W: np.ndarray, info, filters, labels: list, src, x: np.ndarray, cfg: dict) -> float:
    """Max relative difference between W @ x and MNE's apply_lcmv_raw + label extraction on real data x."""
    import mne
    from mne.beamformer import apply_lcmv_raw

    raw = mne.io.RawArray(x, info, verbose="error")
    stc = apply_lcmv_raw(raw, filters, verbose="error")
    ref = np.asarray(mne.extract_label_time_course(stc, labels, src, mode=cfg["source"]["roi_mode"],
                                                   verbose="error"))
    return float(np.abs(W @ x - ref).max() / max(np.abs(ref).max(), 1e-30))
