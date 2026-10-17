"""wMNE ROI operator for the matched wMNE / Xu-protocol suite (``D3_WMNE_XU``; methodology log entry 55).

The ONLY intended difference from the LCMV suite is the inverse solution. Everything else (recordings, segments,
five-minute selection, windows, rejection, bands, PSD / DE / PLV / |ciPLV|, ROIs, stores, split) runs through the
same ``lcmvxu`` code. The inverse settings are those of the historical wMNE pipeline (``configs/pipeline_d3.yaml``
``source``, ``source.inverse.build_roi_operator``): depth-weighted minimum norm (``method: MNE``, ``depth: 0.8``),
loose orientation 0.2, ``lambda2 = 1/9`` (SNR 3), ad-hoc diagonal noise covariance, signed normal component
(``pick_ori: normal``), ``mean_flip`` over the 68 Desikan-Killiany labels.

Shared with the LCMV suite: the fsaverage ico-5 source space, the 3-layer BEM, ``mindist`` and the electrode
positions stored in the recordings (one forward model per geometry; ``beamformer.forward``). The wMNE inverse uses
the free-orientation forward (loose 0.2, as before); LCMV uses its fixed cortical-normal version. This is part of
each method's definition, not a separate choice.

The operator does not depend on any recording (ad-hoc noise covariance, no data covariance), so it is identical for
every participant with the same electrode geometry: one (68, 19) matrix W, ROI signals = W @ x of the
average-referenced data. ``verify`` compares W @ x with MNE's own ``apply_inverse_raw`` + label extraction on real
data of each participant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _inverse(info, cfg: dict, subjects_dir: str | Path, cache_dir: str | Path, geometry_hash: str):
    import mne
    from mne.minimum_norm import make_inverse_operator

    from .beamformer import forward

    w = cfg["source"]["wmne"]
    if w["noise_cov"] != "ad_hoc":
        raise ValueError(f"unsupported noise_cov {w['noise_cov']!r}")
    fwd = forward(info, cfg, subjects_dir, cache_dir, geometry_hash, fixed=False)
    cov = mne.make_ad_hoc_cov(info, verbose="error")
    return make_inverse_operator(info, fwd, cov, loose=w["loose"], depth=w["depth"], verbose="error")


def roi_operator(info, cfg: dict, subjects_dir: str | Path, cache_dir: str | Path, geometry_hash: str,
                 labels: list) -> dict:
    """(68, 19) wMNE ROI operator W (identity probe through ``apply_inverse`` and ``mean_flip``) and the inverse."""
    import mne
    from mne.minimum_norm import apply_inverse

    w = cfg["source"]["wmne"]
    inv = _inverse(info, cfg, subjects_dir, cache_dir, geometry_hash)
    probe = mne.EvokedArray(np.eye(len(info["ch_names"])), info, tmin=0.0, nave=1, verbose="error")
    stc = apply_inverse(probe, inv, lambda2=w["lambda2"], method=w["method"], pick_ori=w["pick_ori"],
                        verbose="error")
    W = np.asarray(mne.extract_label_time_course(stc, labels, inv["src"], mode=cfg["source"]["roi_mode"],
                                                 verbose="error"))
    if W.shape != (len(labels), len(info["ch_names"])) or not np.isfinite(W).all():
        raise RuntimeError(f"invalid wMNE ROI operator {W.shape}")
    return {"W": W, "inv": inv, "n_sources": int(sum(s["nuse"] for s in inv["src"]))}


def verify(W: np.ndarray, info, inv, labels: list, x: np.ndarray, cfg: dict) -> float:
    """Max relative difference between W @ x and MNE's apply_inverse_raw + label extraction on real data x."""
    import mne
    from mne.minimum_norm import apply_inverse_raw

    w = cfg["source"]["wmne"]
    raw = mne.io.RawArray(x, info, verbose="error")
    stc = apply_inverse_raw(raw, inv, lambda2=w["lambda2"], method=w["method"], pick_ori=w["pick_ori"],
                            verbose="error")
    ref = np.asarray(mne.extract_label_time_course(stc, labels, inv["src"], mode=cfg["source"]["roi_mode"],
                                                   verbose="error"))
    return float(np.abs(W @ x - ref).max() / max(np.abs(ref).max(), 1e-30))
