"""Template-anatomy source reconstruction collapsed to a sensor -> ROI linear operator.

The MNE inverse (with ``pick_ori='normal'``) and ``mean_flip`` label extraction are
both linear in the sensor data, so their composition is one (n_rois, n_channels)
matrix W. Because D1 and D2 share the same channel set, montage, reference and
template, the *same* W is used for every recording of both datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def fetch_fsaverage(subjects_dir: str | Path) -> Path:
    import mne

    Path(subjects_dir).mkdir(parents=True, exist_ok=True)
    return Path(mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose="error"))


def load_labels(subjects_dir: str | Path, parc: str) -> list:
    """Cortical labels of the parcellation (``unknown``/medial wall excluded), sorted by name."""
    import mne

    labels = mne.read_labels_from_annot("fsaverage", parc=parc, subjects_dir=subjects_dir, verbose="error")
    labels = [lab for lab in labels if not lab.name.startswith(("unknown", "corpuscallosum"))]
    return sorted(labels, key=lambda lab: lab.name)


def build_roi_operator(info, cfg: dict, subjects_dir: str | Path) -> dict:
    """Forward model, inverse operator and ROI extraction for ``info``.

    ``info`` must carry the average-reference projection and the montage.
    Returns ``{'W', 'roi_names', 'ch_names', 'n_sources', 'fwd_ch_names'}``.
    """
    import mne
    from mne.minimum_norm import apply_inverse, make_inverse_operator

    scfg = cfg["source"]
    fs_dir = fetch_fsaverage(subjects_dir)
    src = mne.read_source_spaces(fs_dir / "bem" / f"fsaverage-{scfg['spacing'].replace('ico', 'ico-')}-src.fif",
                                 verbose="error")
    bem = fs_dir / "bem" / scfg["bem"]
    fwd = mne.make_forward_solution(info, trans="fsaverage", src=src, bem=str(bem), eeg=True, meg=False,
                                    mindist=scfg["mindist_mm"], verbose="error")
    if scfg["noise_cov"] != "ad_hoc":
        raise ValueError(f"unsupported noise_cov {scfg['noise_cov']!r}")
    cov = mne.make_ad_hoc_cov(info, verbose="error")
    inv = make_inverse_operator(info, fwd, cov, loose=scfg["loose"], depth=scfg["depth"], verbose="error")

    # Push an identity "recording" through the inverse: column j = response to channel j.
    n_ch = len(info["ch_names"])
    probe = mne.EvokedArray(np.eye(n_ch), info, tmin=0.0, nave=1, verbose="error")
    stc = apply_inverse(probe, inv, lambda2=scfg["lambda2"], method=scfg["method"],
                        pick_ori=scfg["pick_ori"], verbose="error")
    labels = load_labels(subjects_dir, scfg["parcellation"])
    if scfg["parcellation"] == "aparc" and len(labels) != 68:
        raise RuntimeError(f"expected 68 Desikan-Killiany ROIs, got {len(labels)}: {[lab.name for lab in labels]}")
    W = mne.extract_label_time_course(stc, labels, inv["src"], mode=scfg["roi_mode"], verbose="error")
    return {
        "W": np.asarray(W),
        "roi_names": [lab.name for lab in labels],
        "roi_coords": roi_centroids(labels, subjects_dir),
        "ch_names": list(info["ch_names"]),
        "n_sources": int(sum(s["nuse"] for s in inv["src"])),
        "fwd_ch_names": list(fwd["info"]["ch_names"]),
    }


def roi_centroids(labels: list, subjects_dir: str | Path) -> np.ndarray:
    """MNI coordinates (mm) of each label's surface centre of mass on fsaverage, shape (n_rois, 3).

    Used only for visualization (cortical network plots)."""
    import mne

    coords = []
    for lab in labels:
        vertex = lab.center_of_mass(subject="fsaverage", subjects_dir=subjects_dir)
        hemi = 0 if lab.hemi == "lh" else 1
        coords.append(mne.vertex_to_mni(vertex, hemi, "fsaverage", subjects_dir=subjects_dir))
    return np.asarray(coords, dtype=float).reshape(len(labels), 3)


def save_operator(op: dict, path: str | Path) -> None:
    np.savez(path, W=op["W"], roi_names=np.array(op["roi_names"]), ch_names=np.array(op["ch_names"]),
             fwd_ch_names=np.array(op["fwd_ch_names"]), n_sources=op["n_sources"])


def load_operator(path: str | Path) -> dict:
    d = np.load(path)
    return {"W": d["W"], "roi_names": d["roi_names"].tolist(), "ch_names": d["ch_names"].tolist(),
            "fwd_ch_names": d["fwd_ch_names"].tolist(), "n_sources": int(d["n_sources"])}
