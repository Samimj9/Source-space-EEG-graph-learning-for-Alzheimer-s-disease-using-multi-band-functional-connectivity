"""Electrode coordinates stored in EEGLAB ``.set`` files (Dataset 3; methodology log entry 46).

The D3 forward and inverse models use the coordinates supplied with each recording (``EEG.chanlocs``), not a
template chosen by this pipeline. Coordinates are used only when their coordinate frame can be established;
otherwise ``CoordinateError`` stops the subject (there is no silent fallback to a template).

* **Axes and units.** EEGLAB stores X (towards the nose when ``chaninfo.nosedir`` is ``+X``), Y (left), Z (up).
  RAS = (-Y, X, Z). Units are taken from the configuration (``channels.coordinates.units``) and checked against
  the head radius.
* **Frame.** EEGLAB chanlocs carry no frame label. The frame is established from the provider's own record of
  where the positions came from: ``chaninfo.filename`` (or a ``pop_chanedit(..., 'lookup', <file>)`` line in
  ``EEG.history``) must name a file listed in ``channels.coordinates.accepted_lookups``. For ds004504 this is
  EEGLAB DIPFIT's ``standard_BEM/elec/standard_1005.elc``: template positions on the Colin27/MNI head, the same
  positions as MNE's ``colin27_1005`` montage. As MNE does for those templates, the positions are placed in the
  ``mri`` frame with the fsaverage MNI fiducials, so ``trans='fsaverage'`` maps them onto the fsaverage head.
* **Provenance.** Each geometry gets a hash (``geometry_hash``). Operators are built per distinct geometry, and
  the subject record stores the coordinate source, frame, lookup file, hash and the distance to the matching MNE
  template (0 mm means the stored positions are that template, i.e. not individual measurements).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class CoordinateError(ValueError):
    """Electrode coordinates missing, invalid, or in a frame that cannot be established."""


def enabled(cfg: dict) -> bool:
    return (cfg["channels"].get("coordinates") or {}).get("source") == "eeglab_chanlocs"


def _f(x) -> float:
    x = np.squeeze(np.asarray(x, dtype=object))
    try:
        return float(x) if np.size(x) == 1 else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _text(x) -> str:
    return x if isinstance(x, str) else ""


def read_chanlocs(path: str | Path) -> dict:
    """Labels, XYZ (EEGLAB axes, file units), chaninfo, fiducials and coordinate lookups of one ``.set``
    file. Reads only the small variables (never the signal)."""
    import scipy.io as sio

    path = Path(path)
    try:
        names = {v[0] for v in sio.whosmat(path)}
    except (NotImplementedError, ValueError) as exc:          # MAT v7.3 (HDF5)
        raise CoordinateError(f"{path.name}: cannot read EEGLAB variables ({exc})") from exc
    if "EEG" in names:                                         # struct saved as one variable
        eeg = sio.loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["EEG"])["EEG"]
        get = lambda k: getattr(eeg, k, None)                  # noqa: E731
    else:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False,
                        variable_names=["chanlocs", "chaninfo", "history"])
        get = m.get
    cl = get("chanlocs")
    if cl is None or not np.size(cl):
        raise CoordinateError(f"{path.name}: no chanlocs")
    cl = np.atleast_1d(cl)
    ci = get("chaninfo")
    nod = np.atleast_1d(getattr(ci, "nodatchans", [])) if ci is not None else []
    fids = {str(getattr(n, "description", getattr(n, "labels", ""))): [_f(getattr(n, k, np.nan)) for k in "XYZ"]
            for n in nod if hasattr(n, "_fieldnames") and str(getattr(n, "type", "")).upper() == "FID"}
    h = get("history")
    history = h if isinstance(h, str) else ""                  # an empty MATLAB string loads as an empty array
    lookups = [ln.split("'lookup'", 1)[1].split("'")[1] for ln in history.splitlines()
               if "'lookup'" in ln and ln.split("'lookup'", 1)[1].count("'") >= 2]
    return {"path": str(path), "labels": [str(c.labels).strip() for c in cl],
            "xyz": np.array([[_f(getattr(c, k, np.nan)) for k in "XYZ"] for c in cl], dtype=float),
            "chaninfo_filename": _text(getattr(ci, "filename", "")) if ci is not None else "",
            "nosedir": _text(getattr(ci, "nosedir", "")) if ci is not None else "",
            "fiducials": fids, "history_lookups": lookups}


def _norm(p: str) -> str:
    return p.replace("\\\\", "/").replace("\\", "/").strip().lower()


def establish(loc: dict, channels: list[str], cfg: dict) -> dict:
    """Validate the coordinates of ``channels`` and return the geometry (RAS metres in the configured frame).

    Raises ``CoordinateError`` when a channel is missing, a coordinate is not finite, the head radius does not
    fit the configured units, the orientation is not nose = +X, left electrodes are not on the left, or the
    coordinate frame cannot be established from the provider's lookup record."""
    cc = cfg["channels"]["coordinates"]
    from .eeg import channel_mapping

    mapping, _ = channel_mapping(loc["labels"], channels)
    inv = {v: k for k, v in mapping.items()}
    missing = [c for c in channels if c not in inv]
    if missing:
        raise CoordinateError(f"{Path(loc['path']).name}: no coordinates for {missing}")
    xyz = np.array([loc["xyz"][loc["labels"].index(inv[c])] for c in channels])
    if not np.isfinite(xyz).all():
        bad = [c for c, p in zip(channels, xyz) if not np.isfinite(p).all()]
        raise CoordinateError(f"{Path(loc['path']).name}: missing or non-finite coordinates for {bad}")
    if loc["nosedir"].upper() != "+X":
        raise CoordinateError(f"{Path(loc['path']).name}: chaninfo.nosedir {loc['nosedir']!r}; only '+X' is "
                              "supported (EEGLAB default)")
    scale = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}[cc["units"]]
    ras = np.c_[-xyz[:, 1], xyz[:, 0], xyz[:, 2]] * scale
    radius = float(np.median(np.linalg.norm(ras - ras.mean(axis=0), axis=1)))
    lo, hi = cc.get("head_radius_m", [0.06, 0.13])
    if not lo <= radius <= hi:
        raise CoordinateError(f"{Path(loc['path']).name}: median electrode radius {radius * 1000:.1f} mm with units "
                              f"{cc['units']!r} is outside [{lo * 1000:.0f}, {hi * 1000:.0f}] mm")
    left = [i for i, c in enumerate(channels) if c[-1] in "13579"]
    right = [i for i, c in enumerate(channels) if c[-1] in "2468"]
    if (ras[left, 0] >= 0).any() or (ras[right, 0] <= 0).any():
        raise CoordinateError(f"{Path(loc['path']).name}: odd-numbered (left) / even-numbered (right) electrodes "
                              "are not on the expected sides; orientation cannot be trusted")
    sources = [s for s in [loc["chaninfo_filename"], *loc["history_lookups"]] if s]
    accepted = [_norm(a) for a in cc["accepted_lookups"]]
    hit = next((s for s in sources if any(_norm(s).endswith(a) for a in accepted)), None)
    if hit is None:
        raise CoordinateError(f"{Path(loc['path']).name}: coordinate frame cannot be established: the positions "
                              f"come from {sources or 'an unrecorded source'}, not from one of "
                              f"{cc['accepted_lookups']} (frame {cc['frame']!r}); no template is substituted")
    geom = {"source": cc["source"], "frame": cc["frame"], "lookup": _norm(hit).rsplit("/", 1)[-1],
            "channels": list(channels), "pos_m": np.round(ras, 7).tolist()}
    geom["hash"] = geometry_hash(geom)
    geom["template_deviation_mm"] = template_deviation_mm(ras, channels, cc.get("compare_template"))
    geom["file_fiducials"] = sorted(loc["fiducials"])
    return geom


def geometry_hash(geom: dict) -> str:
    blob = json.dumps({k: geom[k] for k in ("source", "frame", "channels", "pos_m")}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def template_deviation_mm(ras: np.ndarray, channels: list[str], template: str | None) -> float | None:
    """Largest distance (mm) between the stored positions and the same channels of an MNE template montage
    read in its own frame (``colin27_1005``: the template the provider's lookup file contains)."""
    if not template:
        return None
    import mne

    pos = mne.channels.make_standard_montage(template).get_positions()["ch_pos"]
    lower = {k.lower(): v for k, v in pos.items()}
    if not all(c.lower() in lower for c in channels):
        return None
    return float(np.max(np.linalg.norm(ras - np.array([lower[c.lower()] for c in channels]), axis=1)) * 1000)


def montage(geom: dict):
    """DigMontage of a geometry: positions in the configured frame with the fsaverage MNI fiducials (``mri``),
    so ``trans='fsaverage'`` places them on the fsaverage head (MNE's handling of its colin27 templates)."""
    import mne

    if geom["frame"] != "mri":
        raise CoordinateError(f"frame {geom['frame']!r} not supported (only 'mri' = MNI/fsaverage MRI space)")
    lpa, nasion, rpa = (f["r"].copy() for f in mne.coreg.get_mni_fiducials("fsaverage"))
    return mne.channels.make_dig_montage(ch_pos=dict(zip(geom["channels"], np.asarray(geom["pos_m"]))),
                                         nasion=nasion, lpa=lpa, rpa=rpa, coord_frame="mri")


def geometry_of(path: str | Path, channels: list[str], cfg: dict) -> dict:
    return establish(read_chanlocs(path), channels, cfg)


def provenance(geom: dict) -> dict:
    """Flat record for subjects.csv / manifests."""
    return {"coord_source": geom["source"], "coord_frame": geom["frame"], "coord_lookup": geom["lookup"],
            "montage_hash": geom["hash"], "coord_template_deviation_mm": geom["template_deviation_mm"],
            "coord_file_fiducials": json.dumps(geom["file_fiducials"])}
