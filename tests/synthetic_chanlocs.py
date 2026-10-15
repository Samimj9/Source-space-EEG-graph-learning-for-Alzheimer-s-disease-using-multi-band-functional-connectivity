"""Provider-style EEGLAB chanlocs for synthetic Dataset 3 tests (methodology log entry 46).

ds004504 stores, in every .set file, the Colin27 template positions that EEGLAB's
``pop_chanedit(EEG, 'lookup', '...dipfit/standard_BEM/elec/standard_1005.elc')`` assigns (EEGLAB axes:
X nose, Y left, Z up, mm). Synthetic recordings are placeholder files, so tests patch
``adgraph.io.chanlocs.read_chanlocs`` with this record (real MAT files are still read)."""

from pathlib import Path

import numpy as np

LOOKUP = r"D:\eeglab2022.1\plugins\dipfit\standard_BEM\elec\standard_1005.elc"


def template_xyz_mm(channels):
    """EEGLAB-axes coordinates (mm) of MNE's colin27_1005 positions for ``channels``."""
    import mne

    pos = {k.lower(): v for k, v in mne.channels.make_standard_montage("colin27_1005").get_positions()["ch_pos"].items()}
    ras = np.array([pos[c.lower()] for c in channels]) * 1000
    return np.c_[ras[:, 1], -ras[:, 0], ras[:, 2]]


def provider_chanlocs(channels, xyz=None, lookup=LOOKUP, nosedir="+X", path="x.set"):
    return {"path": str(path), "labels": list(channels),
            "xyz": template_xyz_mm(channels) if xyz is None else np.asarray(xyz, float),
            "chaninfo_filename": lookup, "nosedir": nosedir, "fiducials": {},
            "history_lookups": [lookup] if lookup else []}


def patch_reader(monkeypatch, channels, per_file=None):
    """Placeholder files get provider chanlocs (``per_file(path)`` may return another xyz); real MAT files are
    read normally."""
    from adgraph.io import chanlocs

    real = chanlocs.read_chanlocs

    def fake(path):
        if Path(path).exists() and Path(path).read_bytes()[:6] == b"MATLAB":
            return real(path)
        return provider_chanlocs(channels, None if per_file is None else per_file(path), path=path)
    monkeypatch.setattr(chanlocs, "read_chanlocs", fake)


def operator_geometry(cfg):
    """The ``geometry`` entry an operator built for the provider's chanlocs carries (for direct
    ``process_subject`` calls)."""
    from adgraph.config import common_channels
    from adgraph.io import chanlocs

    g = chanlocs.establish(provider_chanlocs(common_channels(cfg)), common_channels(cfg), cfg)
    return {**chanlocs.provenance(g), "channels": g["channels"], "pos_m": g["pos_m"]}
