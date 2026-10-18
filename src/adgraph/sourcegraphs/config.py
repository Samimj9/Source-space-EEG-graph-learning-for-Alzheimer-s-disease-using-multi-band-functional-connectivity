"""The single store configuration (``configs/source_graphs.yaml``) resolved for one inverse and one atlas.

``resolve('lcmv', 'dk68')`` and ``resolve('wmne', 'dk68')`` return exactly the configurations the validated stores
were built with (config hashes 811690218bfa and c32bc6c67f0e; tested), so the existing stores are recognised by
their hash and never rebuilt. A coarser atlas adds ``source.atlas`` and atlas-specific store names, so its stores
can never be confused with the 68-ROI stores.
"""

from __future__ import annotations

import copy
from pathlib import Path

from ..config import config_hash, load_yaml
from ..source.atlas import ATLASES

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "source_graphs.yaml"
INVERSES = ("lcmv", "wmne")
DEFAULT_STORE_NAMES = ["graphsLX", "graphsLX10"]      # build.run_dirs default (not written into the LCMV config)
KNOWN_HASHES = {("lcmv", "dk68"): "811690218bfa", ("wmne", "dk68"): "c32bc6c67f0e"}


def load(path: str | Path | None = None) -> dict:
    return load_yaml(path or CONFIG)


def resolve(inverse: str, atlas: str = "dk68", path: str | Path | None = None) -> dict:
    """Complete store-building configuration for ``inverse`` (lcmv | wmne) and ``atlas``."""
    if inverse not in INVERSES:
        raise ValueError(f"inverse must be one of {INVERSES}, got {inverse!r}")
    if atlas not in ATLASES:
        raise ValueError(f"atlas must be one of {ATLASES}, got {atlas!r}")
    raw = load(path)
    inv = raw["inverses"][inverse]
    cfg = copy.deepcopy(raw["shared"])
    cfg["version"] = raw["version"]
    cfg["output_subdir"] = inv["output_subdir"]
    cfg["source"] = {**cfg["source"], **copy.deepcopy(inv["source"])}
    names = list(inv["store_names"])
    if atlas != "dk68":
        cfg["source"]["atlas"] = atlas
        names = [f"{n}-{atlas}" for n in names]
    if names != DEFAULT_STORE_NAMES:
        cfg["store_names"] = names
    known = KNOWN_HASHES.get((inverse, atlas))
    if known and config_hash(cfg) != known:
        raise RuntimeError(f"configs/source_graphs.yaml no longer reproduces the configuration of the validated "
                           f"{inverse} stores (hash {config_hash(cfg)} != {known})")
    return cfg


def correction(inverse: str, atlas: str = "dk68", path: str | Path | None = None) -> str | None:
    """Name of the correction whose copy of the store is used (``corr1`` for the LCMV dk68 stores), else None."""
    if atlas != "dk68":
        return None
    return (load(path).get("corrected_stores") or {}).get(inverse)


def atlas_of_config(cfg: dict) -> str:
    return cfg["source"].get("atlas", "dk68")
