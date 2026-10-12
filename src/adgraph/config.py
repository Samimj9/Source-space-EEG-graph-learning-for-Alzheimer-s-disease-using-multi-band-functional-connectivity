"""Loading of path and pipeline configs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve(value: Any, root: str) -> Any:
    if isinstance(value, str):
        return value.replace("{drive_root}", root)
    if isinstance(value, dict):
        return {k: _resolve(v, root) for k, v in value.items()}
    return value


def load_paths(path: str | Path) -> dict[str, Any]:
    """Load ``configs/paths.yaml`` and expand ``{drive_root}`` placeholders."""
    raw = load_yaml(path)
    root = raw["drive_root"]
    return {k: _resolve(v, root) for k, v in raw.items()}


def config_hash(cfg: dict[str, Any]) -> str:
    """Short, stable hash of a config dict (key order independent)."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def common_channels(cfg: dict[str, Any]) -> list[str]:
    """Channels entering source modelling, in canonical order."""
    drop = set(cfg["channels"]["drop"])
    return [ch for ch in cfg["channels"]["recorded"] if ch not in drop]
