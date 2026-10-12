"""Provenance helpers shared by graph construction and experiments."""

from __future__ import annotations

import importlib
import platform
import subprocess
from pathlib import Path


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def versions(packages=("numpy", "scipy", "pandas", "mne", "sklearn", "torch")) -> dict:
    out = {"python": platform.python_version()}
    for name in packages:
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:
            out[name] = None
    return out
