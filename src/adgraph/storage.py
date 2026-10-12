"""Guard against writing real-run outputs and checkpoints to ephemeral Colab storage.

In Colab only the mounted Google Drive (``/content/drive``) survives a disconnect; the rest
of the VM disk (``/content``, ``/tmp``, ``/root`` ...) is lost with the runtime. A path
under ``/content/drive`` is itself ephemeral if Drive is *not* mounted: writes then land on
the VM disk under a directory that merely looks like Drive.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

COLAB_ROOT = Path("/content")
DRIVE_MOUNT = Path("/content/drive")


class EphemeralStorageError(RuntimeError):
    """A real-run output/checkpoint location would not survive a Colab disconnect."""


def in_colab() -> bool:
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


def drive_mounted(mount: Path = DRIVE_MOUNT) -> bool:
    return os.path.ismount(mount)


def _under(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


def storage_problem(path: str | Path, colab: bool | None = None, mounted: bool | None = None) -> str | None:
    """Why ``path`` is ephemeral, or None if it is persistent storage."""
    p = Path(path).expanduser().absolute()
    try:
        p = p.resolve(strict=False)             # follow symlinks (e.g. /content/data -> Drive)
    except OSError:
        pass
    if _under(p, DRIVE_MOUNT):
        if mounted is None:
            mounted = drive_mounted()
        return None if mounted else "Google Drive is not mounted at /content/drive (writes would go to the VM disk)"
    if _under(p, COLAB_ROOT):
        return "ephemeral Colab VM storage under /content (lost when the runtime disconnects)"
    if colab is None:
        colab = in_colab()
    if colab:
        return "outside the mounted Google Drive in a Colab runtime (lost when the runtime disconnects)"
    return None


def check_persistent(locations: dict[str, str | Path], allow_ephemeral: bool = False,
                     colab: bool | None = None, mounted: bool | None = None) -> list[str]:
    """Raise ``EphemeralStorageError`` if any real-run location is ephemeral (warn if allowed)."""
    problems = [f"{name} = {path}: {why}" for name, path in locations.items()
                if path is not None and (why := storage_problem(path, colab, mounted))]
    if problems:
        msg = ("real-run checkpoints/outputs must live on the mounted Google Drive:\n  " + "\n  ".join(problems))
        if not allow_ephemeral:
            raise EphemeralStorageError(msg + "\n(--allow-ephemeral-storage overrides this; outputs would be lost)")
        log.warning("%s\n(allowed explicitly; these outputs will be lost on disconnect)", msg)
    return problems
