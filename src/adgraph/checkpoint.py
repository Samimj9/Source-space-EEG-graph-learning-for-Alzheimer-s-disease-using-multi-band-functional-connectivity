"""Checkpointing primitives: atomic writes, validated completion markers, run-status tables.

A *unit* of work (one subject's graph outputs, one outer CV fold, one inner-CV trial, one
permutation, the shared source operator) is complete only when its **marker**
``<name>.done.json`` exists and validates:

* the marker parses as JSON (it is itself written atomically);
* its ``identity`` equals the identity the caller expects (config hash, code version,
  graph run, split, repeat, fold, seed ... whatever defines the unit);
* every payload file listed in the marker exists and has the recorded SHA-256.

Payload files are written first and the marker last, all with write-to-temp + fsync +
``os.replace``. An interrupted unit therefore never has a valid marker and is never
mistaken for completed; a truncated/corrupted payload fails the checksum.

Every unit identity also carries an automatic **code identity** (``code_identity``): the
git commit, a SHA-256 of the source files that determine the computation (which also
catches uncommitted edits) and the versions of the numerical libraries. Any code change
therefore makes old units *incompatible*, whether or not anyone bumped ``CODE_VERSIONS``;
those manual labels remain for documentation only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import provenance
from .provenance import git_commit

CODE_VERSIONS = {
    "operator": "operator-1",          # sensor -> ROI operator (forward, inverse, mean_flip)
    "graph_subject": "graph-subject-3",  # per-subject preprocessing/source/PSD/connectivity (3: boundary-aware, entry 41)
    "graph_assembly": "graph-assembly-1",
    "experiment": "experiment-1",      # outer-fold training/prediction, evaluation
    "inner_trial": "inner-trial-1",    # one hyperparameter combination on one inner fold
    "permutation": "permutation-1",
}

# Package parts (relative to src/adgraph) whose source determines each kind of computation.
CODE_SCOPES = {
    "operator": ("config.py", "pipeline.py", "io", "source"),
    "graph": ("config.py", "pipeline.py", "io", "preprocessing", "source", "features", "connectivity",
              "graphs/store.py"),
    "experiment": ("config.py", "experiments", "models", "evaluation", "graphs", "features"),
    "lx_graph": ("config.py", "pipeline.py", "io", "preprocessing", "source", "features", "connectivity",
                 "graphs/store.py", "evaluation/splits.py", "lcmvxu"),
}
# Numerical libraries whose version can change results of each kind of computation.
SCOPE_LIBRARIES = {
    "operator": ("numpy", "scipy", "mne"),
    "graph": ("numpy", "scipy", "pandas", "mne"),
    "experiment": ("numpy", "scipy", "pandas", "sklearn", "torch"),
    "lx_graph": ("numpy", "scipy", "pandas", "mne"),
}

STATES = ("pending", "running", "completed", "failed")


class IncompatibleCheckpoint(RuntimeError):
    """Existing outputs were produced with a different configuration / graph run / code version."""


# ----------------------------------------------------------------- hashing
def stable_hash(obj, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def code_identity(scope: str) -> dict:
    """Automatic code fingerprint of a computation: git commit + source hash + library versions.

    Looked up through ``provenance`` at call time (tests patch it there)."""
    return {"git_commit": provenance.git_commit(),
            "source_sha256": provenance.source_fingerprint(CODE_SCOPES[scope]),
            "libraries": provenance.versions(SCOPE_LIBRARIES[scope])}


def identity_diff(found: dict | None, expected: dict) -> list[str]:
    """Top-level identity keys that differ (to explain why a unit is incompatible)."""
    found = found or {}
    return sorted(k for k in set(found) | set(expected) if _norm(found.get(k)) != _norm(expected.get(k)))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(identity: dict) -> str:
    return json.dumps(identity, sort_keys=True, default=str)


# ----------------------------------------------------------------- atomic writes
def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_text(path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode())


def atomic_write_json(path, obj) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def atomic_write_csv(path, df: pd.DataFrame) -> Path:
    return atomic_write_text(path, df.to_csv(index=False))


def _atomic_stream(path, write) -> Path:
    """Stream ``write(fileobj)`` into a temp file next to ``path``, fsync, then rename.

    Unlike building the bytes in memory first, peak memory stays at the arrays themselves
    (the compressed archive is never held in RAM). A crash leaves only the temp file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_savez(path, **arrays) -> Path:
    return _atomic_stream(path, lambda f: np.savez_compressed(f, **arrays))


def atomic_save_npy(path, array) -> Path:
    return _atomic_stream(path, lambda f: np.save(f, array))


def cleanup_temp_files(directory: str | Path) -> int:
    """Delete leftover ``.<name>.tmp-<id>`` files of interrupted atomic writes under ``directory``.

    They are never part of a completed unit (a marker lists only renamed payload files).
    Call only when no other process writes to ``directory``. Returns the number removed."""
    n = 0
    for p in Path(directory).rglob(".*.tmp-*"):
        if p.is_file() and "_superseded" not in p.parts:
            p.unlink(missing_ok=True)
            n += 1
    return n


# ----------------------------------------------------------------- units
def marker_path(directory: str | Path, name: str) -> Path:
    return Path(directory) / f"{name}.done.json"


def write_marker(directory, name: str, identity: dict, files: list[str | Path]) -> Path:
    """Write the completion marker LAST, after all payload files exist."""
    directory = Path(directory)
    entries = {}
    for f in files:
        p = Path(f)
        p = p if p.is_absolute() else directory / p
        entries[str(p.relative_to(directory))] = file_sha256(p)
    return atomic_write_json(marker_path(directory, name), {
        "identity": identity, "files": entries, "completed_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit()})


def check_unit(directory, name: str, identity: dict) -> tuple[str, dict | None]:
    """'complete' | 'missing' | 'incompatible' | 'corrupt', plus the marker (if readable)."""
    mp = marker_path(directory, name)
    if not mp.exists():
        return "missing", None
    try:
        marker = json.loads(mp.read_text())
    except (json.JSONDecodeError, OSError):
        return "corrupt", None
    if _norm(marker.get("identity")) != _norm(identity):
        return "incompatible", marker
    for rel, sha in marker.get("files", {}).items():
        p = Path(directory) / rel
        if not p.exists() or file_sha256(p) != sha:
            return "corrupt", marker
    return "complete", marker


def marker_matches(directory, name: str, identity: dict) -> bool:
    """Cheap pre-check (no payload hashing): a marker exists and carries ``identity``.

    For progress display only; whether a unit is complete is always decided by ``check_unit``."""
    mp = marker_path(directory, name)
    try:
        return mp.exists() and _norm(json.loads(mp.read_text()).get("identity")) == _norm(identity)
    except (json.JSONDecodeError, OSError):
        return False


def supersede(path: str | Path, root: str | Path) -> Path | None:
    """Move ``path`` under ``root/_superseded/<stamp>/`` (never delete, never overwrite)."""
    path, root = Path(path), Path(root)
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    dest = root / "_superseded" / stamp / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest))
    for marker in path.parent.glob(f"{path.name}.done.json"):
        shutil.move(str(marker), str(dest.parent / marker.name))
    return dest


# ----------------------------------------------------------------- status table
class StatusTable:
    """Human-readable run status (pending / running / completed / failed) in ``status.csv``.

    Informational only: whether a unit is complete is always decided by its validated
    marker, never by this table (a unit left "running" by a crash is simply recomputed).
    """

    COLUMNS = ["unit", "state", "updated", "attempts", "message"]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.df = (pd.read_csv(self.path, dtype={"message": str}).fillna({"message": ""})
                   if self.path.exists() else pd.DataFrame(columns=self.COLUMNS))

    def get(self, unit: str) -> str | None:
        row = self.df[self.df["unit"] == unit]
        return None if row.empty else row["state"].iloc[0]

    def set(self, unit: str, state: str, message: str = "") -> None:
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}")
        now = datetime.now(timezone.utc).isoformat()
        idx = self.df.index[self.df["unit"] == unit]
        if len(idx):
            i = idx[0]
            attempts = int(self.df.at[i, "attempts"]) + (1 if state == "running" else 0)
            self.df.loc[i, ["state", "updated", "attempts", "message"]] = [state, now, attempts, message]
        else:
            self.df = pd.concat([self.df, pd.DataFrame([{"unit": unit, "state": state, "updated": now,
                                                         "attempts": int(state == "running"),
                                                         "message": message}])], ignore_index=True)
        atomic_write_csv(self.path, self.df)

    def ensure(self, units: list[str]) -> None:
        missing = [u for u in units if u not in set(self.df["unit"])]
        if missing:
            now = datetime.now(timezone.utc).isoformat()
            self.df = pd.concat([self.df, pd.DataFrame([{"unit": u, "state": "pending", "updated": now, "attempts": 0,
                                                         "message": ""} for u in missing])], ignore_index=True)
            atomic_write_csv(self.path, self.df)
