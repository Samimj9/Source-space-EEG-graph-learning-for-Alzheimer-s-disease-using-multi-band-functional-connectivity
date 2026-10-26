"""Paths, records and write-once / checksum helpers of amendment A3."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import yaml

from .. import release
from ..experiments.config import CONFIG_ROOT

AID = "A3"
NAMESPACE = "A3_final"
PARTS = ("release", "A2")                           # release: Exp 0-6 at the release pin; A2: Exp 8a at the A2 pin
A2_DIR = "A2_exp8_surface"


class A3Error(RuntimeError):
    pass


def record() -> dict:
    a = (yaml.safe_load((Path(CONFIG_ROOT) / "amendments.yaml").read_text()) or {}).get(AID)
    if not a:
        raise A3Error("amendment A3 is not in configs/final/amendments.yaml")
    return a


def plan_config() -> dict:
    return yaml.safe_load((Path(CONFIG_ROOT) / "amendments" / "holdout_A3.yaml").read_text())


def stats_config() -> dict:
    return yaml.safe_load((Path(CONFIG_ROOT) / "amendments" / "a3_statistics.yaml").read_text())


def root(processed) -> Path:
    return release.root(processed) / "amendments" / NAMESPACE


def a2_root(processed) -> Path:
    return release.root(processed) / "amendments" / A2_DIR


def master_path(processed) -> Path:
    return root(processed) / "freeze.json"


def pin_path(processed) -> Path:
    return root(processed) / "CODE_COMMIT.txt"


def results_root(processed, part: str) -> Path:
    return release.results_dir(processed) if part == "release" else a2_root(processed) / "results"


def record_path(processed, part: str) -> Path:
    """The frozen record of a part, in the location the pinned code of that part reads."""
    if part == "release":
        return release.root(processed) / "decisions" / "final_analysis.json"
    return a2_root(processed) / "decisions" / "final_analysis.json"


def part_pin(part: str) -> str:
    return record()["release_pin" if part == "release" else "a2_pin"]


def read_pin(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    return next((ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")),
                None)


def pin_file(processed, part: str) -> Path:
    return release.root(processed) / "CODE_COMMIT.txt" if part == "release" else a2_root(processed) / "CODE_COMMIT.txt"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def read_json(path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------------------------------------ write once
# Hard links are not available on every file system (a Google Drive FUSE mount in Colab is not POSIX-complete and has
# not been verified to support ``os.link`` from the development environment). ``_publish`` therefore uses the strongest
# method the file system offers and never overwrites silently:
#   "link"          temporary file hard-linked into place (atomic; fails if the target exists). Preferred.
#   "claim+rename"  when the file system refuses hard links (errno in LINK_UNSUPPORTED): an exclusive-create claim file
#                   ``<name>.claim`` is taken, the target is checked to be absent while the claim is held, the finished
#                   temporary file is renamed onto it, and the content is read back. The same rename is what every other
#                   atomic write of this repository (``checkpoint.atomic_write_*``) already relies on, on Drive too.
# In both methods the content is complete before the target name exists (an interrupted write leaves only a hidden
# temporary file, never a partial record), and an existing target is refused (``FileExistsError``). The fallback is
# protected against a concurrent A3 writer by the claim file; it cannot stop a process that ignores the claim in the
# microseconds between the final check and the rename, which Drive cannot guarantee either way (single-user use).
LINK_UNSUPPORTED = {errno.EPERM, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EACCES, errno.EXDEV,
                    errno.EMLINK, errno.EINVAL}
CLAIM_TTL_S = 600
METHODS: dict[str, str] = {}              # target path -> method that created it, in this process


class WriteInProgress(A3Error):
    """Another (recent) writer holds the claim of this target."""


class WriteConflict(A3Error):
    """The target does not contain what was just written."""


def _acquire_claim(claim: Path) -> None:
    for _ in range(2):
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                created = float(json.loads(claim.read_text())["created"])
            except (OSError, ValueError, KeyError):
                created = claim.stat().st_mtime if claim.exists() else 0.0
            if time.time() - created < CLAIM_TTL_S:
                raise WriteInProgress(f"{claim} is held by another writer (younger than {CLAIM_TTL_S} s); if it is a "
                                      "leftover of an interrupted run it becomes stale after that time") from None
            claim.unlink(missing_ok=True)                   # stale leftover of an interrupted write: take over
            continue
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"pid": os.getpid(), "created": time.time()}))
            f.flush()
            os.fsync(f.fileno())
        return
    raise WriteInProgress(f"could not take {claim}")


def _publish(path: Path, text: str) -> str:
    """Create ``path`` with ``text`` exactly once; returns the method used. Never overwrites."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} exists; frozen records and reports are never overwritten")
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(tmp, path)
            method = "link"
        except FileExistsError:
            raise
        except OSError as exc:
            if exc.errno not in LINK_UNSUPPORTED:
                raise
            claim = path.with_name(path.name + ".claim")
            _acquire_claim(claim)
            try:
                if path.exists():
                    raise FileExistsError(f"{path} exists; frozen records and reports are never overwritten")
                os.replace(tmp, path)
                if path.read_text() != text:
                    raise WriteConflict(f"{path} does not contain what was written")
            finally:
                claim.unlink(missing_ok=True)
            method = "claim+rename"
    finally:
        tmp.unlink(missing_ok=True)
    METHODS[str(path)] = method
    return method


def write_once(path: str | Path, obj) -> Path:
    """Create ``path`` with ``obj`` as JSON exactly once (see ``_publish``)."""
    _publish(Path(path), json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    return Path(path)


def new_file(path: str | Path, text: str) -> Path:
    """Write a text file that must not exist yet (reports, manifests): never overwrites."""
    _publish(Path(path), text)
    return Path(path)


def stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
