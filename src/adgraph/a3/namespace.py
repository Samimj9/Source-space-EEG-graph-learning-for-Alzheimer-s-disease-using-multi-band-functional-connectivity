"""Paths, records and write-once / checksum helpers of amendment A3."""

from __future__ import annotations

import hashlib
import json
import os
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


def write_once(path: str | Path, obj) -> Path:
    """Create ``path`` with ``obj`` as JSON exactly once: the content is written to a temporary file and hard-linked
    into place, which fails (``FileExistsError``) if ``path`` exists, so a frozen record is never overwritten, not even
    by a concurrent run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def new_file(path: str | Path, text: str) -> Path:
    """Write a text file that must not exist yet (reports, manifests): never overwrites."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    tmp.write_text(text)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
