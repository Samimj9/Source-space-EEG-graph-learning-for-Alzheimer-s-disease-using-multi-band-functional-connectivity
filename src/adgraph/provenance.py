"""Provenance helpers shared by graph construction and experiments."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
from functools import lru_cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]


@lru_cache(maxsize=None)
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PACKAGE_DIR,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


@lru_cache(maxsize=None)
def git_dirty() -> bool | None:
    """True if tracked files under src/ or configs/ differ from the commit (None without git)."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "configs"],
                                      cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return bool(out.strip())


@lru_cache(maxsize=None)
def source_fingerprint(parts: tuple[str, ...]) -> str:
    """SHA-256 over the Python source files of the given package parts (files or subpackages).

    Unlike the git commit, this also captures uncommitted edits, and it does not need git."""
    files = []
    for part in parts:
        p = PACKAGE_DIR / part
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    h = hashlib.sha256()
    for f in sorted(set(files)):
        h.update(str(f.relative_to(PACKAGE_DIR)).encode())
        h.update(b"\0")
        h.update(f.read_bytes() if f.exists() else b"<missing>")
        h.update(b"\0")
    return h.hexdigest()


def versions(packages=("numpy", "scipy", "pandas", "mne", "sklearn", "torch")) -> dict:
    out = {"python": platform.python_version()}
    for name in packages:
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:
            out[name] = None
    return out


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def compute_environment(device: str = "cpu") -> dict:
    """Hardware/software description of the machine that produced timing results.

    ``fingerprint`` hashes everything that affects run time (CPU model and count, GPU,
    device used, thread count, platform, library versions) but not the host name or
    time, so runs from the same kind of environment compare equal.
    """
    env = {"device": device, "cpu_model": _cpu_model(), "cpu_count": os.cpu_count(),
           "platform": platform.platform(), "machine": platform.machine(),
           "versions": versions(("numpy", "scipy", "sklearn", "torch")), "gpu": None, "torch_threads": None}
    try:
        import torch

        env["torch_threads"] = torch.get_num_threads()
        if device.startswith("cuda") and torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
    except Exception:
        pass
    env["fingerprint"] = hashlib.sha256(json.dumps(env, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return env


# ----------------------------------------------------------------- pinned environment
def pinned_requirements(path: str | Path | None = None) -> dict[str, str]:
    """``name -> version`` for every exact (``==``) pin in requirements.txt."""
    path = Path(path) if path else REPO_ROOT / "requirements.txt"
    pins = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s#;]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def check_pinned_environment(path: str | Path | None = None) -> list[str]:
    """Problems with the installed versions of exactly pinned packages (empty list = OK)."""
    from importlib.metadata import PackageNotFoundError, version

    problems = []
    for name, want in pinned_requirements(path).items():
        try:
            have = version(name)
        except PackageNotFoundError:
            problems.append(f"{name}=={want} is pinned but not installed")
            continue
        if have != want:
            problems.append(f"{name}=={want} is pinned but {have} is installed")
    return problems


def verify_environment(path: str | Path | None = None) -> None:
    """Raise if the pinned real-run environment is not installed (pip install -r requirements.txt)."""
    problems = check_pinned_environment(path)
    if problems:
        raise RuntimeError("environment does not match requirements.txt: " + "; ".join(problems)
                           + " (run: pip install -r requirements.txt)")
