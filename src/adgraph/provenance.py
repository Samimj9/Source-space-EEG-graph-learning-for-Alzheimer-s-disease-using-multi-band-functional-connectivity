"""Provenance helpers shared by graph construction and experiments."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
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
