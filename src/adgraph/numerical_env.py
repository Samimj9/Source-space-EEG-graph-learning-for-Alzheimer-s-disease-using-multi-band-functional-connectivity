"""Run a later notebook in the exact numerical environment of completed release units (methodology log entry 60).

Experiment units record the versions of Python and of the numerical libraries in their code identity
(``checkpoint.code_identity("experiment")``: python, numpy, scipy, pandas, sklearn, torch). Colab changes its default
interpreter and libraries over time; an amendment that must be comparable with the completed units (A2: Exp 8a next to
Exp 0-7) runs its scripts with an interpreter that reproduces those versions exactly:

* ``reference_libraries``: the versions recorded by a completed release unit (default: an Exp 7 unit). Only the
  library and Python versions are compared, never the git commit or source hash (an amendment has its own commit).
* ``interpreter_versions``: the versions an interpreter actually imports (queried in a subprocess).
* ``ensure_interpreter``: the notebook kernel's interpreter if it already matches; else an existing restored
  environment (``<base>/restore<python digits>``, e.g. ``/content/restore31315``) if it matches; else that environment
  is built with ``uv`` (exact Python, exact library versions, then ``requirements.txt`` under those constraints) and
  verified. A difference that remains raises unless it is accepted explicitly (and then recorded by the amendment
  check). Nothing here is written on Drive.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

# version key in a code identity -> distribution name for pip / uv
DISTRIBUTIONS = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "sklearn": "scikit-learn", "torch": "torch",
                 "mne": "mne"}
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"

_QUERY = ("import importlib, json, platform, sys\n"
          "out = {'python': platform.python_version()}\n"
          "for k in sys.argv[1:]:\n"
          "    try:\n"
          "        out[k] = importlib.import_module(k).__version__\n"
          "    except Exception:\n"
          "        out[k] = None\n"
          "print(json.dumps(out))\n")


class EnvironmentMismatch(RuntimeError):
    pass


def reference_libraries(processed, experiment: str = "exp7_earm_gnn") -> dict:
    """{python, numpy, ...: version} recorded by a completed release unit of ``experiment``."""
    from .experiments.amendment import reference_identity

    ref = reference_identity(processed, experiment)
    lib = ((ref or {}).get("code") or {}).get("libraries") or {}
    if not lib:
        raise EnvironmentMismatch(f"no completed {experiment} unit with a recorded code identity under "
                                  f"{processed}; the reference environment is unknown")
    return dict(lib)


def interpreter_versions(python: str, keys) -> dict | None:
    """Versions imported by ``python`` (None if it cannot be run)."""
    keys = [k for k in keys if k != "python"]
    try:
        p = subprocess.run([str(python), "-c", _QUERY, *keys], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode:
        return None
    return json.loads(p.stdout.strip().splitlines()[-1])


def differences(reference: dict, have: dict | None) -> dict:
    """{key: {"reference": ..., "now": ...}} of every reference version that differs (all, if ``have`` is None)."""
    have = have or {}
    return {k: {"reference": v, "now": have.get(k)} for k, v in reference.items() if have.get(k) != v}


def restore_target(reference: dict, base: str | Path = "/content") -> Path:
    """Folder of the restored environment, named by its Python version (3.13.15 -> <base>/restore31315)."""
    return Path(base) / ("restore" + reference["python"].replace(".", ""))


def requirement(key: str, version: str) -> str:
    return f"{DISTRIBUTIONS.get(key, key)}=={version}"


def restore_commands(reference: dict, target: str | Path, repo: str | Path, uv: str = "uv") -> list[list[str]]:
    """The commands that build ``target`` with exactly the reference versions (torch from its CUDA wheel index
    when the recorded version carries a local tag such as +cu130), then the repository's requirements under those
    pins and the package itself (no dependencies)."""
    target = Path(target)
    py = str(target / "bin" / "python")
    pins = [requirement(k, v) for k, v in reference.items() if k != "python" and v]
    cmds = [[uv, "venv", "--python", reference["python"], str(target)]]
    torch = reference.get("torch")
    extra = []
    if torch and "+" in torch:
        index = TORCH_INDEX.format(tag=torch.split("+", 1)[1])
        cmds.append([uv, "pip", "install", "--python", py, requirement("torch", torch), "--index-url", index])
        extra = ["--extra-index-url", index, "--index-strategy", "unsafe-best-match"]
    cmds.append([uv, "pip", "install", "--python", py, "-r", str(Path(repo) / "requirements.txt"), *pins, *extra])
    cmds.append([uv, "pip", "install", "--python", py, "--no-deps", "-e", str(repo)])
    return cmds


def _uv(run) -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    run(f"{sys.executable} -m pip install -q uv")
    return shutil.which("uv") or "uv"


def _table(reference: dict, rows: dict) -> str:
    keys = list(reference)
    head = ["", "reference (Exp 7)", *rows]
    lines = [head] + [[k, str(reference[k]), *[str((r or {}).get(k)) for r in rows.values()]] for k in keys]
    w = [max(len(line[i]) for line in lines) for i in range(len(head))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(line)) for line in lines)


def ensure_interpreter(reference: dict, repo: str | Path, base: str | Path = "/content",
                       accept_difference: bool = False, run=None, current: str | None = None) -> str:
    """The interpreter that reproduces ``reference`` (see module docstring); builds the restored environment if
    neither the current interpreter nor an existing restored one matches. ``run``: shell runner (``colab.sh``)."""
    import shlex

    if run is None:
        from .colab import sh as run
    current = current or sys.executable
    target = restore_target(reference, base)
    restored = str(target / "bin" / "python")
    have = interpreter_versions(current, reference)
    rows = {"this kernel": have}
    if not differences(reference, have):
        print(_table(reference, rows) + f"\n-> the kernel's interpreter matches: {current}")
        return current
    found = interpreter_versions(restored, reference) if Path(restored).exists() else None
    if Path(restored).exists() and not differences(reference, found):
        print(_table(reference, {**rows, "restored": found}) + f"\n-> existing restored environment: {restored}")
        return restored
    if Path(restored).exists() and (found or {}).get("python") not in (None, reference["python"]):
        raise EnvironmentMismatch(f"{target} exists with Python {found['python']}, not {reference['python']}; "
                                  "remove it (rm -rf) and run again")
    print(_table(reference, rows) + f"\n-> building the restored environment {target} (a few minutes)")
    uv = _uv(run)
    cmds = restore_commands(reference, target, repo, uv)
    for cmd in cmds[1:] if Path(restored).exists() else cmds:      # an existing venv with this Python: re-pin only
        run(shlex.join(cmd))
    found = interpreter_versions(restored, reference)
    diff = differences(reference, found)
    print(_table(reference, {**rows, "restored": found}))
    if not diff:
        print(f"-> restored environment verified: {restored}")
        return restored
    if accept_difference:
        print(f"WARNING: accepted difference {diff}; it is recorded by the amendment check")
        return restored if found else current
    raise EnvironmentMismatch(f"the restored environment still differs from the reference: {diff}")
