"""Orchestration of the descriptive holdout of the A3 frozen analysis (nothing here fits a model itself).

Each part runs the CLI of the commit that produced its development results, in a separate git worktree:

  release part (Exp 0-6)  ->  <release pin>  python scripts/final.py holdout --exp <exp>
  A2 part (Exp 8a)        ->  <A2 pin>       python scripts/final.py holdout --amendment A2 --exp exp8a

The pinned code applies its own gates (``decisions.require_frozen`` with the frozen record and the code commit). A3 adds:
the master freeze must verify before and after every call, the worktree must be clean and at exactly the pin, the
interpreter must import the worktree's package, and every call is logged in ``A3_final/manifests/`` (never overwritten).
Interrupted calls resume from the pinned code's own checkpoints; completed units are returned unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import environment, freeze, inventory
from . import namespace as ns


class HoldoutRefused(RuntimeError):
    pass


def holdout_command(part: str, exp: str, paths_file: str | Path) -> list[str]:
    plan = ns.plan_config()["parts"][part]
    if exp not in plan["experiments"]:
        raise HoldoutRefused(f"{exp} is not an experiment of the {part} part ({plan['experiments']})")
    cmd = ["scripts/final.py", "holdout", "--paths", str(paths_file)]
    if part == "A2":
        cmd += ["--amendment", "A2"]
    return cmd + ["--exp", exp]


def _git(cwd, *a) -> str:
    return subprocess.check_output(["git", *a], cwd=cwd, text=True).strip()


def ensure_worktree(repo: str | Path, base: str | Path, part: str, commit: str) -> Path:
    """A clean detached worktree of ``repo`` at ``commit`` (created if missing; verified if it exists)."""
    wt = Path(base) / f"a3_{part}_{commit[:10]}"
    if not wt.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(wt), commit], cwd=repo, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    head = _git(wt, "rev-parse", "HEAD")
    if head != commit:
        raise HoldoutRefused(f"worktree {wt} is at {head}, not the pin {commit}")
    if _git(wt, "status", "--porcelain", "--untracked-files=no"):
        raise HoldoutRefused(f"worktree {wt} has uncommitted changes")
    return wt


def imports_worktree(python: str, wt: Path) -> bool:
    out = subprocess.run([python, "-c", "import adgraph, sys; print(adgraph.__file__)"], cwd=wt,
                         env={**os.environ, "PYTHONPATH": str(wt / "src")}, stdout=subprocess.PIPE, text=True)
    return out.returncode == 0 and Path(out.stdout.strip()).resolve().is_relative_to((wt / "src").resolve())


def run_part(processed, part: str, exp: str, repo, paths_file, python: str, base="/content", restrict=None,
             runner=None, a3_commit: str | None = None, env_probe=None) -> dict:
    """Evaluate the holdout units of ``exp`` of ``part`` (resumable). Refused unless the A3 freeze verifies."""
    from .. import provenance

    freeze.verify(processed, restrict=restrict)                        # raises FreezeInvalid / refuses
    env_check = check_environment(processed, part, exp, python, restrict, env_probe)
    pin = ns.part_pin(part)
    wt = ensure_worktree(repo, base, part, pin)
    if not imports_worktree(python, wt):
        raise HoldoutRefused(f"{python} does not import the package of {wt}; the pinned code must run")
    cmd = [python, *holdout_command(part, exp, paths_file)]
    started = datetime.now(timezone.utc).isoformat()
    runner = runner or (lambda c, cwd, env: subprocess.run(c, cwd=cwd, env=env).returncode)
    code = runner(cmd, str(wt), {**os.environ, "PYTHONPATH": str(wt / "src"), "MPLBACKEND": "Agg"})
    freeze.verify(processed, restrict=restrict)                        # the call changed no frozen record / result
    man = {"amendment": ns.AID, "part": part, "exp": exp, "pin": pin, "worktree": str(wt), "command": cmd,
           "exit_code": code, "started": started, "finished": datetime.now(timezone.utc).isoformat(),
           "a3_commit": a3_commit or provenance.git_commit(), "interpreter": python,
           "master_sha256": ns.sha256_file(ns.master_path(processed)), "environment": env_check}
    ns.new_file(ns.root(processed) / "manifests" / f"holdout_{part}_{exp}_{ns.stamp()}.json",
                json.dumps(man, indent=2))
    return man


def check_environment(processed, part: str, exp: str, python: str, restrict=None, env_probe=None) -> dict:
    """The interpreter that will refit the units must be identical or documented-compatible (a3/environment.py) with the
    environment each unit's development result was produced in; otherwise the holdout is refused. The comparison is
    recorded in the manifest (a refit in another PyTorch build is not bit-identical)."""
    from .. import numerical_env

    units = [u for u in freeze.planned_units(processed, restrict)[part] if u["exp"] == exp]
    found, _ = inventory.discover(processed, settings=("xu_dev_cv",))
    rows, problems = freeze.dev_results(processed, {part: units}, found)
    if problems:
        raise HoldoutRefused("development results missing for the units to refit: " + "; ".join(problems[:3]))
    probe = env_probe or numerical_env.interpreter_versions
    libs = probe(python, ("numpy", "scipy", "pandas", "sklearn", "torch"))
    res = environment.interpreter_check(libs, [r["libraries"] for r in rows])
    res["interpreter"] = libs
    if res["level"] == "incompatible":
        raise HoldoutRefused(f"the holdout interpreter's numerical environment is incompatible with the development "
                             f"environment of {res['counts']['incompatible']} of {res['n_units']} {exp} units: "
                             f"{res['differences']}")
    return res


def holdout_dir(processed, part: str, u: dict) -> Path:
    return (ns.results_root(processed, part) / u["inverse"] / f"mode_{u['mode']}" / u["atlas"] / u["graph"]
            / u["experiment"] / u["variant"] / "xu_holdout")


def status(processed, restrict=None) -> pd.DataFrame:
    """Completed / partial / pending holdout units of the frozen analysis (read only, checksums verified)."""
    rows = []
    for part, us in freeze.planned_units(processed, restrict).items():
        for u in us:
            d = holdout_dir(processed, part, u)
            if (d / "final.done.json").exists():
                ok, why = inventory._marker_ok(d)
                state = "completed" if ok else f"corrupt ({why})"
            else:
                state = "partial" if d.exists() else "pending"
            rows.append({"part": part, **u, "holdout_state": state})
    return pd.DataFrame(rows)
