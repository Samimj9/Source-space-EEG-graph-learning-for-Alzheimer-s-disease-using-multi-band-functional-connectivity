"""Numerical-environment provenance of the completed results and its compatibility policy (amendment A3; read only).

Every experiment unit records the Python and library versions it ran with (``checkpoint.code_identity``: python, numpy,
scipy, pandas, sklearn, torch). Colab changes its default interpreter and PyTorch build over time, so the completed
D3 results can come from more than one environment, e.g. Python 3.13.15 + torch 2.11.0+cu130 (GPU runtime) and Python
3.13.16 + torch 2.11.0+cpu (CPU runtime). A3 does not require one identical environment; it applies a documented policy:

  identical     every recorded version equal.
  compatible    equal NumPy, SciPy, pandas, scikit-learn, mne, torch release (the part before ``+``) and Python
                minor version; differing only in the Python PATCH version and/or the PyTorch BUILD tag (cpu / cuNNN).
                These builds implement the same numerical methods of the same releases, but they are different binaries
                (CPU kernels versus CUDA kernels, different BLAS / cuDNN paths), so identical floating-point results
                are NOT guaranteed: a GNN refit in another build can differ in the last digits and, over 60 epochs, in
                its predictions by an amount no one has quantified here. The difference is reported, never hidden, and
                the audit states it as a limitation. Classical models (logistic regression, SVM) do not use PyTorch.
  incompatible  anything else (another NumPy / SciPy / pandas / scikit-learn / mne release, another torch release, another
                Python minor): a genuine, unexplained mismatch that fails the audit and refuses the freeze / holdout.

Within one result every outer fold must carry the environment of the result: the fold checkpoints embed the code identity
and a resume in another environment raises ``IncompatibleCheckpoint``, so a mixed result can only arise from tampering or
a forced recomputation; it is checked, not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CORE = ("numpy", "scipy", "pandas", "sklearn", "mne")          # must be identical wherever recorded
GNN_EXPERIMENTS = ("exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", "exp8a")


def parse(libs: dict | str | None) -> dict:
    """Normalised view of a recorded ``libraries`` dict (or its JSON)."""
    if isinstance(libs, str):
        libs = json.loads(libs)
    libs = dict(libs or {})
    py = str(libs.get("python") or "")
    tv = str(libs.get("torch") or "")
    pub, _, build = tv.partition("+")
    out = {k: libs.get(k) for k in CORE}
    out.update({"python": py or None, "python_minor": ".".join(py.split(".")[:2]) if py else None,
                "python_patch": py or None, "torch": tv or None, "torch_release": pub or None,
                "torch_build": (build or ("default" if tv else None))})
    b = out["torch_build"]
    out["device_kind"] = ("cuda" if b and b.startswith("cu") else "cpu" if b == "cpu" else
                          "default" if b else "none")
    return out


def env_id(libs) -> str:
    """Label unique per recorded signature (python, torch and every core library), e.g.
    ``py3.13.15|torch2.11.0+cu130|numpy2.1.3|scipy1.16.3|pandas2.2.3|sklearn1.6.1``."""
    p = parse(libs)
    return "|".join([f"py{p['python'] or '?'}", f"torch{p['torch'] or '-'}"]
                    + [f"{k}{p[k]}" for k in CORE if p[k] is not None])


def compatibility(a, b) -> dict:
    """{'level': identical | compatible | incompatible, 'differences': {field: (a, b)}} for two recorded environments."""
    pa, pb = parse(a), parse(b)
    core = {k: (pa[k], pb[k]) for k in (*CORE, "python_minor", "torch_release") if pa[k] != pb[k]}
    soft = {k: (pa[k], pb[k]) for k in ("python_patch", "torch_build") if pa[k] != pb[k]}
    level = "incompatible" if core else "compatible" if soft else "identical"
    return {"level": level, "differences": {**core, **soft}, "core_differences": core}


def fold_environments(result_dir: str | Path) -> list[str]:
    """Distinct environments (canonical JSON of the recorded libraries) of the fold checkpoints of one result."""
    seen = set()
    for f in sorted((Path(result_dir) / "units").glob("fold_*.done.json")):
        try:
            libs = json.loads(f.read_text())["identity"]["code"]["libraries"]
        except (OSError, ValueError, KeyError, TypeError):
            libs = None
        seen.add(json.dumps(libs, sort_keys=True))
    return sorted(seen)


def inventory(processed, found: pd.DataFrame, plan_keys: set | None = None, check_folds: bool = True) -> pd.DataFrame:
    """One row per completed development result: its environment, the environments of its folds, whether it belongs to
    the 116-unit final plan. ``found``: ``inventory.discover`` table (development results). Read only."""
    from . import inventory as inv

    rows = []
    for r in found.to_dict("records"):
        p = parse(r["libraries"])
        folds = fold_environments(Path(processed) / r["rel_dir"]) if check_folds else []
        own = json.dumps(json.loads(r["libraries"]), sort_keys=True)
        key = tuple(str(r[k]) for k in inv.UNIT_KEY)
        rows.append({"namespace": r["namespace"], "family": r["family"], "exp": r["exp"], "experiment": r["experiment"],
                     "variant": r["variant"], "inverse": r["inverse"], "mode": r["mode"], "atlas": r["atlas"],
                     "graph": r["graph"], "setting": r["setting"], "git_commit": r["git_commit"],
                     "env_id": env_id(r["libraries"]), "python": p["python"], "numpy": p["numpy"], "scipy": p["scipy"],
                     "pandas": p["pandas"], "sklearn": p["sklearn"], "torch": p["torch"], "device_kind": p["device_kind"],
                     "uses_torch": r["exp"] in GNN_EXPERIMENTS, "n_fold_environments": len(folds),
                     "folds_match_result": (not check_folds) or folds == [own],
                     "in_final_plan": (None if plan_keys is None else (r["namespace"],) + key in plan_keys),
                     "libraries": r["libraries"], "rel_dir": r["rel_dir"]})
    return pd.DataFrame(rows)


def summary(inv: pd.DataFrame) -> pd.DataFrame:
    """Results per namespace, experiment and environment (and how many belong to the final plan)."""
    if not len(inv):
        return pd.DataFrame()
    g = inv.groupby(["namespace", "family", "env_id", "device_kind"], dropna=False)
    return g.agg(n_results=("variant", "size"), n_in_final_plan=("in_final_plan", lambda s: int(s.fillna(False).sum())),
                 contexts=("inverse", lambda s: ",".join(sorted(set(s)))),
                 modes=("mode", lambda s: ",".join(sorted(set(s))))).reset_index()


def checks(inv: pd.DataFrame, reference: dict | None = None) -> list[dict]:
    """Audit checks with a level: pass | warn | fail (``ok`` is False only for fail). ``reference``: the environment the
    holdout will be restored to (default: the single environment of the Exp 8a results)."""
    out = []

    def add(name, level, detail=""):
        out.append({"check": name, "ok": level != "fail", "warning": level == "warn", "detail": str(detail)})
    if not len(inv):
        add("numerical environments recorded", "fail", "no completed result")
        return out
    ids = inv.groupby("env_id").size().to_dict()
    envs = {i: json.loads(inv[inv["env_id"] == i].iloc[0]["libraries"]) for i in ids}
    base = next(iter(sorted(envs, key=lambda i: -ids[i])))
    bad = {i: compatibility(envs[base], envs[i]) for i in envs}
    incompatible = {i: c["core_differences"] for i, c in bad.items() if c["level"] == "incompatible"}
    add("recorded numerical libraries identical across all results (numpy, scipy, pandas, scikit-learn, mne; torch release; "
        "Python minor)", "fail" if incompatible else "pass",
        f"{len(envs)} environment(s): {ids}" if not incompatible else f"incompatible with {base}: {incompatible}")
    soft = {i: c["differences"] for i, c in bad.items() if c["level"] == "compatible"}
    if soft:
        gnn_cpu = inv[(inv["device_kind"] == "cpu") & inv["uses_torch"]]
        add("documented compatible differences only (Python patch version, PyTorch cpu / CUDA build)", "warn",
            f"{ids}; differences vs {base}: {soft}; GNN results from a CPU torch build: {len(gnn_cpu)}"
            + (f" ({sorted(gnn_cpu['family'].unique())})" if len(gnn_cpu) else "")
            + "; results are not bit-identical across builds (see docs/final/a3_design.md, environment policy)")
    else:
        add("documented compatible differences only (Python patch version, PyTorch cpu / CUDA build)", "pass",
            "a single environment")
    mixed = inv[~inv["folds_match_result"]]
    add("no result mixes environments across its outer folds", "fail" if len(mixed) else "pass",
        f"{len(mixed)} of {len(inv)} results differ from their fold checkpoints"
        + (f": {mixed[['namespace', 'experiment', 'variant', 'inverse', 'mode']].head(3).values.tolist()}"
           if len(mixed) else ""))
    a2 = inv[inv["namespace"] == "A2"]
    rel_ids = set(inv[inv["namespace"] == "release"]["env_id"])
    ok_a2 = bool(len(a2)) and set(a2["env_id"]) <= rel_ids
    add("Exp 8a (A2) environment also occurs among the release results", "pass" if ok_a2 else "fail",
        f"A2: {sorted(set(a2['env_id']))}; release: {sorted(rel_ids)}")
    return out


def holdout_reference(inv_or_found: pd.DataFrame) -> dict:
    """The environment the holdout fits are restored to: the single environment of the Exp 8a (A2) results, which A2
    chose to equal that of the Exp 7 units. Raises if A2 has none or more than one."""
    a2 = inv_or_found[inv_or_found["namespace"] == "A2"]
    libs = sorted(set(a2["libraries"])) if len(a2) else []
    if len(libs) != 1:
        raise ValueError(f"the Exp 8a results must carry exactly one numerical environment, found {len(libs)}")
    return json.loads(libs[0])


def interpreter_check(interpreter_libs: dict | None, unit_envs: list[str]) -> dict:
    """Compare the holdout interpreter with the development environment of every unit it will refit.
    Returns counts per level and the distinct differences; ``incompatible`` units make the caller refuse."""
    if not interpreter_libs:
        return {"level": "incompatible", "n_units": len(unit_envs), "counts": {"incompatible": len(unit_envs)},
                "differences": ["the interpreter's library versions could not be read"]}
    counts, diffs = {"identical": 0, "compatible": 0, "incompatible": 0}, set()
    for u in unit_envs:
        c = compatibility(json.loads(u), interpreter_libs)
        counts[c["level"]] += 1
        diffs |= {f"{k}: development {a} vs holdout {b}" for k, (a, b) in c["differences"].items()}
    level = "incompatible" if counts["incompatible"] else "compatible" if counts["compatible"] else "identical"
    return {"level": level, "n_units": len(unit_envs), "counts": counts, "differences": sorted(diffs)}
