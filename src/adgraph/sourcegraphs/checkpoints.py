"""Where a store's per-participant checkpoints are, verified (read-only; for diagnostics only).

The store builder saves each participant's checkpoint (``subjects/<key>/``: ``operator.npz``, ``roi_timeseries.npz``,
``windows.csv``, ``subject.done.json``) inside the mode B run directory only. Two kinds of validated store therefore
have no ``subjects/`` directory of their own:

* a corrected store (``<run>_corr1``, ``sourcegraphs.correct``; methodology log entry 53): a row subset of the
  original assembled arrays; no beamformer, feature or connectivity was recomputed, so its source operators and ROI
  signals ARE those of the original build's checkpoints;
* a mode A store (``<run10>``): the first ten accepted windows of the mode B run.

``locate`` returns the directory that holds the checkpoints and verifies, before anything is read, that they belong
to the store: the source run's ``graph_run_id`` is the one the correction recorded, the build configuration is
identical, and for every participant the completion marker exists and carries the same participant key,
configuration hash, recording SHA-256 (the store manifest's input list) and electrode geometry (``montage_hash``),
and the payload files used have the SHA-256 the marker recorded. The store's own ``windows.csv`` stays authoritative:
diagnostics use only its windows (a subset of the checkpoint windows; checked). Nothing is copied, moved or written.

Historical caveat (LCMV ``_corr1``): the operators of the participants the correction changed (D3-23, D3-58) were
estimated in the original build from a data covariance that included one extra, later removed, clean 10 s window
(``covariance_caveat``; methodology log entry 53).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..checkpoint import file_sha256
from ..config import config_hash


class CheckpointProvenanceError(RuntimeError):
    """The checkpoints found do not verifiably belong to the store."""


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def source_dir(store_root: str | Path) -> tuple[Path, str, str | None]:
    """(checkpoint directory, relation, graph_run_id it must have or None)."""
    root = Path(store_root)
    if (root / "subjects").is_dir():
        return root, "own checkpoints", None
    man = _json(root / "manifest.json")
    corr = man.get("correction")
    if corr:
        src = corr["source"]["mode_B"]
        return root.parent / Path(src["run_dir"]).name, f"original build of the corrected store ({corr['name']})", \
            src["graph_run_id"]
    from .build import run_dirs

    b, _ = run_dirs(root.parent.parent, man["config"])            # mode A: the mode B run of the same build
    return root.parent / b.name, "mode B run of the same build", None


def locate(store_root: str | Path, keys: list[str] | None = None, files=("operator.npz",)) -> dict:
    """Verified checkpoint location for the store's participants (``keys``: a subset). Raises
    ``CheckpointProvenanceError`` on any mismatch; participants without a checkpoint are listed in ``missing``."""
    root = Path(store_root)
    src, relation, gid = source_dir(root)
    man = _json(root / "manifest.json")
    problems = []
    if not (src / "subjects").is_dir() or not (src / "manifest.json").exists():
        raise CheckpointProvenanceError(f"{root.name}: no participant checkpoints at {src} ({relation})")
    sman = _json(src / "manifest.json")
    if gid is not None and sman.get("graph_run_id") != gid:
        problems.append(f"source graph_run_id {sman.get('graph_run_id')} != recorded {gid}")
    chash = config_hash(man["config"])
    if config_hash(sman["config"]) != chash:
        problems.append("build configuration differs from the store's")
    subj = pd.read_csv(root / "subjects.csv", dtype={"key": str, "montage_hash": str})
    if "status" in subj:
        subj = subj[subj["status"] == "ok"]
    if keys is not None:
        subj = subj[subj["key"].isin(set(keys))]
    inputs = {r["key"]: r.get("sha256") for r in man.get("inputs", [])}
    win = pd.read_csv(root / "windows.csv", usecols=["window_id", "key"])
    ok, missing = [], []
    for r in subj.itertuples():
        unit = src / "subjects" / r.key
        mp = unit / "subject.done.json"
        if not mp.exists() or any(not (unit / f).exists() for f in files):
            missing.append(r.key)
            continue
        mk = _json(mp)
        ident = mk.get("identity", {})
        if ident.get("key") != r.key or ident.get("config_hash") != chash:
            problems.append(f"{r.key}: checkpoint identity (key / configuration) differs")
        if inputs.get(r.key) and (ident.get("input") or {}).get("sha256") != inputs[r.key]:
            problems.append(f"{r.key}: checkpoint recording SHA-256 differs from the store's input list")
        if "montage_hash" in subj and ident.get("geometry") != str(getattr(r, "montage_hash")):
            problems.append(f"{r.key}: checkpoint electrode geometry differs from the store's")
        for f in files:
            if mk.get("files", {}).get(f) != file_sha256(unit / f):
                problems.append(f"{r.key}: {f} does not match its completion marker")
        if (unit / "windows.csv").exists():
            mine = set(win.loc[win["key"] == r.key, "window_id"])
            if not mine <= set(pd.read_csv(unit / "windows.csv", usecols=["window_id"])["window_id"]):
                problems.append(f"{r.key}: store windows are not a subset of the checkpoint windows")
        ok.append(r.key)
    if problems:
        raise CheckpointProvenanceError(f"{root.name}: checkpoints at {src.name} not verified: " + "; ".join(problems[:8]))
    corr = man.get("correction") or {}
    caveat = sorted(corr.get("participants_changed", [])) if (man.get("source_operator") or {}).get("method") == "lcmv" \
        else []
    return {"store": root.name, "checkpoint_dir": str(src), "relation": relation, "verified": True,
            "participants": ok, "missing": missing, "n_verified": len(ok),
            "checks": ["source graph_run_id" if gid else "same run", "build configuration hash",
                       "per participant: marker key, configuration hash, recording SHA-256, electrode geometry",
                       f"payload SHA-256 of {', '.join(files)}", "store windows subset of checkpoint windows"],
            "covariance_caveat": {k: "LCMV covariance of the original build included one extra clean 10 s window "
                                     "later removed by the correction (methodology log entry 53)" for k in caveat}}
