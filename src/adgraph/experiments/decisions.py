"""Recorded decisions of the release and the frozen final analysis (``<processed>/FINAL_v1/decisions/``).

    atlas.json               the analysis atlas, recorded by the AUTHOR after notebook 04 (``record_atlas``;
                             ``scripts/final.py choose-atlas``). Code never chooses it; ``--atlas selected`` reads it.
    graph_construction.json  the development-only graph-construction decision (``scripts/final.py select``), made
                             on the recorded atlas and valid for that atlas only.
    final_analysis.json      the FROZEN final analysis (``freeze``): the exact list of (experiment, variant, context)
                             units that are evaluated once on the 13-participant holdout, the code commit and the
                             development-CV results they rest on. Written once, after every one of these units has
                             completed development CV; never overwritten.

Holdout evaluation (``xu_holdout``) of a real store is refused for any unit that is not in the frozen final analysis
(``require_frozen``; enforced in ``plan.run_experiment``), so the holdout cannot be run during the ROI comparison,
the graph-construction comparison, on exploratory variants, or before the selections are fixed
(release.HOLDOUT_POLICY). Which units are frozen is pre-specified in ``configs/final/holdout.yaml``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .. import release
from ..checkpoint import atomic_write_json
from ..source.atlas import ATLASES
from .config import CONFIG_ROOT, expand_variants, experiment_configs, load_experiment


class DecisionMissing(RuntimeError):
    """A recorded decision the step needs does not exist (yet)."""


class HoldoutNotFrozen(RuntimeError):
    """Holdout evaluation requested for a unit that is not part of the frozen final analysis."""


def decisions_dir(processed: str | Path) -> Path:
    return release.root(processed) / "decisions"


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# ------------------------------------------------------------------------------------------------ atlas
def atlas_path(processed) -> Path:
    return decisions_dir(processed) / "atlas.json"


def record_atlas(processed, atlas: str, reason: str, evidence: str | None = None) -> dict:
    """The author's atlas decision (once; a different later decision is refused, the same one is a no-op)."""
    if atlas not in ATLASES:
        raise ValueError(f"atlas must be one of {ATLASES}")
    if not reason or len(reason.strip()) < 10:
        raise ValueError("record the scientific reason for the atlas decision (--reason)")
    prev = _read(atlas_path(processed))
    if prev:
        if prev["atlas"] != atlas:
            raise RuntimeError(f"atlas {prev['atlas']!r} is already recorded ({atlas_path(processed)}); a decision "
                               "is never overwritten (changing it needs a logged protocol amendment)")
        return prev
    d = {"atlas": atlas, "reason": reason.strip(), "evidence": evidence, "recorded_by": "author",
         "decided_at": datetime.now(timezone.utc).isoformat(), "release": release.release_manifest(),
         "note": "chosen after the development-only ROI comparison (notebook 04); the holdout was not evaluated"}
    atomic_write_json(atlas_path(processed), d)
    return d


def selected_atlas(processed) -> str:
    d = _read(atlas_path(processed))
    if not d:
        raise DecisionMissing(f"no atlas decision recorded ({atlas_path(processed)}): finish notebook 04 and record "
                              "it there (scripts/final.py choose-atlas --atlas <dk68|r18> --reason ...)")
    return d["atlas"]


# ------------------------------------------------------------------------------------------------ graph construction
def graph_path(processed) -> Path:
    return decisions_dir(processed) / "graph_construction.json"


def graph_decision(processed, atlas: str) -> dict:
    """The recorded graph-construction decision, which must have been made on ``atlas`` and be complete."""
    d = _read(graph_path(processed))
    if not d:
        raise DecisionMissing(f"no graph-construction decision recorded ({graph_path(processed)}); run notebook 03 "
                              "first, or pass --graph dense explicitly")
    if d.get("reason", "").startswith("comparison incomplete"):
        raise DecisionMissing(f"the graph-construction comparison is incomplete ({d.get('missing')}); finish "
                              "notebook 03 first")
    if d.get("atlas", "dk68") != atlas:
        raise DecisionMissing(f"the graph-construction decision was made on atlas {d.get('atlas')!r}, not {atlas!r}")
    return d


# ------------------------------------------------------------------------------------------------ frozen final analysis
FINAL_FILE = "final_analysis.json"
UNIT_KEY = ("experiment", "variant", "inverse", "mode", "atlas", "graph")


def final_path(processed) -> Path:
    return decisions_dir(processed) / FINAL_FILE


def holdout_plan(root: str | Path | None = None) -> dict:
    return yaml.safe_load((Path(root or CONFIG_ROOT) / "holdout.yaml").read_text())


def planned_units(atlas: str, graph: str, plan: dict | None = None) -> list[dict]:
    """Every (experiment, variant, context) of the final analysis: the plan's experiments and inverses in its mode,
    on the recorded atlas and graph construction (Exp 0: sensor space, dk68; Exp 0-1: no graph construction)."""
    plan = plan or holdout_plan()
    cfgs = experiment_configs()
    out = []
    for exp in plan["experiments"]:
        cfg = load_experiment(cfgs[exp])
        for v in expand_variants(cfg):
            classical = v["input"] == "classical"
            sensor = classical and v["features"]["source"] == "sensor"
            for inv in (["sensor"] if sensor else plan["inverses"]):
                out.append({"exp": exp, "experiment": v["experiment"], "variant": v["variant"], "inverse": inv,
                            "mode": plan["mode"], "atlas": "dk68" if sensor else atlas,
                            "graph": "none" if classical else graph})
    return out


def _key(d: dict) -> tuple:
    return tuple(str(d[k]) for k in UNIT_KEY)


def freeze(processed, table: pd.DataFrame, commit: str | None) -> dict:
    """Freeze the final analysis: needs the atlas and graph decisions and a completed development-CV result for
    every planned unit; written once (the same freeze again is a no-op, a different one is refused)."""
    atlas = selected_atlas(processed)
    graph = graph_decision(processed, atlas)["selected"]
    units = planned_units(atlas, graph)
    dev = table[table["setting"] == "xu_dev_cv"] if len(table) else table
    have = {_key(r): r for r in dev.to_dict("records")}
    missing = [u for u in units if _key(u) not in have]
    if missing:
        raise DecisionMissing(f"{len(missing)} planned units have no completed development-CV result (first: "
                              f"{[_key(u) for u in missing[:3]]}); finish the experiment notebooks first")
    prev = _read(final_path(processed))
    if prev:
        if sorted(map(_key, prev["units"])) != sorted(map(_key, units)):
            raise RuntimeError(f"a different final analysis is already frozen ({final_path(processed)})")
        return prev
    d = {"atlas": atlas, "graph": graph, "git_commit": commit, "units": units,
         "development_cv": [{**{k: str(have[_key(u)][k]) for k in UNIT_KEY}, "auc": float(have[_key(u)]["auc"])}
                            for u in units],
         "plan": holdout_plan(), "frozen_at": datetime.now(timezone.utc).isoformat(),
         "release": release.release_manifest(), "holdout_policy": release.HOLDOUT_POLICY}
    atomic_write_json(final_path(processed), d)
    return d


def require_frozen(processed, variants: list[dict], commit: str | None) -> dict:
    """Raise ``HoldoutNotFrozen`` unless every variant (with its run context) is a unit of the frozen final
    analysis and the code is the frozen commit."""
    d = _read(final_path(processed))
    if not d:
        raise HoldoutNotFrozen("the holdout is evaluated only in the frozen final analysis (notebook 99: "
                               "scripts/final.py freeze, then scripts/final.py holdout); nothing is frozen yet")
    frozen = {_key(u) for u in d["units"]}
    bad = [v["name"] for v in variants
           if _key({"experiment": v["experiment"], "variant": v["variant"], **v["context"]}) not in frozen]
    if bad:
        raise HoldoutNotFrozen(f"not part of the frozen final analysis (holdout refused): {bad[:5]}")
    if d.get("git_commit") and commit and d["git_commit"] != commit:
        raise HoldoutNotFrozen(f"the final analysis was frozen at commit {d['git_commit']}, the code is {commit}")
    return d
