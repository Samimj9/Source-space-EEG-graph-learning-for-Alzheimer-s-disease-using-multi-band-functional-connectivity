"""The isolated output namespace of the cross-dataset experiment and its safety rules.

    <processed>/CROSS_D3_D2_19CH_v1/
        CODE_COMMIT.txt            the experiment's own code pin (scripts/colab_setup.py --namespace cross)
        authorizations.json        stages the author has authorized (build_d2, train); recorded, never inferred
        audit/                     channel-mapping audit, harmonization manifest, compatibility report, EDF headers
        stores/                    the matched-channel D2 stores (graphsXD2L / XD2L10 / XD2W / XD2W10) + cache/
        plan/                      decision audit, frozen campaign record (before any target label is read)
        results/<direction>/...    source-only fits, label-blind target scores, source-only CV
        evaluation/, reports/      participant-level evaluation after the freeze, tables, figures

Everything the release wrote (``FINAL_v1/``: pins, decisions, store lock, QC, results), amendment A2
(``FINAL_v1/amendments/``) and the D3 stores (``D3_LCMV_XU``, ``D3_WMNE_XU``) are READ ONLY: ``guard`` refuses any
write path outside this namespace.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = "CROSS_D3_D2_19CH_v1"
STAGES = {
    "build_d2": "construct the full matched-channel D2 stores from the original recordings",
    "train": "real-data cross-dataset training and label-blind target prediction",
    "stability": "the source-only stability analysis (training seeds 1-4, repeated D2 source CV)",
}


class NamespaceError(RuntimeError):
    pass


class NotAuthorized(RuntimeError):
    pass


def root(processed) -> Path:
    return Path(processed) / ROOT


def pin_file() -> str:
    return f"{ROOT}/CODE_COMMIT.txt"


def sub(processed, *parts: str) -> Path:
    return guard(processed, root(processed).joinpath(*parts))


def guard(processed, path) -> Path:
    """``path`` if it lies inside the namespace, else ``NamespaceError`` (nothing outside is ever written)."""
    p, r = Path(path).resolve(), root(processed).resolve()
    if p != r and r not in p.parents:
        raise NamespaceError(f"refusing to write outside {r}: {p}")
    return Path(path)


def protected(processed) -> list[Path]:
    """Read-only locations of earlier work (the tests snapshot them before and after every write)."""
    from .. import release

    p = Path(processed)
    return [release.root(p), p / "D3_LCMV_XU", p / "D3_WMNE_XU"]


def _auth_path(processed) -> Path:
    return root(processed) / "authorizations.json"


def authorizations(processed) -> dict:
    f = _auth_path(processed)
    return json.loads(f.read_text()) if f.exists() else {}


def authorize(processed, stage: str, by: str, note: str = "") -> dict:
    """Record the author's authorization of a stage (once; never rewritten)."""
    from ..checkpoint import atomic_write_json
    from ..provenance import git_commit

    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {list(STAGES)}")
    if not by.strip():
        raise ValueError("an authorization names who gave it")
    a = authorizations(processed)
    if stage in a:
        return a[stage]
    a[stage] = {"stage": stage, "what": STAGES[stage], "by": by.strip(), "note": note.strip(),
                "git_commit": git_commit(), "at": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(guard(processed, _auth_path(processed)), a)
    return a[stage]


def require_authorized(processed, stage: str) -> dict:
    a = authorizations(processed).get(stage)
    if not a:
        raise NotAuthorized(f"stage {stage!r} ({STAGES[stage]}) has not been authorized by the author; nothing was "
                            f"run (scripts/cross_dataset.py authorize --stage {stage} --by '<name>')")
    return a
