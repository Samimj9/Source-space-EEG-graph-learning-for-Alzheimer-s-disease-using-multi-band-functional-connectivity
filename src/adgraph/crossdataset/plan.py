"""The pre-specified cross-dataset campaign (``configs/cross_dataset/campaign.yaml``), its decision audit and freeze.

* ``variants``        the experiment variants, loaded from the existing definitions (Exp 0, 1 as reference; Exp 3-7
                      and amendment A2's Exp 8a), never redefined here; the original Exp 8 is refused;
* ``contextualize``   a variant in a run context (inverse, mode, recorded atlas and graph construction);
* ``units``           every (direction, inverse, mode, experiment, variant, seed) of a stage;
* ``decision_audit``  what earlier decisions this campaign inherits, from the release's records (read only), and what
                      was inspected before (disclosed in every report);
* ``freeze``          ``plan/frozen.json``: the campaign, its units, the store identities and the code commit, written
                      once before any source fit; ``require_frozen`` refuses training and evaluation otherwise.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..checkpoint import atomic_write_json, code_identity, stable_hash
from ..experiments.config import expand_variants, experiment_configs, load_experiment, with_context
from . import namespace

CAMPAIGN = Path(__file__).resolve().parents[3] / "configs" / "cross_dataset" / "campaign.yaml"
DIRECTIONS = ("D3_to_D2", "D2_to_D3")


class NotFrozen(RuntimeError):
    pass


def load_campaign(path: str | Path | None = None) -> dict:
    c = yaml.safe_load(Path(path or CAMPAIGN).read_text())
    if set(c["directions"]) != set(DIRECTIONS):
        raise ValueError(f"directions must be {DIRECTIONS}")
    if "exp8" in c["experiments"]["gnn"] + c["experiments"]["reference"]:
        raise ValueError("the original exp8 is undefined on lobes12; the campaign uses amendment A2's exp8a")
    if set(c["training"]["seeds"]) & set(c["stability"]["seeds"]):
        raise ValueError("stability seeds must differ from the primary seed")
    return c


def recording_condition_check(c: dict) -> dict:
    """Decision D6: the recording condition of each dataset is stated (``unknown`` is allowed, never invented), and an
    unknown condition carries its stated implication for external validity."""
    rc = c.get("recording_condition") or {}
    ok = all(str(rc.get(d, "")).strip() for d in ("D3", "D2"))
    unknown = [d for d in ("D3", "D2") if str(rc.get(d, "")).strip().lower().startswith("unknown")]
    if unknown and not str(rc.get("implication", "")).strip():
        ok = False
    return {"passed": ok, "unknown": unknown, "D3": rc.get("D3"), "D2": rc.get("D2"),
            "implication": rc.get("implication")}


def campaign_hash(c: dict) -> str:
    return stable_hash(c)


def config_paths() -> dict[str, Path]:
    from ..experiments import amendment

    paths = dict(experiment_configs())
    paths.pop("exp8")
    paths["exp8a"] = amendment.experiment_configs("A2")["exp8a"]
    return paths


def variants(c: dict, which: tuple[str, ...] = ("gnn", "reference")) -> dict[str, list[dict]]:
    """{experiment id: [variants]} of the campaign, from the unchanged experiment definitions."""
    paths = config_paths()
    out = {}
    for group in which:
        for exp in c["experiments"][group]:
            out[exp] = expand_variants(load_experiment(paths[exp]))
    return out


def is_sensor(v: dict) -> bool:
    return v["input"] == "classical" and v["features"]["source"] == "sensor"


def contextualize(v: dict, inverse: str, mode: str, c: dict) -> dict:
    """The variant in its run context. Exp 0 (sensor PSD / DE) runs in sensor space (inverse ``sensor``; the release
    labels its atlas dk68, it reads no ROI)."""
    if is_sensor(v):
        return with_context(v, "sensor", mode, "dk68", None)
    if v["input"] == "classical":
        return with_context(v, inverse, mode, c["atlas"], None)
    return with_context(v, inverse, mode, c["atlas"], c["graph"])


def units(c: dict, stage: str = "primary", experiments: list[str] | None = None,
          variant_filter: list[str] | None = None) -> list[dict]:
    """Every unit of a stage (``primary``: the protocol seed; ``stability``: the additional seeds)."""
    if stage not in ("primary", "stability"):
        raise ValueError(f"unknown stage {stage!r}")
    seeds = c["training"]["seeds"] if stage == "primary" else c["stability"]["seeds"]
    out = []
    for exp, vs in variants(c).items():
        if experiments and exp not in experiments:
            continue
        for v in vs:
            if variant_filter and v["variant"] not in variant_filter:
                continue
            for d in DIRECTIONS:
                for mode in c["modes"]:
                    for inv in (["sensor"] if is_sensor(v) else c["inverses"]):
                        for s in seeds:
                            out.append({"direction": d, "inverse": inv, "mode": mode, "exp": exp,
                                        "experiment": v["experiment"], "variant": v["variant"], "seed": int(s),
                                        "input": v["input"],
                                        "exploratory": exp in c["experiments"]["exploratory"],
                                        "reference": exp in c["experiments"]["reference"]})
    return out


def unit_key(u: dict) -> tuple:
    return tuple(str(u[k]) for k in ("direction", "inverse", "mode", "experiment", "variant", "seed"))


# ------------------------------------------------------------------------------------------------ decision audit
DISCLOSURES = [
    "The D3 holdout (13 participants) was inspected in earlier work, so it is NOT an untouched holdout; it is never "
    "used to fit or select anything here and is reported separately as a descriptive stratum of the D2 -> D3 target.",
    "D2 was analysed within-dataset in earlier work (legacy D1 / D2 pipeline: 30 channels, wMNE, 68 ROIs, 20 s "
    "windows). The model families were designed with knowledge of those and of earlier D3 results; no configuration "
    "of this campaign is chosen on D2, and the cross-dataset transfer itself has never been run before this freeze.",
    "The atlas (lobes12) and the graph construction (dense) were recorded on D3 development data only (notebooks 04 "
    "and 03); they are inherited, not chosen here.",
    "Exp 7 (EARM-GNN) was designed after earlier results; Exp 8a is amendment A2 (post-design, pre-result). Both are "
    "exploratory model-family comparisons.",
    "All variants of every listed experiment are evaluated and reported; no winning architecture, variant, inverse, "
    "window mode or preprocessing is selected with target-domain results.",
    "The D2 preprocessing (19 electrodes, 0.5-45 Hz band-pass, edge crops) was fixed from the D3 montage, the D3 "
    "provider's documented filter, the pre-specified label-blind edge rule (entries 40-41) and the filter's impulse "
    "response; no D2 classification result informed it.",
    "Identical template electrode coordinates do not establish identical physical electrode placement: neither dataset "
    "has digitized positions, and the D2 cap may sit differently on each head.",
    "The D2 recording condition (eyes closed / open) is unknown unless stated otherwise in the frozen campaign; D3 is "
    "eyes closed.",
]


def decision_audit(processed, c: dict) -> dict:
    """The inherited decisions (read from the release's records, never written) and the disclosures."""
    from ..experiments import decisions

    out = {"campaign_hash": campaign_hash(c), "disclosures": DISCLOSURES, "checks": {}}

    def check(name, ok, detail=""):
        out["checks"][name] = {"passed": bool(ok), "detail": detail}

    a = decisions._read(decisions.atlas_path(processed))
    check("recorded atlas = campaign atlas", bool(a) and a.get("atlas") == c["atlas"], (a or {}).get("atlas"))
    g = decisions._read(decisions.graph_path(processed))
    check("recorded graph construction = campaign graph, decided on that atlas",
          bool(g) and g.get("selected") == c["graph"] and g.get("atlas", "dk68") == c["atlas"],
          {k: (g or {}).get(k) for k in ("selected", "atlas")})
    from ..experiments import amendment

    check("Exp 8a is amendment A2 (configs/final/amendments.yaml)", amendment.record("A2").get("config", "").endswith(
        "exp8a_xu_dual_gcn_surface.yaml"), amendment.record("A2").get("kind"))
    tgt = namespace.root(processed) / "evaluation"
    check("no target evaluation exists in this namespace before the freeze",
          not any(tgt.glob("**/*.json")) if tgt.exists() else True, str(tgt))
    rc = recording_condition_check(c)
    check("recording condition of both datasets stated (unknown allowed, with its implication)", rc["passed"],
          {k: rc[k] for k in ("D3", "D2")})
    out["recording_condition"] = rc
    out["atlas_decision"] = {k: (a or {}).get(k) for k in ("atlas", "reason", "decided_at")}
    out["graph_decision"] = {k: (g or {}).get(k) for k in ("selected", "atlas", "reason")}
    out["passed"] = all(x["passed"] for x in out["checks"].values())
    return out


# ------------------------------------------------------------------------------------------------ freeze
def frozen_path(processed) -> Path:
    return namespace.sub(processed, "plan", "frozen.json")


def freeze(processed, c: dict, stores: dict, audit: dict, synthetic: bool = False) -> dict:
    """Write the frozen campaign once (a different freeze is refused). ``stores``: {name: graph_run_id} of every
    store the campaign reads."""
    from ..provenance import git_commit

    if not audit.get("passed") and not synthetic:
        raise NotFrozen(f"decision audit failed: {[k for k, v in audit['checks'].items() if not v['passed']]}")
    us = units(c, "primary") + units(c, "stability")
    rec = {"campaign": c, "campaign_hash": campaign_hash(c), "units_hash": stable_hash([unit_key(u) for u in us]),
           "n_units_primary": len(units(c, "primary")), "n_units_stability": len(units(c, "stability")),
           "stores": stores, "git_commit": git_commit(), "code": code_identity("cross_experiment"),
           "decision_audit": audit, "synthetic": synthetic, "frozen_at": datetime.now(timezone.utc).isoformat(),
           "rule": "target labels are read only by the evaluation of units of this record, at this commit"}
    f = frozen_path(processed)
    if f.exists():
        prev = json.loads(f.read_text())
        same = (prev["campaign_hash"], prev["units_hash"], prev["stores"]) == (rec["campaign_hash"],
                                                                              rec["units_hash"], rec["stores"])
        if not same:
            raise NotFrozen(f"a different campaign is already frozen ({f})")
        return prev
    atomic_write_json(f, rec)
    return rec


def require_frozen(processed, c: dict | None = None) -> dict:
    from ..provenance import git_commit

    f = frozen_path(processed)
    if not f.exists():
        raise NotFrozen("the campaign is not frozen (scripts/cross_dataset.py freeze); nothing is trained or evaluated")
    rec = json.loads(f.read_text())
    if c is not None and rec["campaign_hash"] != campaign_hash(c):
        raise NotFrozen("configs/cross_dataset/campaign.yaml differs from the frozen campaign")
    if rec["git_commit"] != git_commit():
        raise NotFrozen(f"the campaign was frozen at commit {rec['git_commit']}, this code is {git_commit()} (use the "
                        "namespace's code pin)")
    return rec
