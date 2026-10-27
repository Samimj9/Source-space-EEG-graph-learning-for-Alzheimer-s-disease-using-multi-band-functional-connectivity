"""The A3 dual-namespace freeze (methodology log entry 61).

Nothing is trained here. ``freeze`` (dry run by default) checks that every unit of the A3 plan has a complete,
checksum-verified development-CV result produced at the right commit, and only with ``write=True`` (and an approved A3
record) creates, each exactly once:

  release record  FINAL_v1/decisions/final_analysis.json                       Exp 0-6 units, commit = release pin
  A2 record       FINAL_v1/amendments/A2_exp8_surface/decisions/final_analysis.json   Exp 8a units, commit = A2 pin
  master record   FINAL_v1/amendments/A3_final/freeze.json                     both records' SHA-256, pins, protected
                                                                               files (QC decisions, store lock, pins)

The two records use the format the pinned code already reads (``decisions.require_frozen`` at the release pin,
``amendment.holdout`` at the A2 pin): unit membership + commit, plus A3 provenance fields. A record that already exists
is accepted only if it is the identical A3 freeze (rerunning completes an interrupted freeze); any other is refused.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import release
from ..experiments import amendment, decisions
from ..experiments.config import expand_variants, load_experiment
from . import environment, inventory
from . import namespace as ns


class FreezeRefused(RuntimeError):
    """A precondition of the freeze failed (nothing was written)."""


class FreezeConflict(RuntimeError):
    """A different frozen record already exists (never overwritten)."""


class FreezeInvalid(RuntimeError):
    """A frozen analysis no longer verifies (record altered, pin moved, development result changed)."""


def _key(u: dict) -> tuple:
    return tuple(str(u[k]) for k in inventory.UNIT_KEY)


def recorded_decisions(processed) -> tuple[str, str]:
    atlas = decisions.selected_atlas(processed)
    graph = decisions.graph_decision(processed, atlas)["selected"]
    return atlas, graph


def planned_units_for(atlas: str, graph: str, restrict=None) -> dict[str, list[dict]]:
    """{part: units} of the A3 plan for a recorded atlas and graph (configuration only; no Drive access)."""
    plan = ns.plan_config()
    out = {"release": decisions.planned_units(atlas, graph, {"mode": plan["mode"], "inverses": plan["inverses"],
                                                              "experiments": plan["parts"]["release"]["experiments"]})}
    cfg = amendment.experiment_configs("A2")["exp8a"]
    out["A2"] = [{"exp": "exp8a", "experiment": v["experiment"], "variant": v["variant"], "inverse": inv,
                  "mode": plan["mode"], "atlas": atlas, "graph": graph}
                 for inv in plan["inverses"] for v in expand_variants(load_experiment(cfg))]
    if restrict:
        out = {p: [u for u in us if restrict(u)] for p, us in out.items()}
    return out


def planned_units(processed, restrict=None) -> dict[str, list[dict]]:
    """{part: units} on the atlas and graph recorded in ``processed``. ``restrict`` (tests only) filters units and marks
    the freeze synthetic; the real CLI never passes it."""
    rec = ns.record()
    atlas, graph = recorded_decisions(processed)
    if (atlas, graph) != (rec["recorded_atlas"], rec["recorded_graph"]):
        raise FreezeRefused(f"the recorded decisions are atlas {atlas!r}, graph {graph!r}; amendment A3 was defined "
                            f"for {rec['recorded_atlas']!r} / {rec['recorded_graph']!r}")
    return planned_units_for(atlas, graph, restrict)


# ------------------------------------------------------------------------------------------------ preconditions
def _chk(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def protected_state(processed) -> dict:
    """SHA-256 of the files A3 promises not to change: both pins, the store lock, the QC decisions."""
    def h(p):
        return ns.sha256_file(p) if Path(p).exists() else None
    return {"release_pin_file": h(ns.pin_file(processed, "release")), "a2_pin_file": h(ns.pin_file(processed, "A2")),
            "stores_lock": h(Path(processed) / release.LOCK_FILE),
            "qc_decisions": h(release.root(processed) / "qc" / "decisions.yaml")}


def dev_results(processed, units: dict[str, list[dict]], found: pd.DataFrame | None = None) -> tuple[list, list]:
    """(development rows of every planned unit, problems). A unit is eligible only if its result is complete (marker
    and payload checksums), a development-CV result, and was produced at the part's pin."""
    if found is None:
        found, _ = inventory.discover(processed, settings=("xu_dev_cv",))
    dev = found[found["setting"] == "xu_dev_cv"] if len(found) else found
    idx = {(r["namespace"],) + _key(r): r for r in dev.to_dict("records")}
    rows, problems = [], []
    for part, us in units.items():
        pin = ns.part_pin(part)
        for u in us:
            r = idx.get((part,) + _key(u))
            if r is None:
                problems.append(f"{part}: no completed development-CV result for {_key(u)}")
            elif r["git_commit"] != pin:
                problems.append(f"{part}: {_key(u)} was produced at {r['git_commit']}, not the pin {pin}")
            else:
                rows.append({"part": part, **r})
    return rows, problems


def preconditions(processed, restrict=None, found: pd.DataFrame | None = None, strict_holdout_yaml: bool = True
                  ) -> dict:
    rec = ns.record()
    checks = []
    checks.append(_chk("original holdout.yaml unchanged (SHA-256 recorded in A3)",
                       ns.sha256_file(Path(decisions.CONFIG_ROOT) / "holdout.yaml") == rec["original_holdout_plan_sha256"]
                       or not strict_holdout_yaml))
    plan, orig = ns.plan_config(), decisions.holdout_plan()
    want = [plan["replaces"].get(e, e) for e in orig["experiments"]]
    checks.append(_chk("A3 plan = holdout.yaml with exp8 -> exp8a (mode, inverses, experiments)",
                       plan["experiments"] == want and plan["mode"] == orig["mode"]
                       and plan["inverses"] == orig["inverses"]))
    checks.append(_chk("Exp 7 and window mode A are outside the plan",
                       "exp7" not in plan["experiments"] and plan["mode"] == "B"))
    rp, ap = ns.read_pin(ns.pin_file(processed, "release")), ns.read_pin(ns.pin_file(processed, "A2"))
    checks.append(_chk("release pin file holds the A3 release_pin", rp == rec["release_pin"], f"{rp}"))
    checks.append(_chk("A2 pin file holds the A3 a2_pin", ap == rec["a2_pin"], f"{ap}"))
    checks.append(_chk("A2 record's base_release_pin = release pin",
                       amendment.record("A2")["base_release_pin"] == rec["release_pin"]))
    try:
        units = planned_units(processed, restrict)
        checks.append(_chk("atlas and graph decisions recorded and as defined in A3", True,
                           "/".join(recorded_decisions(processed))))
    except (decisions.DecisionMissing, FreezeRefused) as exc:
        checks.append(_chk("atlas and graph decisions recorded and as defined in A3", False, str(exc)))
        return {"checks": checks, "ok": False, "units": {}, "dev": []}
    sensor = [u for u in units["release"] if u["exp"] == "exp0"]
    checks.append(_chk("Exp 0 units are sensor space (inverse sensor, dk68, no graph)",
                       all((u["inverse"], u["atlas"], u["graph"]) == ("sensor", "dk68", "none") for u in sensor)))
    checks.append(_chk("no exp8 unit in the plan", not any(u["exp"] == "exp8" for us in units.values() for u in us)))
    rows, problems = dev_results(processed, units, found)
    checks.append(_chk("every planned unit has a complete, verified development-CV result at its pin",
                       not problems, f"{len(rows)} of {sum(map(len, units.values()))} units; "
                       + "; ".join(problems[:3]) + (f" (+{len(problems) - 3} more)" if len(problems) > 3 else "")))
    if rows:                                   # per-unit numerical-environment policy (a3/environment.py)
        env_inv = environment.inventory(processed, pd.DataFrame(rows))
        for c in environment.checks(env_inv):
            checks.append({**c, "check": "planned units: " + c["check"]})
    existing = [p for p in ns.PARTS if ns.record_path(processed, p).exists()]
    checks.append(_chk("no conflicting frozen record exists",
                       all(_matches(processed, p, units[p], raise_=False) for p in existing),
                       f"existing: {existing}"))
    hold = [str(d) for p in ns.PARTS for d in Path(ns.results_root(processed, p)).glob("*/mode_*/*/*/*/*/xu_holdout")]
    checks.append(_chk("no holdout result exists before the freeze", not hold or ns.master_path(processed).exists(),
                       f"{len(hold)} existing"))
    return {"checks": checks, "ok": all(c["ok"] for c in checks), "units": units, "dev": rows}


def dev_signature(rows: list[dict]) -> str:
    """Checksum over the metrics / predictions hashes of the development results a freeze rests on."""
    return ns.sha256_json(sorted((r["part"], list(_key(r)), r["metrics_sha256"], r["predictions_sha256"])
                                 for r in rows))


def inventory_table(units: dict[str, list[dict]], dev: list[dict]) -> pd.DataFrame:
    by = {(r["part"],) + _key(r): r for r in dev}
    rows = []
    for part, us in units.items():
        for u in us:
            r = by.get((part,) + _key(u), {})
            rows.append({"part": part, "pin": ns.part_pin(part)[:10], **u, "dev_auc": r.get("auc"),
                         "dev_ci_low": r.get("auc_ci_low"), "dev_ci_high": r.get("auc_ci_high"),
                         "dev_env": environment.env_id(r["libraries"]) if r else None,
                         "dev_n_participants": r.get("n_participants"), "dev_result": r.get("rel_dir")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ freeze
def _record(processed, part: str, units: list[dict], dev: list[dict], atlas: str, graph: str, a3_commit: str,
            synthetic: bool) -> dict:
    plan = ns.plan_config()
    sub = plan["parts"][part]
    mine = {(r["part"],) + _key(r): r for r in dev}
    return {
        "amendment": ns.AID, "namespace": part, "atlas": atlas, "graph": graph, "git_commit": ns.part_pin(part),
        "units": units,
        "development_cv": [{**{k: str(u[k]) for k in inventory.UNIT_KEY},
                            "auc": float(mine[(part,) + _key(u)]["auc"]),
                            "metrics_sha256": mine[(part,) + _key(u)]["metrics_sha256"],
                            "predictions_sha256": mine[(part,) + _key(u)]["predictions_sha256"],
                            "rel_dir": mine[(part,) + _key(u)]["rel_dir"]} for u in units],
        "plan": {k: v for k, v in plan.items() if k != "parts"} | {"part": sub},
        "plan_sha256": ns.sha256_file(Path(decisions.CONFIG_ROOT) / "amendments" / "holdout_A3.yaml"),
        "original_plan_sha256": ns.record()["original_holdout_plan_sha256"],
        "substitution": {"exp8": "exp8a", "reason": ns.record()["reason"].strip()},
        "frozen_at": datetime.now(timezone.utc).isoformat(), "frozen_by_a3_commit": a3_commit,
        "synthetic": synthetic, "holdout_policy": release.HOLDOUT_POLICY,
        "note": "frozen by amendment A3 (methodology log entry 61); nothing was trained; the holdout is descriptive only",
    }


def _matches(processed, part: str, units: list[dict], raise_: bool = True) -> bool:
    """An existing record is acceptable only if it is the A3 freeze of exactly these units at this part's pin."""
    d = ns.read_json(ns.record_path(processed, part))
    ok = (d is not None and d.get("amendment") == ns.AID and d.get("namespace") == part
          and sorted(map(_key, d.get("units", []))) == sorted(map(_key, units))
          and d.get("git_commit") == ns.part_pin(part))
    if not ok and raise_:
        raise FreezeConflict(f"a different frozen analysis already exists ({ns.record_path(processed, part)}); frozen "
                             "records are never overwritten")
    return ok


def freeze(processed, write: bool = False, restrict=None, a3_commit: str | None = None, require_clean: bool = True,
           found: pd.DataFrame | None = None, strict_holdout_yaml: bool = True) -> dict:
    """Dry run (default): the checks and the inventory. ``write=True``: create the missing records and the master."""
    from .. import provenance

    pre = preconditions(processed, restrict, found, strict_holdout_yaml)
    inv = inventory_table(pre["units"], pre["dev"]) if pre["units"] else pd.DataFrame()
    out = {"dry_run": not write, "checks": pre["checks"], "ok": pre["ok"], "inventory": inv, "written": [],
           "write_methods": {}}
    if not write:
        return out
    if not pre["ok"]:
        bad = [c for c in pre["checks"] if not c["ok"]]
        raise FreezeRefused("the freeze was not written: " + "; ".join(f"{c['check']} ({c['detail']})" for c in bad))
    rec = ns.record()
    if not str(rec.get("status", "")).startswith("approved") or not rec.get("approved_by"):
        raise FreezeRefused("amendment A3 is not approved (configs/final/amendments.yaml: set approved_by and "
                            "status: approved, commit, and run the freeze from that commit)")
    a3_commit = a3_commit or provenance.git_commit()
    if require_clean and provenance.git_dirty():
        raise FreezeRefused("the working tree has uncommitted changes under src/ or configs/; the freeze records the "
                            "A3 commit and must run from a clean checkout")
    atlas, graph = recorded_decisions(processed)
    units, dev = pre["units"], pre["dev"]
    synthetic = bool(restrict)
    for part in ns.PARTS:                                    # records first, master last (an interrupted freeze resumes)
        path = ns.record_path(processed, part)
        if path.exists():
            _matches(processed, part, units[part])
            continue
        ns.write_once(path, _record(processed, part, units[part], dev, atlas, graph, a3_commit, synthetic))
        out["written"].append(str(path))
        out["write_methods"][str(path)] = ns.METHODS.get(str(path))
    master = ns.master_path(processed)
    if master.exists():
        verify(processed, restrict=restrict)
        out["master"] = ns.read_json(master)
        return out
    m = {"amendment": ns.AID, "frozen_at": datetime.now(timezone.utc).isoformat(), "a3_commit": a3_commit,
         "release_pin": rec["release_pin"], "a2_pin": rec["a2_pin"], "atlas": atlas, "graph": graph,
         "records": {p: {"path": str(ns.record_path(processed, p).relative_to(processed)),
                         "sha256": ns.sha256_file(ns.record_path(processed, p)), "n_units": len(units[p]),
                         "git_commit": ns.part_pin(p)} for p in ns.PARTS},
         "n_units": sum(map(len, units.values())),
         "development_results_sha256": dev_signature(dev),
         "protected": protected_state(processed), "amendment_record": rec,
         "development_environments": {k: int(v) for k, v in pd.Series(
             [environment.env_id(r["libraries"]) for r in dev]).value_counts().items()},
         "environment_policy": "a3/environment.py: identical or compatible (Python patch, torch cpu/CUDA build) only",
         "plan_sha256": ns.sha256_file(Path(decisions.CONFIG_ROOT) / "amendments" / "holdout_A3.yaml"),
         "original_holdout_sha256": rec["original_holdout_plan_sha256"], "synthetic": synthetic,
         "substitution": {"exp8": "exp8a", "reason": rec["reason"].strip()}, "holdout_policy": release.HOLDOUT_POLICY}
    ns.write_once(master, m)
    out["written"].append(str(master))
    out["write_methods"][str(master)] = ns.METHODS.get(str(master))
    out["master"] = m
    if not (ns.root(processed) / "freeze_inventory.csv").exists():
        ns.new_file(ns.root(processed) / "freeze_inventory.csv", inv.to_csv(index=False))
    return out


# ------------------------------------------------------------------------------------------------ verification
def verify(processed, restrict=None, check_development: bool = True) -> dict:
    """Raise ``FreezeInvalid`` unless the frozen analysis still verifies: master and both records present, record
    checksums as frozen, units as planned, commits equal to the pins, pins / QC decisions / store lock unchanged, and
    (optionally) the development results unchanged. Read only."""
    m = ns.read_json(ns.master_path(processed))
    if m is None:
        raise FreezeInvalid(f"no A3 master freeze record ({ns.master_path(processed)}); the holdout is refused")
    problems = []
    units = planned_units(processed, restrict)
    for p in ns.PARTS:
        path = ns.record_path(processed, p)
        if not path.exists():
            problems.append(f"{p}: record missing")
            continue
        if ns.sha256_file(path) != m["records"][p]["sha256"]:
            problems.append(f"{p}: record changed since the freeze (checksum)")
        d = ns.read_json(path)
        if sorted(map(_key, d["units"])) != sorted(map(_key, units[p])):
            problems.append(f"{p}: units differ from the A3 plan")
        if d["git_commit"] != ns.part_pin(p) or m["records"][p]["git_commit"] != ns.part_pin(p):
            problems.append(f"{p}: record commit is not the pin")
    prot_now, prot_then = protected_state(processed), m["protected"]
    for k, v in prot_then.items():
        if prot_now.get(k) != v:
            problems.append(f"protected file changed since the freeze: {k}")
    if check_development:
        found, _ = inventory.discover(processed, settings=("xu_dev_cv",))
        rows, probs = dev_results(processed, units, found)
        problems += probs
        sig = dev_signature(rows)
        if sig != m["development_results_sha256"]:
            problems.append("development-CV results changed since the freeze")
    if problems:
        raise FreezeInvalid("; ".join(problems))
    return {"ok": True, "master": m}
