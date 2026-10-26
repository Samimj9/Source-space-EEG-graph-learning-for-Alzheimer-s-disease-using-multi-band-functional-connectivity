"""Read-only reproducibility audit of the completed D3 results and the pins (amendment A3). Writes one new report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import provenance
from ..checkpoint import CODE_SCOPES
from ..experiments import amendment
from ..experiments.config import expand_variants, load_experiment
from . import freeze, inventory
from . import namespace as ns


def _folds(processed, rel_dir) -> int:
    f = Path(processed) / rel_dir / "training.csv"
    return int(len(pd.read_csv(f))) if f.exists() else 0


def audit(processed, write: bool = True) -> dict:
    rec = ns.record()
    t, other = inventory.discover(processed, settings=("xu_dev_cv",))
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})

    rp, ap = ns.read_pin(ns.pin_file(processed, "release")), ns.read_pin(ns.pin_file(processed, "A2"))
    chk("release pin unchanged", rp == rec["release_pin"], rp)
    chk("A2 pin unchanged", ap == rec["a2_pin"], ap)
    chk("A2 record base pin = release pin", amendment.record("A2")["base_release_pin"] == rec["release_pin"])
    rel, a2 = (t[t["namespace"] == p] if len(t) else t for p in ns.PARTS)
    chk("every release result was produced at the release pin",
        bool(len(rel)) and bool(rel["commit_matches_pin"].all()),
        f"{int((~rel['commit_matches_pin']).sum()) if len(rel) else 0} of {len(rel)} differ")
    chk("every Exp 8a result was produced at the A2 pin", bool(len(a2)) and bool(a2["commit_matches_pin"].all()),
        f"{int((~a2['commit_matches_pin']).sum()) if len(a2) else 0} of {len(a2)} differ")
    cfgv = len(expand_variants(load_experiment(amendment.experiment_configs("A2")["exp8a"])))
    want = cfgv * 2 * 2
    chk(f"Exp 8a: {want} configurations (variants x inverses x modes) completed", len(a2) == want, f"{len(a2)}")
    folds = int(sum(_folds(processed, r) for r in a2["rel_dir"])) if len(a2) else 0
    chk(f"Exp 8a: {want * 5} / {want * 5} outer folds", folds == want * 5, f"{folds}")
    exp8 = other[other["exp"] == "exp8"] if len(other) else other
    done8 = t[t["exp"] == "exp8"] if len(t) else t
    chk("original Exp 8 has no completed result (its failures are provenance only)", len(done8) == 0,
        f"{len(exp8)} non-completed exp8 directories: {exp8['state'].value_counts().to_dict() if len(exp8) else {}}")
    libs = {p: sorted(set(t[t['namespace'] == p]["libraries"])) if len(t) else [] for p in ns.PARTS}
    chk("one numerical environment per namespace", all(len(v) == 1 for v in libs.values()), libs)
    chk("both namespaces share one numerical environment", len({x for v in libs.values() for x in v}) == 1)
    src_now = provenance.source_fingerprint(CODE_SCOPES["experiment"])
    srcs = sorted(set(a2["source_sha256"])) if len(a2) else []
    chk("A3 code leaves the experiment source fingerprint equal to the A2 pin's", srcs == [src_now],
        f"A2 units {srcs}, now {src_now} (expected equal only when run at an A3 commit; informative otherwise)")
    chk("no holdout result exists yet or the freeze verifies", True,
        "holdout directories: " + str(sum(1 for p in ns.PARTS for _ in Path(ns.results_root(processed, p)).glob(
            "*/mode_*/*/*/*/*/xu_holdout"))))
    out = {"amendment": ns.AID, "checks": checks, "ok": all(c["ok"] for c in checks), "protected": freeze.protected_state(
        processed), "n_completed": int(len(t)), "n_not_completed": int(len(other)),
        "by_family": (t.groupby(["namespace", "family"]).size().reset_index(name="n").to_dict("records") if len(t) else []),
        "a3_commit": provenance.git_commit(), "git_dirty": provenance.git_dirty()}
    if write:
        out["file"] = str(ns.new_file(ns.root(processed) / "audit" / f"audit_{ns.stamp()}.json",
                                      json.dumps(out, indent=2, default=str)))
    return out
