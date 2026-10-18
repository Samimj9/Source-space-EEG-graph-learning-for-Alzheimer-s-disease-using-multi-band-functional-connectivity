"""Which validated store an experiment reads: inverse x window mode x atlas -> one directory, verified and locked.

    open_store(processed, inverse, mode, atlas)        inverse: lcmv | wmne | sensor (Exp 0); mode: B | A

``verify_store`` checks, without changing anything:

* the directory is the one the store configuration resolves to (``sourcegraphs.config.resolve``: same config hash),
  using the corrected copy where one exists (LCMV dk68: ``_corr1``), assembled and complete;
* manifest mode, inverse method, the five Xu bands and the atlas's ROI names in store order;
* ``validation.json`` passed for the store's current graph_run_id (``sourcegraphs.build_stores validate``);
* wMNE: ``match.json`` passed against the LCMV store of the same mode and atlas (identical windows, participants,
  split and sensor features);
* a coarser atlas (e.g. r18): ``atlas_match.json`` passed against the 68-ROI store of the same inverse and mode
  (identical participants, labels, windows, split, sampling and sensor features; ``sourcegraphs.match.match_atlas``);
* the frozen participant split: 65 participants (36 AD / 29 HC), the saved 52 / 13 partition equals
  ``xu_partition``, and it is identical to the reference LCMV dk68 store of the same mode;
* the lock: ``FINAL_v1/stores.lock.json`` records the graph_run_id the first time a store is verified; a store whose
  content changed afterwards (another graph_run_id) is refused, so all results of the release come from one
  version of each store.

``sensor`` (Exp 0) reads the sensor-space arrays of the LCMV dk68 store; the wMNE stores hold identical sensor
features (checked by ``match.json``), so Exp 0 is run once per mode, independently of the inverse.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import release
from .checkpoint import atomic_write_json
from .config import config_hash
from .evaluation.splits import xu_partition
from .graphs.store import GraphStore, cohort_problems
from .source.atlas import ATLASES, region_names
from .sourcegraphs.build import run_dirs
from .sourcegraphs.config import INVERSES, correction, resolve

MODE_NAMES = {"B": "B_all_windows", "A": "A_xu_tenfold"}
XU_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
EXPECTED_COHORT = {"n": 65, "AD": 36, "HC": 29, "development": 52, "holdout": 13}


class StoreError(RuntimeError):
    """The requested store is missing, unvalidated, unmatched, inconsistent or changed since it was locked."""


def store_inverse(inverse: str) -> str:
    """The inverse whose store is read: Exp 0 (``sensor``) reads the LCMV store's sensor arrays."""
    return "lcmv" if inverse == "sensor" else inverse


def store_dir(processed: str | Path, inverse: str, mode: str, atlas: str = "dk68") -> Path:
    inv = store_inverse(inverse)
    if inv not in INVERSES or mode not in MODE_NAMES or atlas not in ATLASES:
        raise ValueError(f"unknown store ({inverse}, {mode}, {atlas})")
    b, a = run_dirs(processed, resolve(inv, atlas))
    d = b if mode == "B" else a
    corr = correction(inv, atlas)
    return d.with_name(f"{d.name}_{corr}") if corr else d


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def verify_store(processed: str | Path, inverse: str, mode: str, atlas: str = "dk68") -> dict:
    """All checks of the module docstring; returns {passed, checks, dir, graph_run_id, ...} (never raises for a
    failed check; ``open_store`` does)."""
    inv = store_inverse(inverse)
    d = store_dir(processed, inv, mode, atlas)
    checks = {}

    def check(name, ok, detail=""):
        checks[name] = {"passed": bool(ok), "detail": detail}
        return ok

    if not check("store assembled", (d / "assembly.done.json").exists() and (d / "manifest.json").exists(), str(d)):
        return {"passed": False, "dir": str(d), "checks": checks, "graph_run_id": None}
    man = _json(d / "manifest.json")
    cfg = resolve(inv, atlas)
    check("built with the resolved store configuration", man.get("config_hash") == config_hash(cfg),
          f"{man.get('config_hash')} vs {config_hash(cfg)}")
    check("window mode", man.get("mode") == MODE_NAMES[mode], man.get("mode"))
    check("inverse method", (man.get("source_operator") or {}).get("method") == inv,
          (man.get("source_operator") or {}).get("method"))
    store = GraphStore.load(d, keys=())
    check("five Xu bands in order", store.bands == XU_BANDS, store.bands)
    check(f"ROI names of atlas {atlas} in store order", store.roi_names == region_names(atlas),
          f"{len(store.roi_names)} ROIs")
    probs = cohort_problems(store)
    check("complete cohort (no failed participant, not a smoke test)", not probs, "; ".join(probs))
    val = _json(d / "validation.json") or {}
    check("validation.json passed for this graph_run_id",
          val.get("passed") and val.get("graph_run_id") == man.get("graph_run_id"),
          "missing" if not val else f"passed={val.get('passed')}")
    if inv == "wmne":
        ref = store_dir(processed, "lcmv", mode, atlas)
        mt = _json(d / "match.json") or {}
        rid = (_json(ref / "manifest.json") or {}).get("graph_run_id")
        check("match.json passed against the LCMV store of the same mode and atlas",
              mt.get("passed") and mt.get("graph_run_id") == man.get("graph_run_id")
              and mt.get("reference_graph_run_id") == rid, "missing" if not mt else f"reference {ref.name}")
    if atlas != "dk68":
        ref = store_dir(processed, inv, mode, "dk68")
        am = _json(d / "atlas_match.json") or {}
        rid = (_json(ref / "manifest.json") or {}).get("graph_run_id")
        check("atlas_match.json passed against the 68-ROI store of the same inverse and mode",
              am.get("passed") and am.get("graph_run_id") == man.get("graph_run_id")
              and am.get("reference_graph_run_id") == rid, "missing" if not am else f"reference {ref.name}")
    subj = store.subjects
    ok = subj.loc[subj["status"] == "ok", ["key", "dataset", "y"]] if "status" in subj else subj[["key", "dataset", "y"]]
    check("65 participants, 36 AD / 29 HC", (len(ok), int(ok["y"].sum()), int((ok["y"] == 0).sum()))
          == (EXPECTED_COHORT["n"], EXPECTED_COHORT["AD"], EXPECTED_COHORT["HC"]),
          f"{len(ok)} participants, {int(ok['y'].sum())} AD")
    split = _json(d / "xu_split.json") or {}
    dev, test = xu_partition(ok)
    check("frozen split = xu_partition (52 development / 13 holdout)",
          (dev, test) == (split.get("development"), split.get("test"))
          and (len(dev), len(test)) == (EXPECTED_COHORT["development"], EXPECTED_COHORT["holdout"]),
          f"{len(dev)} / {len(test)}")
    ref = store_dir(processed, "lcmv", mode, "dk68")
    if ref != d:
        rs = _json(ref / "xu_split.json") or {}
        check("split identical to the reference LCMV dk68 store", rs.get("development") == split.get("development")
              and rs.get("test") == split.get("test"), ref.name)
    out = {"passed": all(c["passed"] for c in checks.values()), "dir": str(d), "inverse": inverse, "mode": mode,
           "atlas": atlas, "graph_run_id": man.get("graph_run_id"), "config_hash": man.get("config_hash"),
           "n_windows": int(store.n_windows), "checks": checks}
    lock = _lock_entry(processed, inverse, mode, atlas)
    if lock is not None:
        check("same store content as when the release locked it", lock["graph_run_id"] == out["graph_run_id"],
              f"locked {lock['graph_run_id']}, found {out['graph_run_id']}")
        out["passed"] = all(c["passed"] for c in checks.values())
    return out


def _lock_key(inverse: str, mode: str, atlas: str) -> str:
    return f"{inverse}/{mode}/{atlas}"


def _lock_entry(processed, inverse, mode, atlas) -> dict | None:
    lock = _json(Path(processed) / release.LOCK_FILE) or {}
    return (lock.get("stores") or {}).get(_lock_key(inverse, mode, atlas))


def lock_store(processed: str | Path, res: dict) -> None:
    """Record a verified store in the release lock (first verification only; never rewritten)."""
    path = Path(processed) / release.LOCK_FILE
    lock = _json(path) or {"release": release.RELEASE, "stores": {}}
    key = _lock_key(res["inverse"], res["mode"], res["atlas"])
    if key in lock["stores"]:
        return
    lock["stores"][key] = {"dir": res["dir"], "graph_run_id": res["graph_run_id"], "config_hash": res["config_hash"],
                           "n_windows": res["n_windows"], "locked_at": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(path, lock)


def open_store(processed: str | Path, inverse: str, mode: str, atlas: str = "dk68", keys: tuple | None = None,
               lock: bool = True) -> tuple[GraphStore, dict]:
    """Verified store (raises ``StoreError`` naming every failed check) and its verification record."""
    res = verify_store(processed, inverse, mode, atlas)
    if not res["passed"]:
        bad = [f"{k}: {c['detail']}" for k, c in res["checks"].items() if not c["passed"]]
        raise StoreError(f"store {inverse}/{mode}/{atlas} ({res['dir']}) refused:\n  " + "\n  ".join(bad))
    if lock:
        lock_store(processed, res)
    return GraphStore.load(res["dir"], keys=keys), res


def overview(processed: str | Path, atlases=("dk68",)) -> pd.DataFrame:
    """One row per (inverse, mode, atlas): directory, state of every check (nothing is changed)."""
    rows = []
    for inv in INVERSES:
        for mode in MODE_NAMES:
            for atlas in atlases:
                try:
                    r = verify_store(processed, inv, mode, atlas)
                    failed = [k for k, c in r["checks"].items() if not c["passed"]]
                    rows.append({"inverse": inv, "mode": mode, "atlas": atlas, "dir": Path(r["dir"]).name,
                                 "passed": r["passed"], "graph_run_id": r["graph_run_id"],
                                 "n_windows": r.get("n_windows"), "failed_checks": "; ".join(failed)})
                except Exception as exc:              # reported, never silent
                    rows.append({"inverse": inv, "mode": mode, "atlas": atlas, "passed": False,
                                 "failed_checks": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)
