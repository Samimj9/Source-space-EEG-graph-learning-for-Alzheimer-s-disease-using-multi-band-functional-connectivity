"""Run every variant of ONE experiment config in one run context, resumably (what each final notebook calls).

    run_experiment("configs/final/experiments/exp5_multibranch.yaml", processed, inverse="lcmv", mode="B")

1. the store of (inverse, mode, atlas) is opened through ``stores.open_store`` (verified and locked);
2. the participant QC of that store is loaded or computed once (``qc.report.require_qc``; the experiment stops
   while a category A finding has no decision);
3. holdout units (``xu_holdout``) of a real store run only if they belong to the frozen final analysis
   (``decisions.require_frozen``; release.HOLDOUT_POLICY);
4. every variant x setting is run with ``runner.run_variant`` (completed results are reused; one failing unit is
   recorded and the others continue); the status table is written after every unit.
Nothing else is started: one experiment per call.
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import progress, release
from ..checkpoint import IncompatibleCheckpoint, atomic_write_csv
from ..graphs.dataset import build_dataset
from ..graphs.store import GraphStore
from ..memory import MemoryTracker
from .config import expand_variants, load_experiment, with_context
from .runner import experiment_status, run_variant, with_store_context

log = logging.getLogger(__name__)
STATUS_COLUMNS = ["experiment", "variant", "setting", "inverse", "mode", "atlas", "graph", "state", "participant_auc",
                  "mean_fold_auc", "balanced_accuracy", "seconds", "mem_peak_mb", "message", "updated"]


class EmptySchedule(RuntimeError):
    """A configuration scheduled no unit."""


def variant_matches(v: dict, filters) -> bool:
    """True if a filter equals whole ``__``-separated tokens of the variant name (in order)."""
    tokens = v["variant"].split("__")
    for f in filters:
        want = [t for t in str(f).strip("_").split("__") if t]
        if any(tokens[i:i + len(want)] == want for i in range(len(tokens) - len(want) + 1)):
            return True
    return False


def experiment_variants(config: str | Path, inverse: str, mode: str, atlas: str = "dk68", graph: str | None = None,
                        variants=None) -> list[dict]:
    vs = [with_context(v, inverse, mode, atlas, graph) for v in expand_variants(load_experiment(config))]
    if variants:
        vs = [v for v in vs if variant_matches(v, variants)]
    if not vs:
        raise EmptySchedule(f"{config}: no variant selected ({variants})")
    return vs


def units(variants: list[dict], settings=None) -> list[tuple[dict, str]]:
    out = [(v, s) for v in variants for s in v["settings"] if not settings or s in settings]
    if not out:
        raise EmptySchedule("zero units (variants x settings)")
    return out


def open_context(processed, inverse: str, mode: str, atlas: str = "dk68") -> tuple[GraphStore, dict, dict]:
    """(store, store verification record, QC report) of a run context; raises when the store or its QC is not
    usable."""
    from ..qc.report import require_qc
    from ..stores import open_store, store_inverse

    store, record = open_store(processed, inverse, mode, atlas)
    qc = require_qc(store, processed, store_inverse(inverse), mode, atlas)
    return store, record, qc


def run_experiment(config: str | Path, processed: str | Path, inverse: str, mode: str, atlas: str = "dk68",
                   graph: str | None = None, variants=None, settings=None, results_root: str | Path | None = None,
                   store: GraphStore | None = None, store_record: dict | None = None, qc_report: dict | None = None,
                   allow_ephemeral: bool = False, status_path: str | Path | None = None,
                   frozen_path: str | Path | None = None) -> pd.DataFrame:
    vs = experiment_variants(config, inverse, mode, atlas, graph, variants)
    todo = units(vs, settings)
    if store is None:
        store, store_record, qc_report = open_context(processed, inverse, mode, atlas)
    elif qc_report is None and not store.manifest.get("synthetic"):
        raise ValueError("a real store needs its participant QC report (plan.open_context); QC is never skipped")
    holdout = [v for v, s in todo if s == "xu_holdout"]
    if holdout and not store.manifest.get("synthetic"):           # the holdout only in the frozen final analysis
        from ..provenance import git_commit
        from .decisions import require_frozen

        require_frozen(processed, holdout, git_commit(), frozen_path)
    results_root = Path(results_root or release.results_dir(processed))
    ctx = vs[0]["context"]
    status_path = Path(status_path) if status_path else (
        results_root / ctx["inverse"] / f"mode_{ctx['mode']}" / ctx["atlas"] / ctx["graph"] / vs[0]["experiment"]
        / "_run_status.csv")
    total = done = 0
    before = {}
    for v, s in todo:
        st = experiment_status(store, v, s, results_root)
        before[(v["name"], s)] = st["state"]
        total += st["folds_total"]
        done += st["folds_completed"] if st["state"] != "incompatible" else 0
    rows, cache = [], {}
    with progress.reporter(total, vs[0]["experiment"], initial=done, unit="folds", eta=False,
                           counts={"units done": 0, "resumed": 0, "failed": 0, "pending": len(todo)}) as bar:
        for v, s in todo:
            row = {"experiment": v["experiment"], "variant": v["variant"], "setting": s, **v["context"],
                   "participant_auc": float("nan"), "mean_fold_auc": float("nan"),
                   "balanced_accuracy": float("nan"), "seconds": float("nan"), "mem_peak_mb": float("nan"),
                   "message": ""}
            bar.set(variant=v["variant"], setting=s)
            t0 = time.perf_counter()
            try:
                with MemoryTracker() as m:
                    if v["name"] not in cache:
                        cache.clear()                                  # one variant's inputs in memory at a time
                        cache[v["name"]] = build_dataset(store, with_store_context(v, store))
                    met = run_variant(store, v, s, results_root, data=cache[v["name"]], store_record=store_record,
                                      qc_report=qc_report, allow_ephemeral=allow_ephemeral)
                row.update(state="completed", participant_auc=met["participant"]["roc_auc"],
                           mean_fold_auc=(met.get("folds") or {}).get("mean_auc", float("nan")),
                           balanced_accuracy=met["participant"]["balanced_accuracy"],
                           mem_peak_mb=m.record()["mem_peak_mb"])
            except IncompatibleCheckpoint as exc:
                row.update(state="incompatible", message=str(exc)[:500])
                log.error("%s / %s: %s", v["name"], s, exc)
            except Exception as exc:                                   # recorded; the other units continue
                row.update(state="failed", message=f"{type(exc).__name__}: {exc}"[:500])
                log.error("%s / %s failed:\n%s", v["name"], s, traceback.format_exc())
            row["seconds"] = round(time.perf_counter() - t0, 1)
            row["updated"] = datetime.now(timezone.utc).isoformat()
            key = "units done" if row["state"] == "completed" else "failed"
            if key == "units done" and before.get((v["name"], s)) == "completed":
                key = "resumed"
            bar.counts[key] += 1
            bar.counts["pending"] -= 1
            bar.set()
            rows.append(row)
            atomic_write_csv(status_path, pd.DataFrame(rows, columns=STATUS_COLUMNS))
        bar.set(variant=None, setting=None)
    return pd.DataFrame(rows, columns=STATUS_COLUMNS)


def status_table(config: str | Path, processed: str | Path, inverse: str, mode: str, atlas: str = "dk68",
                 graph: str | None = None, results_root: str | Path | None = None,
                 store: GraphStore | None = None, settings=None) -> pd.DataFrame:
    """Progress of every unit, read from the checkpoints (nothing is computed or locked)."""
    from ..stores import store_dir

    vs = experiment_variants(config, inverse, mode, atlas, graph)
    if store is None:
        store = GraphStore.load(store_dir(processed, inverse, mode, atlas), keys=())
    results_root = Path(results_root or release.results_dir(processed))
    return pd.DataFrame([{**experiment_status(store, v, s, results_root), **v["context"]}
                         for v, s in units(vs, settings)])
