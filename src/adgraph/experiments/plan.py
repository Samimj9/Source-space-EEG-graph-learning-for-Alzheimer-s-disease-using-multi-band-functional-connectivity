"""Run every experiment of a committed run plan, resumably, in priority order.

The plan (``configs/run_plan.yaml``) lists experiment configs in the order they should be
run: primary models first, so the most important results exist even if a long run is
stopped part-way. Each (variant, setting) is one ``run_variant`` call, so completed work
is reused and interrupted work continues (see ``runner``). One failing unit does not stop
the others: its error is recorded and the plan moves on. Cross-dataset settings are
skipped, with the reason recorded, when the graph run does not pass the DESIGN.md §7 gate.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from ..checkpoint import IncompatibleCheckpoint, atomic_write_csv
from ..evaluation.splits import CROSS_SETTINGS
from ..graphs.dataset import build_dataset
from ..graphs.store import GraphStore
from .config import expand_variants, load_experiment, set_dotted
from .gate import gate_failures
from .runner import experiment_status, run_variant

log = logging.getLogger(__name__)

STATUS_COLUMNS = ["order", "config", "experiment", "variant", "setting", "state", "subject_roc_auc",
                  "subject_balanced_accuracy", "message", "updated"]


def variant_matches(v: dict, filters) -> bool:
    """True if a filter equals whole ``__``-separated tokens of the variant name (in order).

    ``plv`` matches ``plv__full`` but not ``ciplv__full``; ``plv__full`` matches exactly."""
    tokens = v["variant"].split("__")
    for f in filters:
        want = [t for t in str(f).strip("_").split("__") if t]
        if any(tokens[i:i + len(want)] == want for i in range(len(tokens) - len(want) + 1)):
            return True
    return False


SETTING_PASSES = (("within_D1", "within_D2"), ("pooled",), ("cross_D1_to_D2", "cross_D2_to_D1"))


def experiment_entries(config: str | Path, config_root: str | Path = ".") -> list[dict]:
    """Plan entries for ONE experiment config: all its variants and settings, in passes
    (within-dataset first, then pooled, then cross-dataset). Nothing else is included."""
    cfg = load_experiment(Path(config_root) / config)
    entries = []
    for p in SETTING_PASSES:
        settings = [x for x in p if x in cfg["settings"]]
        if settings:
            entries.append({"config": str(config), "settings": settings})
    return entries


def load_plan(path: str | Path) -> list[dict]:
    plan = yaml.safe_load(Path(path).read_text())
    entries = plan["experiments"]
    for e in entries:
        if "config" not in e:
            raise ValueError(f"plan entry without 'config': {e}")
    return entries


def plan_units(entries: list[dict], config_root: str | Path = ".") -> list[tuple[int, str, dict, str]]:
    """(order, config path, variant, setting) for every unit of the plan, in run order."""
    units, order = [], 0
    for e in entries:
        cfg = load_experiment(Path(config_root) / e["config"])
        for key, value in (e.get("overrides") or {}).items():
            cfg = set_dotted(cfg, key, value)
        variants = expand_variants(cfg)
        if e.get("variants"):
            variants = [v for v in variants if variant_matches(v, e["variants"])]
        for v in variants:
            for s in e.get("settings") or cfg["settings"]:
                units.append((order, e["config"], v, s))
                order += 1
    return units


def _metric(summary: dict, name: str):
    try:
        return float(summary["subject_level"]["all"][name]["mean"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def run_plan(entries: list[dict], store: GraphStore, out_dir: str | Path, config_root: str | Path = ".",
             resume: bool = True, allow_ephemeral: bool = False,
             status_path: str | Path | None = None) -> pd.DataFrame:
    """Run (or resume) every unit; returns and saves the status table (default
    ``<out_dir>/_plan_status.csv``)."""
    out_dir = Path(out_dir)
    units = plan_units(entries, config_root)
    gate = gate_failures(store)
    rows, data_cache = [], {}
    status_path = Path(status_path) if status_path else out_dir / "_plan_status.csv"
    for i, (order, config, v, setting) in enumerate(units):
        row = {"order": order, "config": config, "experiment": v["experiment"], "variant": v["variant"],
               "setting": setting, "subject_roc_auc": float("nan"), "subject_balanced_accuracy": float("nan"),
               "message": ""}
        allow = v["evaluation"].get("allow_ungated_cross_dataset", False)
        if setting in CROSS_SETTINGS and gate and not allow:
            row.update(state="skipped", message="cross-dataset gate not passed: " + "; ".join(gate))
        else:
            log.info("[%d/%d] %s / %s", i + 1, len(units), v["name"], setting)
            try:
                if v["name"] not in data_cache:
                    data_cache.clear()                    # one variant's dataset in memory at a time
                    data_cache[v["name"]] = build_dataset(store, v)
                summary = run_variant(store, v, setting, out_dir, data=data_cache[v["name"]], resume=resume,
                                      allow_ephemeral=allow_ephemeral)
                row.update(state="completed", subject_roc_auc=_metric(summary, "roc_auc"),
                           subject_balanced_accuracy=_metric(summary, "balanced_accuracy"))
            except IncompatibleCheckpoint as exc:
                row.update(state="incompatible", message=str(exc)[:500])
                log.error("%s / %s: %s", v["name"], setting, exc)
            except Exception as exc:                      # recorded; KeyboardInterrupt etc. propagate
                row.update(state="failed", message=f"{type(exc).__name__}: {exc}"[:500])
                log.error("%s / %s failed:\n%s", v["name"], setting, traceback.format_exc())
        row["updated"] = datetime.now(timezone.utc).isoformat()
        rows.append(row)
        atomic_write_csv(status_path, pd.DataFrame(rows, columns=STATUS_COLUMNS))
    return pd.DataFrame(rows, columns=STATUS_COLUMNS)


def plan_status(entries: list[dict], store: GraphStore, out_dir: str | Path,
                config_root: str | Path = ".") -> pd.DataFrame:
    """Progress of every unit of the plan, read from the checkpoints (nothing is computed)."""
    gate = gate_failures(store)
    rows = []
    for order, config, v, setting in plan_units(entries, config_root):
        if setting in CROSS_SETTINGS and gate and not v["evaluation"].get("allow_ungated_cross_dataset", False):
            rows.append({"order": order, "config": config, "experiment": v["experiment"], "variant": v["variant"],
                         "setting": setting, "state": "skipped (gate)"})
            continue
        st = experiment_status(store, v, setting, out_dir)
        rows.append({"order": order, "config": config, **st})
    return pd.DataFrame(rows)


def experiments_overview(configs: list[str], store: GraphStore, out_dir: str | Path,
                         config_root: str | Path = ".") -> pd.DataFrame:
    """One row per experiment config: how many (variant, setting) results are completed,
    partial, not started, failed or skipped by the gate (read from checkpoints only)."""
    rows = []
    for config in configs:
        st = plan_status(experiment_entries(config, config_root), store, out_dir, config_root)
        counts = st["state"].value_counts().to_dict()
        rows.append({"config": config, "experiment": st["experiment"].iloc[0] if len(st) else "",
                     "units": len(st), **{k: counts.get(k, 0) for k in
                                          ("completed", "partial", "not started", "failed", "incompatible",
                                           "skipped (gate)")}})
    return pd.DataFrame(rows)
