"""Runtime estimate for one experiment, measured on the machine that will run it.

1. **Exact work count.** For every (variant, setting) the splits the protocol will use are
   generated (same seeds as the runner) and every model fit is listed with its number of
   training windows: outer folds x repeats x (inner folds x grid size + 1 refit).
2. **Measured cost per window.** One real fit (full epochs, first grid point) is timed per
   distinct architecture ("cost class": classical model / GNN encoder, band mode, shared
   encoder, number of bands) on a real training fold, after a 1-epoch warm-up. Fit time is
   scaled linearly with the number of training windows.
3. **Measured checkpoint cost.** A few checkpoint units are written to the output folder
   (Google Drive in Colab) and timed; every inner trial and outer fold writes one unit.
4. **Remaining work.** Outer folds already completed (validated checkpoints) are subtracted.

The estimate is an approximation (fit time is not exactly linear in windows; inner-trial
caches of an unfinished fold are not credited), but it is measured, not guessed.
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from ..checkpoint import atomic_write_csv, write_marker
from ..evaluation.splits import CROSS_SETTINGS, fold_indices, inner_folds
from ..graphs.dataset import _meta, build_dataset, target_column
from ..graphs.store import GraphStore
from .config import apply_hparams, hparam_grid, set_dotted
from .gate import gate_failures
from .runner import _splits, _strata, experiment_status, make_model, run_device


def cost_class(v: dict) -> str:
    if v["input"] == "classical":
        return f"classical:{v['model']['name']}:{v['variant']}"
    g, m = v["graph"], v["model"]
    return (f"gnn:{m.get('encoder', 'gcn')}:{g.get('band_mode', 'multi')}:"
            f"{'shared' if m.get('shared_encoder', True) else 'per-band'}:{len(g.get('bands', []))}bands")


def fit_plan(meta: pd.DataFrame, v: dict, setting: str) -> tuple[list[int], int]:
    """Training-window counts of every fit, and the number of checkpoint units written."""
    target = target_column(v.get("target", "label"))
    sub, m, splits = _splits(meta, v, setting, target)
    n_grid, ev = len(hparam_grid(v)), v["evaluation"]
    fits, units = [], 0
    for sp in splits:
        tr, _ = fold_indices(m, sp)
        if n_grid > 1:
            seed = int(ev["seed"] + 1000 * sp.repeat + sp.fold)
            tm = m.iloc[tr].reset_index(drop=True)
            strata = _strata(setting) if setting == "pooled" else ("y",)
            for itr, _ in inner_folds(tm, ev["inner_k"], seed, strata, target):
                fits += [len(itr)] * n_grid
                units += n_grid if v["input"] == "graph" else 0     # classical trials are not cached
        fits.append(len(tr))
        units += 1
    return fits, units


def time_fit(store: GraphStore, v: dict, setting: str) -> tuple[float, int, str]:
    """(seconds, training windows, device) of one real fit on the first outer training fold."""
    data = build_dataset(store, v)
    sub, m, splits = _splits(data.meta, v, setting, data.target, n_repeats=1)
    tr, _ = fold_indices(m, splits[0])
    train = data.view(sub[tr], "train")
    vp = apply_hparams(v, hparam_grid(v)[0])
    if v["input"] == "graph":                                 # warm-up (CUDA init, kernel caches)
        make_model(set_dotted(vp, "training.epochs", 1), 0).fit(train)
    t0 = time.perf_counter()
    make_model(vp, 0).fit(train)
    return time.perf_counter() - t0, len(tr), run_device(v)


def time_checkpoint_unit(directory: str | Path, n: int = 5) -> float:
    """Seconds to write one checkpoint unit (payload + validated marker) in ``directory``."""
    probe = Path(directory) / f"_io_probe_{uuid.uuid4().hex[:8]}"
    df = pd.DataFrame({"window_id": [f"w{i}" for i in range(120)], "key": "D1-1", "score": np.linspace(0, 1, 120)})
    try:
        t0 = time.perf_counter()
        for i in range(n):
            atomic_write_csv(probe / f"u{i}.val.csv", df)
            write_marker(probe, f"u{i}", {"probe": i}, [f"u{i}.val.csv"])
        return (time.perf_counter() - t0) / n
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def estimate_experiment(store: GraphStore, variants: list[dict], settings_of, out_dir: str | Path,
                        timing_store: GraphStore | None = None) -> tuple[pd.DataFrame, dict]:
    """Per (variant, setting) estimate; ``settings_of(v)`` gives the settings to run.

    ``store`` may be loaded without arrays (meta only) for counting; ``timing_store`` (full
    arrays) is used for the timing fits. Returns (table, info)."""
    timing_store = timing_store or store
    gate = gate_failures(store)
    meta_cache, rate, devices, rows = {}, {}, set(), []
    io_s = time_checkpoint_unit(out_dir)
    for v in variants:
        tkey = v.get("target", "label")
        if tkey not in meta_cache:
            meta_cache[tkey] = _meta(store, tkey)
        meta = meta_cache[tkey]
        for s in settings_of(v):
            row = {"variant": v["variant"], "setting": s, "cost_class": cost_class(v)}
            if s in CROSS_SETTINGS and gate and not v["evaluation"].get("allow_ungated_cross_dataset", False):
                rows.append({**row, "state": "skipped (gate)", "fits": 0, "est_hours": 0.0, "remaining_hours": 0.0})
                continue
            fits, units = fit_plan(meta, v, s)
            cc = row["cost_class"]
            if cc not in rate:
                calib = next((x for x in settings_of(v) if x == "pooled"), s)
                secs, n_win, dev = time_fit(timing_store, v, calib)
                rate[cc] = secs / max(n_win, 1)
                devices.add(dev)
            compute_h = rate[cc] * sum(fits) / 3600
            io_h = io_s * units / 3600
            st = experiment_status(store, v, s, out_dir)
            done = st["folds_completed"] / st["folds_total"] if st["folds_total"] else 0.0
            rows.append({**row, "state": st["state"], "fits": len(fits), "units": units,
                         "folds_done": f"{st['folds_completed']}/{st['folds_total']}",
                         "compute_hours": compute_h, "checkpoint_hours": io_h, "est_hours": compute_h + io_h,
                         "remaining_hours": (compute_h + io_h) * (1 - done)})
    table = pd.DataFrame(rows)
    info = {"devices": sorted(devices), "checkpoint_seconds_per_unit": io_s,
            "seconds_per_training_window": rate,
            "total_hours": float(table["est_hours"].sum()), "remaining_hours": float(table["remaining_hours"].sum())}
    return table, info


def by_variant(table: pd.DataFrame) -> pd.DataFrame:
    """Totals per variant (the experiment's major configurations)."""
    t = table[table["fits"] > 0]
    return (t.groupby(["variant", "cost_class"], sort=False)[["fits", "est_hours", "remaining_hours"]]
            .sum().reset_index())
