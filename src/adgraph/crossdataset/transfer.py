"""Source-only fits and label-blind target predictions (design section 7), checkpointed.

One transfer unit = (direction, inverse, mode, experiment, variant, seed):

1. a FRESH model (``experiments.runner.make_model``: the unchanged definition and fixed protocol) is fitted on the
   SOURCE training participants only (D3 -> D2: the 52 frozen D3 development participants; D2 -> D3: all 23 D2
   participants). Every fitted transform (node-feature scaler, graph construction, class / participant weights,
   classical inner CV for C) lives inside the model and sees only the source training view;
2. the target store is BLINDED (``blind``: every label replaced by -1 / "masked", participant metadata removed) and
   its windows are scored by the fitted model; the unit saves window scores and one mean score per participant,
   without labels; target labels are read only by ``evaluate`` after the freeze;
3. before predicting, the source and target inputs are checked to be the same representation (``check_inputs``:
   tensor shapes, feature names, band order, ROI order, sensor channels, adjacency kind).

Units are saved as ``<results>/<direction>/<inverse>/mode_<m>/<atlas>/<graph>/<experiment>/<variant>/transfer/
seed_<s>/`` with a validated marker; identity = frozen campaign + training-config hash + source and target store
graph_run_ids + source participants + seed + code identity. A completed unit is never recomputed; a unit of another
identity raises ``IncompatibleCheckpoint`` (``force`` moves it to ``_superseded/``).

Source-only development estimates: D2 -> D3: participant-stratified 5-fold CV of D2 (``source_cv``, 5 repeats); D3 ->
D2: the completed release / A2 development CV of the same variant (``release_source_cv``; read only, used only if its
training-config hash and store graph_run_id equal this unit's).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import progress
from ..checkpoint import (IncompatibleCheckpoint, atomic_savez, atomic_write_csv, atomic_write_json, check_unit,
                          code_identity, stable_hash, supersede, write_marker)
from ..experiments.runner import (make_model, model_score_type, run_device, score_threshold, select_hparams,
                                  training_hash, with_store_context)
from ..graphs.dataset import build_dataset
from ..graphs.store import GraphStore, graph_run_id
from . import d2store, namespace, plan

MASK_Y, MASK_LABEL = -1, "masked"


class InputMismatch(RuntimeError):
    """Source and target inputs are not the same representation."""


# ------------------------------------------------------------------------------------------------ stores
def store_spec(inverse: str, c: dict) -> tuple[str, str]:
    """(store inverse, D3 store atlas) of a run inverse: Exp 0 (``sensor``) reads the LCMV stores' sensor arrays
    (D3: the dk68 LCMV store the release's Exp 0 used; sensor arrays are atlas-independent)."""
    return ("lcmv", "dk68") if inverse == "sensor" else (inverse, c["atlas"])


def open_d3(processed, inverse: str, mode: str, c: dict) -> GraphStore:
    from ..stores import open_store

    inv, atlas = store_spec(inverse, c)
    return open_store(processed, inv, mode, atlas, lock=False)[0]      # verified against the release lock; read only


def open_d2(processed, inverse: str, mode: str, c: dict) -> GraphStore:
    inv, _ = store_spec(inverse, c)
    return d2store.open_d2(d2store.store_dir(processed, inv, mode))


def blind(store: GraphStore) -> GraphStore:
    """A copy whose labels are unreadable: windows y = -1, label 'masked'; subjects reduced to identifiers."""
    w = store.windows.copy()
    w["y"], w["label"] = MASK_Y, MASK_LABEL
    s = store.subjects[[c for c in ("key", "dataset", "subject_id", "status", "n_windows_kept")
                        if c in store.subjects]].copy()
    return GraphStore(store.arrays, w, store.bands, store.band_edges, store.roi_names, store.channels, s,
                      {**store.manifest, "blinded": True}, store.root, store.roi_coords)


def source_keys(direction: str, src: GraphStore) -> list[str]:
    """Source training participants: D3 -> D2: the frozen 52 D3 development participants (equal to the store's
    saved split); D2 -> D3: all D2 participants."""
    from ..evaluation.splits import subject_table, xu_partition

    subj = subject_table(src.windows)
    if direction == "D3_to_D2":
        dev, test = xu_partition(subj)
        saved = (src.root / "xu_split.json") if src.root else None
        if saved is not None and saved.exists():
            sp = json.loads(saved.read_text())
            if (sp["development"], sp["test"]) != (dev, test):
                raise RuntimeError("D3 store split differs from the frozen xu_partition")
        if not src.manifest.get("synthetic") and (len(dev), len(test)) != (52, 13):
            raise RuntimeError(f"D3 partition is {len(dev)} / {len(test)}, expected 52 / 13")
        return dev
    return sorted(subj["key"])


def check_inputs(src, tgt, v: dict) -> dict:
    """Raise ``InputMismatch`` unless source and target inputs are the same representation."""
    a, b = src.info, tgt.info
    problems = []
    if src.kind != tgt.kind:
        problems.append(f"kind {src.kind} vs {tgt.kind}")
    for k in ("feature_names", "graph_bands", "roi_names", "connectivity", "adjacency", "band_mode", "source"):
        if a.get(k) != b.get(k):
            problems.append(f"{k}: {a.get(k)} vs {b.get(k)}")
    for k, arr in src.payload.items():
        if k not in tgt.payload or arr.shape[1:] != tgt.payload[k].shape[1:] or arr.dtype != tgt.payload[k].dtype:
            problems.append(f"payload {k}: {arr.shape[1:]} {arr.dtype} vs "
                            f"{tgt.payload[k].shape[1:] if k in tgt.payload else None}")
    if problems:
        raise InputMismatch(f"{v['name']}: " + "; ".join(problems))
    return {"kind": src.kind, "shape": {k: list(x.shape[1:]) for k, x in src.payload.items()},
            "feature_names": a.get("feature_names"), "graph_bands": a.get("graph_bands"),
            "roi_names": a.get("roi_names")}


def check_stores(d3: GraphStore, d2: GraphStore, sensor: bool) -> None:
    if d3.bands != d2.bands or not np.allclose(d3.band_edges, d2.band_edges):
        raise InputMismatch(f"bands {d3.bands} vs {d2.bands}")
    if d3.channels != d2.channels:
        raise InputMismatch(f"sensor channels {d3.channels} vs {d2.channels}")
    if not sensor and d3.roi_names != d2.roi_names:
        raise InputMismatch(f"ROIs {d3.roi_names} vs {d2.roi_names}")
    if set(d3.windows["key"]) & set(d2.windows["key"]):
        raise InputMismatch("participant keys shared between the datasets")


def stores_record(processed, c: dict) -> dict:
    """{"D3|D2/<inverse>/<mode>": graph_run_id} of every store the campaign reads (each verified when opened)."""
    out = {}
    for inv in list(c["inverses"]) + ["sensor"]:
        for mode in c["modes"]:
            out[f"D3/{inv}/{mode}"] = graph_run_id(open_d3(processed, inv, mode, c))
            out[f"D2/{inv}/{mode}"] = graph_run_id(open_d2(processed, inv, mode, c))
    return out


def check_frozen_stores(frozen: dict, inverse: str, mode: str, d3: GraphStore, d2: GraphStore) -> None:
    """The stores are exactly those of the frozen campaign (a rebuilt or replaced store is refused)."""
    for ds, st in (("D3", d3), ("D2", d2)):
        key = f"{ds}/{inverse}/{mode}"
        want = (frozen.get("stores") or {}).get(key)
        if want != graph_run_id(st):
            raise RuntimeError(f"store {key} is {graph_run_id(st)}, the frozen campaign recorded {want}")


# ------------------------------------------------------------------------------------------------ units
def unit_dir(processed, u: dict, c: dict) -> Path:
    inv = u["inverse"]
    atlas = "dk68" if inv == "sensor" else c["atlas"]
    graph = "none" if u["input"] == "classical" else c["graph"]
    return namespace.sub(processed, "results", u["direction"], inv, f"mode_{u['mode']}", atlas, graph,
                         u["experiment"], u["variant"])


def transfer_identity(frozen: dict, v: dict, u: dict, src: GraphStore, tgt: GraphStore, keys: list[str]) -> dict:
    return {"kind": "cross_transfer", "campaign_hash": frozen["campaign_hash"], "direction": u["direction"],
            "training_config_hash": training_hash(v), "context": v["context"],
            "source_graph_run_id": graph_run_id(src), "target_graph_run_id": graph_run_id(tgt),
            "source_participants": stable_hash(sorted(keys)), "seed": int(u["seed"]),
            "code": code_identity("cross_experiment")}


def _datasets(v, src_store, tgt_store):
    src = build_dataset(src_store, v)
    tgt = build_dataset(blind(tgt_store), v)
    return src, tgt


def run_unit(processed, u: dict, c: dict, frozen: dict, d3: GraphStore, d2: GraphStore,
             cache: dict | None = None, force: bool = False) -> str:
    """Fit on the source, score the blinded target; returns 'completed' or 'reused'."""
    src_store, tgt_store = (d3, d2) if u["direction"] == "D3_to_D2" else (d2, d3)
    v0 = next(x for x in plan.variants(c, ("gnn", "reference"))[u["exp"]] if x["variant"] == u["variant"])
    v = with_store_context(plan.contextualize(v0, u["inverse"], u["mode"], c), src_store)
    keys = source_keys(u["direction"], src_store)
    out = unit_dir(processed, u, c) / "transfer" / f"seed_{u['seed']}"
    ident = transfer_identity(frozen, v, u, src_store, tgt_store, keys)
    state, _ = check_unit(out, "transfer", ident)
    if state == "complete":
        return "reused"
    if state == "incompatible":
        if not force:
            raise IncompatibleCheckpoint(f"{out} holds a unit of another identity; use force to supersede it")
        supersede(out, namespace.root(processed))
    elif state == "corrupt":
        supersede(out, namespace.root(processed))
    cache = {} if cache is None else cache
    ck = (u["direction"], u["inverse"], u["mode"], u["experiment"], u["variant"])
    if ck not in cache:
        cache.clear()
        cache[ck] = _datasets(v, src_store, tgt_store)
    src, tgt = cache[ck]
    shapes = check_inputs(src, tgt, v)
    idx = np.flatnonzero(src.meta["key"].isin(keys).to_numpy())
    if set(src.meta["key"].iloc[idx]) != set(keys):
        raise RuntimeError("source participants without windows")
    if set(src.meta["key"]) & set(tgt.meta["key"]):
        raise RuntimeError("participants on both sides of the transfer")
    seed = int(u["seed"])
    t0 = time.perf_counter()
    params, inner = select_hparams(src, idx, v, seed)            # classical: inner CV inside the source only
    from ..experiments.config import apply_hparams

    model = make_model(apply_hparams(v, params), seed, src.info.get("roi_names")).fit(src.view(idx, "train"))
    fit_s = time.perf_counter() - t0
    view = tgt.view(np.arange(len(tgt.meta)), "test")
    if "y" in view.meta and (view.meta["y"] != MASK_Y).any():
        raise RuntimeError("target labels are not masked")
    win = view.meta[["window_id", "key", "subject_id", "dataset"]].copy()
    if v["input"] == "graph":
        score, bw = model.predict_score(view, return_band_weights=True)
        for i, b in enumerate(tgt.info["graph_bands"]):
            win[f"band_weight_{b}"] = bw[:, i]
    else:
        score = model.predict_score(view)
    win["score"] = score
    part = (win.groupby("key", sort=True).agg(dataset=("dataset", "first"), subject_id=("subject_id", "first"),
                                              score=("score", "mean"), n_windows=("score", "size")).reset_index())
    training = {"seed": seed, "selected": params, "inner_cv_scores": inner, "fit_seconds": fit_s,
                "n_source_participants": len(keys), "n_source_windows": int(len(idx)),
                "n_target_participants": int(part.shape[0]), "n_target_windows": int(len(win)),
                "n_parameters": model.n_parameters(), "score_type": model_score_type(v),
                "threshold": score_threshold(v), "device": run_device(v), "inputs": shapes,
                "variant_config": v, "source_participants": keys,
                "finished_at": datetime.now(timezone.utc).isoformat()}
    if hasattr(model, "training_summary"):
        training["training"] = model.training_summary()
    files = [atomic_write_csv(out / "target_windows_blind.csv", win),
             atomic_write_csv(out / "target_participants_blind.csv", part),
             atomic_write_json(out / "training.json", training)]
    if hasattr(model, "state_arrays"):
        files.append(atomic_savez(out / "model.npz", **model.state_arrays()))
    write_marker(out, "transfer", ident, files)
    return "completed"


# ------------------------------------------------------------------------------------------------ source-only CV
def d2_source_cv(processed, u: dict, c: dict, frozen: dict, d2: GraphStore, force: bool = False,
                 cv: dict | None = None, subdir: str = "source_cv") -> dict:
    """Participant-stratified K-fold CV inside D2 (source of D2 -> D3): primary ``source_cv.D2`` (5 folds x 1
    repeat), or the stability analysis (``stability.d2_source_cv``, saved in ``source_cv_stability``); one checkpoint
    per fold; returns the metrics (also saved as ``<subdir>/metrics.json``)."""
    from ..evaluation.metrics import aggregate_subjects, auc_delong, binary_metrics
    from ..evaluation.splits import assert_disjoint, stratified_subject_folds, subject_table
    from ..experiments.config import apply_hparams

    sc = cv or c["source_cv"]["D2"]
    v0 = next(x for x in plan.variants(c, ("gnn", "reference"))[u["exp"]] if x["variant"] == u["variant"])
    v = with_store_context(plan.contextualize(v0, u["inverse"], u["mode"], c), d2)
    out = unit_dir(processed, {**u, "direction": "D2_to_D3"}, c) / subdir
    base = {"kind": "cross_source_cv", "campaign_hash": frozen["campaign_hash"], "training_config_hash":
            training_hash(v), "context": v["context"], "graph_run_id": graph_run_id(d2), "cv": sc,
            "code": code_identity("cross_experiment")}
    data = build_dataset(d2, v)
    subj = subject_table(data.meta)
    thr = score_threshold(v)
    preds = []
    for r in range(int(sc["repeats"])):
        for f, (tr, te) in enumerate(stratified_subject_folds(subj, int(sc["folds"]), int(sc["seed"]) + r)):
            name = f"fold_r{r:03d}_f{f:02d}"
            ident = {**base, "unit": name, "split": stable_hash({"train": sorted(tr), "test": sorted(te)})}
            st, _ = check_unit(out / "units", name, ident)
            if st == "incompatible" and not force:
                raise IncompatibleCheckpoint(f"{out / 'units' / name} has another identity")
            if st == "complete":
                preds.append(pd.read_csv(out / "units" / f"{name}.pred.csv"))
                continue
            keys = data.meta["key"].to_numpy()
            tr_i, te_i = np.flatnonzero(np.isin(keys, tr)), np.flatnonzero(np.isin(keys, te))
            assert_disjoint(data.meta.iloc[tr_i], data.meta.iloc[te_i])
            seed = int(v["evaluation"]["seed"]) + 1000 * r + f
            params, _ = select_hparams(data, tr_i, v, seed)
            model = make_model(apply_hparams(v, params), seed, data.info.get("roi_names")).fit(data.view(tr_i, "train"))
            test = data.view(te_i, "test")
            p = test.meta[["window_id", "key", "dataset"]].assign(y=test.y, score=model.predict_score(test),
                                                                  repeat=r, fold=f)
            write_marker(out / "units", name, ident, [atomic_write_csv(out / "units" / f"{name}.pred.csv", p)])
            preds.append(p)
            progress.advance(timed=False)
    pred = pd.concat(preds, ignore_index=True)
    part = aggregate_subjects(pred, by=("repeat",))
    per = [{"repeat": int(r), **binary_metrics(g["y"], g["score"], thr)} for r, g in part.groupby("repeat")]
    per = pd.DataFrame(per)
    mean_part = part.groupby("key").agg(y=("y", "first"), score=("score", "mean"), dataset=("dataset", "first"))
    m = {"source": "D2", "status": "computed", "cv": sc, "threshold": thr,
         "auc_mean": float(per["roc_auc"].mean()), "auc_sd": float(per["roc_auc"].std(ddof=1)) if len(per) > 1
         else float("nan"), "balanced_accuracy_mean": float(per["balanced_accuracy"].mean()),
         "balanced_accuracy_sd": float(per["balanced_accuracy"].std(ddof=1)) if len(per) > 1 else float("nan"),
         "auc_repeat_averaged_scores": auc_delong(mean_part["y"], mean_part["score"]),
         "per_repeat": per.to_dict("records"), "n_participants": int(len(mean_part))}
    atomic_write_csv(out / "predictions_participants.csv", part)
    atomic_write_json(out / "metrics.json", m)
    return m


def release_source_cv(processed, u: dict, c: dict, d3: GraphStore) -> dict:
    """The completed release (or amendment A2) development-CV result of the same variant and context, read only;
    'available' only if its training-config hash and D3 store graph_run_id equal this campaign's."""
    from .. import release
    from ..experiments import amendment
    from ..experiments.runner import result_dir

    v0 = next(x for x in plan.variants(c, ("gnn", "reference"))[u["exp"]] if x["variant"] == u["variant"])
    v = with_store_context(plan.contextualize(v0, u["inverse"], u["mode"], c), d3)
    root = amendment.results_dir(processed, "A2") if u["exp"] == "exp8a" else release.results_dir(processed)
    d = result_dir(root, v, "xu_dev_cv")
    rec = {"source": "D3", "dir": str(d), "status": "missing"}
    if not (d / "final.done.json").exists() or not (d / "_identity.json").exists():
        return rec
    ident = json.loads((d / "_identity.json").read_text())["identity"]
    want = {"training_config_hash": training_hash(v), "graph_run_id": graph_run_id(d3), "setting": "xu_dev_cv",
            "experiment": v["experiment"], "variant": v["variant"], "context": v["context"]}
    for k, val in want.items():                               # decision D9: the exact identities must match
        if ident.get(k) != val:
            return {**rec, "status": f"mismatch: {k}"}
    m = json.loads((d / "metrics.json").read_text())
    return {**rec, "status": "available", "git_commit": (ident.get("code") or {}).get("git_commit"),
            "auc": m["participant"]["roc_auc"], "balanced_accuracy": m["participant"]["balanced_accuracy"],
            "auc_delong": m.get("participant_auc_delong"), "threshold": m.get("threshold"),
            "predictions": str(d / "predictions_participants.csv")}


# ------------------------------------------------------------------------------------------------ driver
def run(processed, c: dict, stage: str = "primary", experiments=None, variants=None, directions=None,
        inverses=None, modes=None, synthetic: bool = False, force: bool = False) -> pd.DataFrame:
    """Every selected unit of the frozen campaign (resumable); D2 source CV for the D2 -> D3 units. Real data need
    the recorded ``train`` authorization."""
    frozen = plan.require_frozen(processed, c)
    if not synthetic:
        namespace.require_authorized(processed, "train")
        if stage == "stability":
            namespace.require_authorized(processed, c["stability"]["authorization"])
    us = [u for u in plan.units(c, stage, experiments, variants)
          if (not directions or u["direction"] in directions) and (not inverses or u["inverse"] in inverses)
          and (not modes or u["mode"] in modes)]
    rows, stores, cache = [], {}, {}
    with progress.reporter(len(us), f"cross-dataset units ({stage})", unit="units") as bar:
        for u in us:
            sk = (u["inverse"], u["mode"])
            if sk not in stores:
                stores.clear()
                d3, d2 = open_d3(processed, u["inverse"], u["mode"], c), open_d2(processed, u["inverse"], u["mode"], c)
                check_frozen_stores(frozen, u["inverse"], u["mode"], d3, d2)
                check_stores(d3, d2, u["inverse"] == "sensor")
                stores[sk] = (d3, d2)
            d3, d2 = stores[sk]
            bar.set(unit=f"{u['direction']} {u['inverse']} {u['mode']} {u['experiment']} {u['variant']} s{u['seed']}")
            try:
                state = run_unit(processed, u, c, frozen, d3, d2, cache, force)
                if u["direction"] == "D2_to_D3" and stage == "primary":
                    d2_source_cv(processed, u, c, frozen, d2, force)
                elif u["direction"] == "D2_to_D3" and u["seed"] == c["stability"]["seeds"][0]:
                    d2_source_cv(processed, u, c, frozen, d2, force, c["stability"]["d2_source_cv"],
                                 "source_cv_stability")
            except Exception as exc:                          # recorded; the next unit continues
                state = f"failed: {type(exc).__name__}: {exc}"[:500]
            rows.append({**{k: u[k] for k in ("direction", "inverse", "mode", "experiment", "variant", "seed")},
                         "state": state})
            bar.advance()
    t = pd.DataFrame(rows)
    atomic_write_csv(namespace.sub(processed, "results", f"_run_status_{stage}.csv"), t)
    return t


def status(processed, c: dict, stage: str = "primary") -> pd.DataFrame:
    """Completed / pending transfer units (markers only; nothing is run or loaded)."""
    rows = []
    for u in plan.units(c, stage):
        d = unit_dir(processed, u, c) / "transfer" / f"seed_{u['seed']}"
        rows.append({**{k: u[k] for k in ("direction", "inverse", "mode", "experiment", "variant", "seed")},
                     "state": "completed" if (d / "transfer.done.json").exists() else "pending"})
    return pd.DataFrame(rows)
