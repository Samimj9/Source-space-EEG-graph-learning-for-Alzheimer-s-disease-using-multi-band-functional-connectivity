"""Run one experiment variant in one setting of the final protocol (DESIGN.md v2.0, docs/final/framework.md).

Per outer split (participant-level; ``evaluation.splits``):
    train view (role "train", the outer training participants only)
      -> classical: C chosen by participant-level inner CV inside the training view; GNN: the fixed configuration
      -> model.fit(train view)      (every fitted transform lives in the model: scaling, graph construction, ...)
      -> fold QC: outlier thresholds fitted on the training participants; flags recorded, nobody removed
    test view (role "test", built only after fitting)
      -> window scores -> participant score = mean of its window scores
         (P(AD) with threshold 0.5, or the SVM decision_function margin with threshold 0)

Outputs per (variant, setting) directory
``<results>/<inverse>/mode_<A|B>/<atlas>/<graph>/<experiment>/<variant>/<setting>/``:
    predictions_windows.csv        repeat, fold, window_id, key, subject_id, y, score [, band weights]
    predictions_participants.csv   repeat, fold, key, label, y, score, n_windows, predicted, correct
    metrics.json                   ``evaluation.reporting.evaluate_setting`` (pooled OOF AUC, fold AUCs, balanced
                                   accuracy, sensitivity, specificity, confusion matrix, DeLong / bootstrap / Wilson CIs)
    fold_metrics.csv               participant metrics within each development fold
    folds.csv                      participant -> (repeat, fold, side)
    training.csv                   per outer fit: parameters, fit seconds, classical inner-CV selection, GNN training
                                   history (loss curve, gradient norm, training AUC), graph-construction statistics,
                                   fold-QC summary, seed, split id, compute fingerprint
    fold_qc.csv                    fold-level QC flags of every participant (thresholds from that fold's training set)
    manifest.json                  resolved variant config, release manifest, store verification, QC summary
    units/, status.csv, _identity.json, final.done.json   checkpoints (methodology log entries 31-32)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import progress, release
from ..checkpoint import (CODE_VERSIONS, IncompatibleCheckpoint, StatusTable, atomic_savez, atomic_write_csv,
                          atomic_write_json, atomic_write_text, check_unit, code_identity, identity_diff, stable_hash,
                          supersede, write_marker)
from ..config import config_hash
from ..evaluation.metrics import aggregate_subjects, binary_metrics, permutation_p_value
from ..evaluation.reporting import evaluate_setting
from ..evaluation.splits import SETTINGS, fold_indices, inner_folds, make_splits, setting_subset
from ..graphs.dataset import Dataset, build_dataset
from ..graphs.store import GraphStore, graph_run_id, require_complete_cohort
from ..provenance import compute_environment
from ..storage import check_persistent
from .config import apply_hparams, hparam_grid

log = logging.getLogger(__name__)


def model_score_type(v: dict) -> str:
    """Score scale produced by a variant's model: 'probability' or 'decision_function'."""
    if v["input"] == "classical":
        from ..models.classical import score_type

        return score_type(v["model"]["name"], v["model"].get("calibration", "none"))
    return "probability"


def score_threshold(v: dict) -> float:
    """Fixed decision threshold on the model's score scale (never tuned)."""
    ev = v["evaluation"]
    return ev.get("decision_threshold", 0.0) if model_score_type(v) == "decision_function" else ev["threshold"]


def run_device(v: dict) -> str:
    if v["input"] == "classical":
        return "cpu"
    if v.get("device"):
        return v["device"]
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def with_store_context(v: dict, store: GraphStore) -> dict:
    """Store facts an architecture needs at build time (part of the identity): Exp 8 derives its fixed anatomical
    adjacency from the store's ROI names and order."""
    if (v.get("model") or {}).get("architecture") == "xu_dual_gcn":
        v = {**v, "model": {**v["model"], "roi_names": list(store.roi_names)}}
    return v


def make_model(v: dict, seed: int, roi_names: list | None = None):
    if v["input"] == "classical":
        from ..models.classical import ClassicalModel

        m = v["model"]
        return ClassicalModel(m["name"], m.get("params", {}), v.get("preprocessing", {}), seed,
                              m.get("calibration", "none"), m.get("calibration_folds", 3))
    from ..models.trainer import GNNModel

    return GNNModel(v["model"], v["training"], v["node_features"].get("scaling", "node"), v.get("graph_construction"),
                    seed, v.get("device"), roi_names)


# ====================================================================== checkpointing
EVAL_ONLY_KEYS = ("n_repeats", "n_bootstrap", "permutation")


def training_hash(v: dict) -> str:
    """Hash of everything that affects what a fold trains / predicts (incl. the run context and the graph
    construction). Evaluation-only settings are excluded, so e.g. a different bootstrap count reuses the folds."""
    vv = {k: val for k, val in v.items() if k not in ("name", "variant", "settings")}
    vv["evaluation"] = {k: val for k, val in v["evaluation"].items() if k not in EVAL_ONLY_KEYS}
    return config_hash(vv)


def split_id(train_keys, test_keys) -> str:
    return stable_hash({"train": sorted(train_keys), "test": sorted(test_keys)})


class ExperimentCheckpoint:
    """Checkpoint directory of one (variant, setting): units are reused only if their identity matches."""

    def __init__(self, out: Path, identity: dict):
        self.out, self.identity = out, identity
        self.units = out / "units"
        self.status = StatusTable(out / "status.csv")

    @classmethod
    def open(cls, out: Path, identity: dict, root: Path, resume: bool = True, force: bool = False):
        if out.exists() and any(out.iterdir()):
            if force:
                moved = supersede(out, root)
                log.warning("--force: previous outputs moved to %s", moved)
            else:
                ident_file = out / "_identity.json"
                if not ident_file.exists():
                    raise IncompatibleCheckpoint(f"{out} has outputs without a checkpoint identity; use force")
                found = json.loads(ident_file.read_text())["identity"]
                if json.dumps(found, sort_keys=True) != json.dumps(identity, sort_keys=True):
                    diff = {k: (found.get(k), identity.get(k)) for k in identity_diff(found, identity)}
                    raise IncompatibleCheckpoint(f"{out} was produced with a different configuration, store or code "
                                                 f"{diff}; refusing to reuse it (use force to supersede)")
                if not resume:
                    raise FileExistsError(f"{out} exists; resume (default) continues it, force supersedes it")
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "_identity.json").exists():
            atomic_write_json(out / "_identity.json", {"identity": identity,
                                                      "created_at": datetime.now(timezone.utc).isoformat()})
        return cls(out, identity)

    def unit_identity(self, unit: str, **fields) -> dict:
        code = {"fold": "experiment", "inner": "inner_trial", "perm": "permutation"}[unit]
        return {**self.identity, "unit": unit, "unit_code_version": CODE_VERSIONS[code], **fields}

    def _check(self, name: str, ident: dict) -> bool:
        state, _ = check_unit(self.units, name, ident)
        if state == "incompatible":
            raise IncompatibleCheckpoint(f"unit {name} in {self.out} has a different identity; use force")
        if state == "corrupt":
            log.warning("unit %s corrupt; recomputing", name)
            for f in self.units.glob(f"{name}.*"):
                supersede(f, self.out)
        return state == "complete"

    def load_fold(self, name: str, ident: dict):
        if not self._check(name, ident):
            return None
        meta = json.loads((self.units / f"{name}.meta.json").read_text())
        qc = pd.read_csv(self.units / f"{name}.qc.csv") if (self.units / f"{name}.qc.csv").exists() else None
        return pd.read_csv(self.units / f"{name}.pred.csv"), meta["folds"], meta["training"], qc

    def save_fold(self, name, ident, pred: pd.DataFrame, folds: list, training: dict, qc: pd.DataFrame | None,
                  model_state: dict | None = None):
        files = [atomic_write_csv(self.units / f"{name}.pred.csv", pred),
                 atomic_write_json(self.units / f"{name}.meta.json", {"folds": folds, "training": training})]
        if qc is not None:
            files.append(atomic_write_csv(self.units / f"{name}.qc.csv", qc))
        if model_state:
            files.append(atomic_savez(self.units / f"{name}.model.npz", **model_state))
        write_marker(self.units, name, ident, files)

    def load_perm(self, name, ident):
        return json.loads((self.units / f"{name}.json").read_text())["value"] if self._check(name, ident) else None

    def save_perm(self, name, ident, value: float):
        write_marker(self.units, name, ident, [atomic_write_json(self.units / f"{name}.json", {"value": value})])


def select_hparams(data: Dataset, train_idx: np.ndarray, v: dict, seed: int) -> tuple[dict, list]:
    """Participant-level inner CV inside the outer training participants (classical models); returns (best params,
    score table). With a single configuration (every GNN) nothing is searched."""
    grid = hparam_grid(v)
    if len(grid) == 1:
        return grid[0], []
    ev = v["evaluation"]
    train_view = data.view(train_idx, "train")
    folds = inner_folds(train_view.meta, ev["inner_k"], seed, ("y",), data.target)
    scores = []
    for params in grid:
        vp = apply_hparams(v, params)
        preds = []
        for i, (itr, iva) in enumerate(folds):
            val = train_view.take(iva, "val")
            model = make_model(vp, seed + i).fit(train_view.take(itr, "train"))
            preds.append(val.meta[["window_id", "key", "dataset"]].assign(score=model.predict_score(val), y=val.y))
        subj = aggregate_subjects(pd.concat(preds))
        scores.append({**params, "inner_score": binary_metrics(subj["y"], subj["score"],
                                                               score_threshold(v))[ev["selection_metric"]]})
    best = max(range(len(grid)), key=lambda i: -np.inf if np.isnan(scores[i]["inner_score"])
               else scores[i]["inner_score"])
    return grid[best], scores


def _splits(meta: pd.DataFrame, v: dict, setting: str, n_repeats: int | None = None):
    ev = v["evaluation"]
    sub = setting_subset(meta, setting)
    m = meta.iloc[sub].reset_index(drop=True)
    reps = ev["n_repeats"] if n_repeats is None else n_repeats
    return sub, m, make_splits(m, setting, ev["k"], reps, ev["seed"], "y")


def fold_unit_name(sp) -> str:
    return f"fold_r{sp.repeat:03d}_f{sp.fold:02d}"


def cross_validate(data: Dataset, v: dict, setting: str, n_repeats: int | None = None,
                   ckpt: ExperimentCheckpoint | None = None, qc_summaries: pd.DataFrame | None = None,
                   collect_band_weights: bool = True):
    """Outer loop of one setting: (window predictions, fold table, training table, fold-QC table). With a
    checkpoint each completed outer split is saved immediately and reused on restart."""
    from ..qc.fold import fold_qc, fold_summary

    ev = v["evaluation"]
    sub, meta, splits = _splits(data.meta, v, setting, n_repeats)
    preds, folds, trains, qcs = [], [], [], []
    if ckpt is not None:
        ckpt.status.ensure([fold_unit_name(sp) for sp in splits])
    n_rep, n_fold = len({sp.repeat for sp in splits}), len({sp.fold for sp in splits})
    done_here: list[float] = []
    for i_sp, sp in enumerate(splits):
        if ckpt is not None:
            progress.set(repeat=f"{sp.repeat + 1}/{n_rep}", fold=f"{sp.fold + 1}/{n_fold}")
        tr, te = fold_indices(meta, sp)
        tr_g, te_g = sub[tr], sub[te]
        seed = int(ev["seed"] + 1000 * sp.repeat + sp.fold)
        sp_id = split_id(sp.train_keys, sp.test_keys)
        name, ident = fold_unit_name(sp), None
        if ckpt is not None:
            ident = ckpt.unit_identity("fold", split_id=sp_id, repeat=sp.repeat, fold=sp.fold, seed=seed)
            cached = ckpt.load_fold(name, ident)
            if cached is not None:
                pred_f, fold_rows, train_row, qc = cached
                preds.append(pred_f)
                folds += fold_rows
                trains.append(train_row)
                if qc is not None:
                    qcs.append(qc)
                if ckpt.status.get(name) != "completed":
                    ckpt.status.set(name, "completed", "validated existing checkpoint")
                continue
            ckpt.status.set(name, "running")
        t_fold = time.perf_counter()
        try:
            qc = None
            if qc_summaries is not None:
                qc = fold_qc(qc_summaries, sp.train_keys, sp.test_keys).assign(repeat=sp.repeat, fold=sp.fold)
            params, scores = select_hparams(data, tr_g, v, seed)
            t0 = time.perf_counter()
            model = make_model(apply_hparams(v, params), seed, data.info.get("roi_names")).fit(
                data.view(tr_g, "train"))
            fit_seconds = time.perf_counter() - t0
            model_state = (model.state_arrays() if ev.get("save_model_state") and hasattr(model, "state_arrays")
                           else None)
            test = data.view(te_g, "test")          # created only after fitting
            out = test.meta[["window_id", "key", "subject_id"]].copy()
            out["dataset"] = test.meta["dataset"].to_numpy()
            out["y"] = test.y
            if v["input"] == "graph" and collect_band_weights:
                score, bw = model.predict_score(test, return_band_weights=True)
                for i, bname in enumerate(data.info["graph_bands"]):
                    out[f"band_weight_{bname}"] = bw[:, i]
            else:
                score = model.predict_score(test)
            out["score"] = score
            out["repeat"], out["fold"] = sp.repeat, sp.fold
            fold_rows = ([{"repeat": sp.repeat, "fold": sp.fold, "key": k, "side": "train"} for k in sp.train_keys]
                         + [{"repeat": sp.repeat, "fold": sp.fold, "key": k, "side": "test"} for k in sp.test_keys])
            train_row = {"repeat": sp.repeat, "fold": sp.fold, "seed": seed, "split_id": sp_id,
                         "n_train_participants": len(sp.train_keys), "n_test_participants": len(sp.test_keys),
                         "n_train_windows": int(len(tr_g)), "n_test_windows": int(len(te_g)),
                         "selected": json.dumps(params), "inner_cv_scores": json.dumps(scores),
                         "n_parameters": model.n_parameters(), "fit_seconds": fit_seconds,
                         "compute_fingerprint": compute_environment(run_device(v))["fingerprint"]}
            if hasattr(model, "training_summary"):
                train_row.update(model.training_summary())
            if qc is not None:
                train_row.update(fold_summary(qc))
        except Exception as exc:
            if ckpt is not None:
                ckpt.status.set(name, "failed", f"{type(exc).__name__}: {exc}"[:500])
            raise
        if ckpt is not None:
            ckpt.save_fold(name, ident, out, fold_rows, train_row, qc, model_state)
            ckpt.status.set(name, "completed")
            done_here.append(time.perf_counter() - t_fold)
            remaining = len(splits) - i_sp - 1
            eta = (sum(done_here) / len(done_here) * remaining) if len(done_here) >= 2 and remaining else None
            progress.set(**{"unit ETA": progress.fmt_seconds(eta) if eta is not None else None})
            progress.advance(timed=False)
        preds.append(out)
        folds += fold_rows
        trains.append(train_row)
        if qc is not None:
            qcs.append(qc)
    if ckpt is not None:
        progress.set(repeat=None, fold=None, **{"unit ETA": None})
    return (pd.concat(preds, ignore_index=True), pd.DataFrame(folds), pd.DataFrame(trains),
            pd.concat(qcs, ignore_index=True) if qcs else pd.DataFrame())


def permute_participant_labels(data: Dataset, rng: np.random.Generator) -> Dataset:
    meta = data.meta.copy()
    subj = meta.drop_duplicates("key")[["key", data.target]]
    perm = dict(zip(subj["key"], rng.permutation(subj[data.target].to_numpy())))
    meta[data.target] = meta["key"].map(perm)
    return Dataset(meta, data.payload, data.kind, data.info, data.target)


FINAL_FILES = ("predictions_windows.csv", "predictions_participants.csv", "metrics.json", "fold_metrics.csv",
               "folds.csv", "training.csv", "fold_qc.csv", "manifest.json")


def result_dir(results_root: str | Path, v: dict, setting: str) -> Path:
    c = v["context"]
    return (Path(results_root) / c["inverse"] / f"mode_{c['mode']}" / c["atlas"] / c["graph"] / v["experiment"]
            / v["variant"] / setting)


def experiment_identity(store: GraphStore, v: dict, setting: str) -> dict:
    return {"kind": "final_experiment", "release": release.RELEASE, "code_version": CODE_VERSIONS["experiment"],
            "training_config_hash": training_hash(v), "graph_run_id": graph_run_id(store), "setting": setting,
            "experiment": v["experiment"], "variant": v["variant"], "context": v["context"],
            "code": code_identity("experiment")}


def run_variant(store: GraphStore, v: dict, setting: str, results_root: str | Path, data: Dataset | None = None,
                store_record: dict | None = None, qc_report: dict | None = None, resume: bool = True,
                force: bool = False, allow_ephemeral: bool = False) -> dict:
    """Run (or resume) one variant in one setting; returns its metrics. Completed results with the same identity
    are returned without recomputation; results of another configuration, store or code are never reused."""
    if "context" not in v:
        raise ValueError(f"{v.get('name')}: no run context (experiments.config.with_context)")
    v = with_store_context(v, store)
    ev = v["evaluation"]
    require_complete_cohort(store)
    if not store.manifest.get("synthetic"):
        check_persistent({"experiment output root": results_root}, allow_ephemeral=allow_ephemeral)
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; choose from {list(SETTINGS)}")
    out = result_dir(results_root, v, setting)
    ck = ExperimentCheckpoint.open(out, experiment_identity(store, v, setting), Path(results_root), resume, force)
    final_ident = {**ck.identity, "full_config_hash": config_hash(v)}
    state, _ = check_unit(out, "final", final_ident)
    if state == "complete":
        log.info("%s / %s: completed results found, nothing recomputed", v["name"], setting)
        return json.loads((out / "metrics.json").read_text())
    if state != "missing":
        for f in FINAL_FILES + ("final.done.json",):
            supersede(out / f, Path(results_root))

    data = data or build_dataset(store, v)
    summaries = None
    if qc_report is not None:
        from ..qc.report import fold_reference_summaries

        summaries = fold_reference_summaries(qc_report)
    pred, folds, training, fqc = cross_validate(data, v, setting, ckpt=ck, qc_summaries=summaries)
    thr = score_threshold(v)
    metrics, subj, fold_tab = evaluate_setting(pred, setting, thr, ev["n_bootstrap"], ev["seed"])
    metrics.update({"score_type": model_score_type(v), "subject_aggregation": "mean of window scores",
                    "experiment": v["experiment"], "variant": v["variant"], "context": v["context"]})
    perm_cfg = ev.get("permutation", {}) or {}
    if perm_cfg.get("n", 0) > 0:
        null = []
        for p in range(perm_cfg["n"]):
            name = f"perm_{p:04d}"
            ident = ck.unit_identity("perm", index=p, n_repeats=perm_cfg.get("n_repeats", 1), seed=ev["seed"])
            val = ck.load_perm(name, ident)
            progress.set(permutation=f"{p + 1}/{perm_cfg['n']}")
            if val is None:
                rng = np.random.default_rng([ev["seed"], p])
                p_pred, *_ = cross_validate(permute_participant_labels(data, rng), v, setting,
                                            n_repeats=perm_cfg.get("n_repeats", 1), collect_band_weights=False)
                p_subj = aggregate_subjects(p_pred, by=("repeat",))
                val = float(np.nanmean([binary_metrics(g["y"], g["score"], thr)["roc_auc"]
                                        for _, g in p_subj.groupby("repeat")]))
                ck.save_perm(name, ident, val)
                progress.advance(timed=False)
            null.append(val)
        progress.set(permutation=None)
        observed = metrics["participant"]["roc_auc"]
        metrics["permutation"] = {"metric": "participant pooled roc_auc", "observed": observed, "n": len(null),
                                  "p_value": permutation_p_value(observed, np.array(null)), "null": null}
    finals = [atomic_write_csv(out / "predictions_windows.csv", pred),
              atomic_write_csv(out / "predictions_participants.csv", subj),
              atomic_write_text(out / "metrics.json", json.dumps(metrics, indent=2, default=float)),
              atomic_write_csv(out / "fold_metrics.csv", fold_tab), atomic_write_csv(out / "folds.csv", folds),
              atomic_write_csv(out / "training.csv", training), atomic_write_csv(out / "fold_qc.csv", fqc)]
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "setting": setting, "variant_config": v,
        "variant_config_hash": config_hash(v), "training_config_hash": training_hash(v),
        "checkpoint_identity": ck.identity, "score_type": model_score_type(v), "score_threshold": thr,
        "data_info": {k: (list(x) if isinstance(x, tuple) else x) for k, x in data.info.items()},
        "store": {"root": str(store.root), "graph_run_id": graph_run_id(store),
                  "config_hash": store.manifest.get("config_hash"), "verification": store_record},
        "qc": (qc_report or {}).get("summary"), "holdout_policy": release.HOLDOUT_POLICY,
        "release": release.release_manifest(), "compute_environment": compute_environment(run_device(v)),
    }
    finals.append(atomic_write_json(out / "manifest.json", manifest))
    write_marker(out, "final", final_ident, finals)
    log.info("%s / %s: participant AUC %.3f", v["name"], setting, metrics["participant"]["roc_auc"])
    return metrics


def experiment_status(store: GraphStore, v: dict, setting: str, results_root: str | Path) -> dict:
    """Completed / pending / failed outer folds of one variant and setting, without running anything."""
    v = with_store_context(v, store)
    out = result_dir(results_root, v, setting)
    _, _, splits = _splits(store.windows, v, setting)
    row = {"experiment": v["experiment"], "variant": v["variant"], "setting": setting, "folds_total": len(splits),
           "folds_completed": 0, "folds_failed": 0, "state": "not started", "changed": ""}
    if not out.exists():
        return row
    ident_file = out / "_identity.json"
    ident = experiment_identity(store, v, setting)
    if not ident_file.exists() or json.loads(ident_file.read_text())["identity"] != ident:
        row["state"] = "incompatible"
        row["changed"] = (",".join(identity_diff(json.loads(ident_file.read_text())["identity"], ident))
                          if ident_file.exists() else "no identity file")
        return row
    ck = ExperimentCheckpoint(out, ident)
    ev = v["evaluation"]
    for sp in splits:
        seed = int(ev["seed"] + 1000 * sp.repeat + sp.fold)
        uid = ck.unit_identity("fold", split_id=split_id(sp.train_keys, sp.test_keys), repeat=sp.repeat,
                               fold=sp.fold, seed=seed)
        st, _ = check_unit(ck.units, fold_unit_name(sp), uid)
        if st == "complete":
            row["folds_completed"] += 1
        elif ck.status.get(fold_unit_name(sp)) == "failed":
            row["folds_failed"] += 1
    final_state, _ = check_unit(out, "final", {**ident, "full_config_hash": config_hash(v)})
    row["state"] = ("completed" if final_state == "complete" else "failed" if row["folds_failed"] else
                    "partial" if row["folds_completed"] else "not started")
    return row


def dry_run(store: GraphStore, variants: list[dict]) -> pd.DataFrame:
    """Build inputs and splits only (no training): shapes and participant / window counts per variant and setting."""
    rows = []
    for v in variants:
        data = build_dataset(store, with_store_context(v, store))
        for s in v["settings"]:
            sub, meta, splits = _splits(data.meta, v, s)
            for sp in splits:
                fold_indices(meta, sp)
            rows.append({"variant": v["name"], "setting": s, "n_participants": meta["key"].nunique(),
                         "n_windows": len(meta), "n_splits": len(splits), "n_hparam_combos": len(hparam_grid(v)),
                         "shapes": json.dumps({k: list(a.shape) for k, a in data.payload.items()})})
    return pd.DataFrame(rows)
