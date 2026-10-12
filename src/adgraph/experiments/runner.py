"""Run one experiment variant in one evaluation setting (DESIGN.md §2-4).

Per outer split:
    train view (role "train", training subjects only)
      -> hyperparameters chosen by subject-level inner CV inside the train view
      -> model.fit(train view)      (all fitted transforms live in the model)
    test view (role "test", built only after fitting)
      -> window scores -> subject score = mean over windows
         (scores: probabilities, threshold 0.5; or SVM decision-function margins, threshold 0)

Outputs per (variant, setting) directory:
    predictions_windows.csv   repeat, fold, window_id, key, dataset, subject_id, y, score [, band weights]
    predictions_subjects.csv  repeat, key, dataset, y, score, n_windows
    metrics_repeats.csv       subject-level metrics per repeat (+ per dataset in pooled runs)
    folds.csv                 subject -> (repeat, fold, side)
    hparams.csv               selected hyperparameters per split + inner-CV scores, n_parameters, fit_seconds
    embeddings.npz            optional (evaluation.save_embeddings): fused graph embeddings of the train and
                              test windows of every split, each from that split's single fitted model
    summary.json              mean/SD/percentiles, bootstrap CI, window-level metrics, permutation test
    manifest.json             resolved variant config, graph run provenance, git commit, versions
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import config_hash
from ..evaluation.metrics import (aggregate_subjects, binary_metrics, bootstrap_ci, permutation_p_value,
                                  summarize_repeats)
from ..evaluation.splits import (CROSS_SETTINGS, SETTINGS, fold_indices, inner_folds, make_splits, setting_subset)
from ..graphs.dataset import Dataset, build_dataset
from ..graphs.store import GraphStore
from ..provenance import git_commit, versions
from .config import apply_hparams, hparam_grid
from .gate import gate_failures

log = logging.getLogger(__name__)


def model_score_type(v: dict) -> str:
    """Score scale produced by a variant's model: 'probability' or 'decision_function'."""
    if v["input"] == "classical":
        from ..models.classical import score_type

        return score_type(v["model"]["name"], v["model"].get("calibration", "none"))
    return "probability"


def score_threshold(v: dict) -> float:
    """Fixed decision threshold on the model's score scale (never tuned on test subjects)."""
    ev = v["evaluation"]
    return ev.get("decision_threshold", 0.0) if model_score_type(v) == "decision_function" else ev["threshold"]


def make_model(v: dict, seed: int):
    if v["input"] == "classical":
        from ..models.classical import ClassicalModel

        m = v["model"]
        return ClassicalModel(m["name"], m.get("params", {}), v.get("preprocessing", {}), seed,
                              m.get("calibration", "none"), m.get("calibration_folds", 3))
    from ..models.trainer import GNNModel

    return GNNModel(v["model"], v["training"], v["node_features"].get("scaling", "node"), v.get("sparsify"), seed,
                    v.get("device"))


def _strata(setting: str) -> tuple[str, ...]:
    return SETTINGS[setting][2]


def select_hparams(data: Dataset, train_idx: np.ndarray, v: dict, setting: str, seed: int) -> tuple[dict, list]:
    """Subject-level inner CV inside the training subjects; returns (best params, score table)."""
    grid = hparam_grid(v)
    if len(grid) == 1:
        return grid[0], []
    ev = v["evaluation"]
    train_view = data.view(train_idx, "train")
    folds = inner_folds(train_view.meta, ev["inner_k"], seed, _strata(setting) if setting == "pooled" else ("y",),
                        data.target)
    scores = []
    for params in grid:
        vp = apply_hparams(v, params)
        preds = []
        for i, (itr, iva) in enumerate(folds):
            model = make_model(vp, seed + i).fit(train_view.take(itr, "train"))
            val = train_view.take(iva, "val")
            preds.append(val.meta.assign(score=model.predict_score(val), y=val.y))
        subj = aggregate_subjects(pd.concat(preds))
        score = binary_metrics(subj["y"], subj["score"], score_threshold(v))[ev["selection_metric"]]
        scores.append({**params, "inner_score": score})
    best = max(range(len(grid)), key=lambda i: -np.inf if np.isnan(scores[i]["inner_score"])
               else scores[i]["inner_score"])
    return grid[best], scores


def cross_validate(data: Dataset, v: dict, setting: str, n_repeats: int | None = None,
                   collect_band_weights: bool = True,
                   embeddings_out: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Outer loop for one setting. Returns window predictions, fold table, hparam table."""
    ev = v["evaluation"]
    sub = setting_subset(data.meta, setting)
    meta = data.meta.iloc[sub].reset_index(drop=True)
    cross = setting in CROSS_SETTINGS
    reps = n_repeats if n_repeats is not None else (ev["cross_n_seeds"] if cross else ev["n_repeats"])
    splits = make_splits(meta, setting, ev["k"], reps, ev["seed"], data.target, ev.get("scheme", "kfold"))
    preds, folds, hps = [], [], []
    for sp in splits:
        tr, te = fold_indices(meta, sp)
        tr_g, te_g = sub[tr], sub[te]
        seed = int(ev["seed"] + 1000 * sp.repeat + sp.fold)
        params, scores = select_hparams(data, tr_g, v, setting, seed)
        t0 = time.perf_counter()
        model = make_model(apply_hparams(v, params), seed).fit(data.view(tr_g, "train"))
        fit_seconds = time.perf_counter() - t0
        test = data.view(te_g, "test")      # created only after fitting
        out = test.meta[["window_id", "key", "dataset", "subject_id"]].copy()
        out["y"] = test.y
        if v["input"] == "graph" and collect_band_weights:
            score, bw = model.predict_score(test, return_band_weights=True)
            for i, name in enumerate(data.info["graph_bands"]):
                out[f"band_weight_{name}"] = bw[:, i]
        else:
            score = model.predict_score(test)
        out["score"] = score
        out["repeat"], out["fold"] = sp.repeat, sp.fold
        preds.append(out)
        folds += [{"repeat": sp.repeat, "fold": sp.fold, "key": k, "side": "train"} for k in sp.train_keys]
        folds += [{"repeat": sp.repeat, "fold": sp.fold, "key": k, "side": "test"} for k in sp.test_keys]
        hps.append({"repeat": sp.repeat, "fold": sp.fold, "selected": json.dumps(params), "scores": json.dumps(scores),
                    "n_parameters": model.n_parameters(), "fit_seconds": fit_seconds})
        if embeddings_out is not None and hasattr(model, "embed"):
            for side, idx in (("train", tr_g), ("test", te_g)):
                view = data.view(idx, "test")          # read-only use of an already fitted model
                embeddings_out.append({"emb": model.embed(view), "window_id": view.meta["window_id"].to_numpy(),
                                       "key": view.meta["key"].to_numpy(), "side": side,
                                       "y": view.y, "dataset": view.meta["dataset"].to_numpy(),
                                       "repeat": sp.repeat, "fold": sp.fold})
    return pd.concat(preds, ignore_index=True), pd.DataFrame(folds), pd.DataFrame(hps)


def evaluate(pred: pd.DataFrame, threshold: float = 0.5, n_boot: int = 2000, seed: int = 0):
    """Subject-level metrics per repeat (+ per dataset), summary and bootstrap CI.

    ``threshold`` must be on the score scale of the model (``score_threshold``)."""
    subj = aggregate_subjects(pred, by=("repeat",))
    rows = []
    for r, g in subj.groupby("repeat"):
        rows.append({"repeat": r, "scope": "all", **binary_metrics(g["y"], g["score"], threshold)})
        if g["dataset"].nunique() > 1:
            for ds, gd in g.groupby("dataset"):
                rows.append({"repeat": r, "scope": ds, **binary_metrics(gd["y"], gd["score"], threshold)})
    per_repeat = pd.DataFrame(rows)
    # Each subject's score averaged over repeats -> one prediction per subject -> bootstrap CI.
    mean_subj = subj.groupby("key").agg(dataset=("dataset", "first"), y=("y", "first"), score=("score", "mean"))
    summary = {
        "n_subjects": int(len(mean_subj)), "n_windows": int(pred["window_id"].nunique()),
        "n_repeats": int(subj["repeat"].nunique()),
        "subject_level": {scope: summarize_repeats(g) for scope, g in per_repeat.groupby("scope")},
        "subject_level_repeat_averaged": binary_metrics(mean_subj["y"], mean_subj["score"], threshold),
        "bootstrap_ci_95": bootstrap_ci(mean_subj["y"].to_numpy(), mean_subj["score"].to_numpy(), n_boot, seed,
                                        threshold),
        "window_level": summarize_repeats(pd.DataFrame(
            [binary_metrics(g["y"], g["score"], threshold) for _, g in pred.groupby("repeat")])),
        "confusion_repeat_averaged": {k: int(v) for k, v in binary_metrics(mean_subj["y"], mean_subj["score"],
                                                                          threshold).items()
                                      if k in ("tn", "fp", "fn", "tp")},
    }
    return subj, per_repeat, summary


def permute_subject_labels(data: Dataset, rng: np.random.Generator) -> Dataset:
    """Copy of ``data`` with labels permuted across subjects (within each dataset)."""
    meta = data.meta.copy()
    subj = meta.drop_duplicates("key")[["key", "dataset", data.target]]
    perm = {}
    for _, g in subj.groupby("dataset"):
        perm.update(dict(zip(g["key"], rng.permutation(g[data.target].to_numpy()))))
    meta[data.target] = meta["key"].map(perm)
    return Dataset(meta, data.payload, data.kind, data.info, data.target)


def run_variant(store: GraphStore, v: dict, setting: str, out_dir: str | Path, data: Dataset | None = None) -> dict:
    ev = v["evaluation"]
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; choose from {list(SETTINGS)}")
    if v.get("target", "label") == "dataset" and setting != "pooled":
        raise ValueError("the dataset-identity probe is only defined for the pooled setting")
    if setting in CROSS_SETTINGS:
        fails = gate_failures(store)
        if fails and not ev.get("allow_ungated_cross_dataset", False):
            raise RuntimeError("cross-dataset gate not passed (DESIGN.md §7): " + "; ".join(fails))
    data = data or build_dataset(store, v)
    out = Path(out_dir) / v["experiment"] / v["variant"] / setting
    out.mkdir(parents=True, exist_ok=False)

    emb = [] if (ev.get("save_embeddings", False) and v["input"] == "graph") else None
    pred, folds, hps = cross_validate(data, v, setting, embeddings_out=emb)
    if emb:
        n = [len(e["window_id"]) for e in emb]
        np.savez_compressed(out / "embeddings.npz", emb=np.concatenate([e["emb"] for e in emb]).astype(np.float32),
                            window_id=np.concatenate([e["window_id"] for e in emb]),
                            key=np.concatenate([e["key"] for e in emb]),
                            y=np.concatenate([e["y"] for e in emb]),
                            dataset=np.concatenate([e["dataset"] for e in emb]),
                            side=np.repeat([e["side"] for e in emb], n),
                            repeat=np.repeat([e["repeat"] for e in emb], n),
                            fold=np.repeat([e["fold"] for e in emb], n))
    thr = score_threshold(v)
    subj, per_repeat, summary = evaluate(pred, thr, ev["n_bootstrap"], ev["seed"])
    summary["score"] = {"type": model_score_type(v), "threshold": thr,
                        "subject_aggregation": "mean of window scores"}

    perm_cfg = ev.get("permutation", {}) or {}
    if perm_cfg.get("n", 0) > 0:
        rng = np.random.default_rng(ev["seed"])
        null = []
        for _ in range(perm_cfg["n"]):
            p_pred, _, _ = cross_validate(permute_subject_labels(data, rng), v, setting,
                                          n_repeats=perm_cfg.get("n_repeats", 1), collect_band_weights=False)
            _, p_rep, _ = evaluate(p_pred, thr, n_boot=0)
            null.append(p_rep.loc[p_rep["scope"] == "all", "roc_auc"].mean())
        observed = per_repeat.loc[per_repeat["scope"] == "all", "roc_auc"].mean()
        summary["permutation"] = {"metric": "roc_auc", "observed": observed, "n": len(null),
                                  "p_value": permutation_p_value(observed, np.array(null)), "null": null}

    pred.to_csv(out / "predictions_windows.csv", index=False)
    subj.to_csv(out / "predictions_subjects.csv", index=False)
    per_repeat.to_csv(out / "metrics_repeats.csv", index=False)
    folds.to_csv(out / "folds.csv", index=False)
    hps.to_csv(out / "hparams.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "setting": setting, "variant_config": v,
        "variant_config_hash": config_hash(v), "score_type": model_score_type(v),
        "score_threshold": score_threshold(v), "data_info": {k: (list(x) if isinstance(x, tuple) else x)
                                                               for k, x in data.info.items()},
        "graph_run": {"root": str(store.root), "config_hash": store.manifest.get("config_hash"),
                      "git_commit": store.manifest.get("git_commit")},
        "cross_dataset_gate_failures": gate_failures(store) if setting in CROSS_SETTINGS else None,
        "git_commit": git_commit(), "versions": versions(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("%s / %s: subject AUC %.3f", v["name"], setting,
             summary["subject_level"]["all"]["roc_auc"]["mean"])
    return summary


def dry_run(store: GraphStore, variants: list[dict], settings: list[str]) -> pd.DataFrame:
    """Build inputs and splits only (no training): shapes and subject/window counts per variant and setting."""
    rows = []
    for v in variants:
        data = build_dataset(store, v)
        for s in settings:
            sub = setting_subset(data.meta, s)
            meta = data.meta.iloc[sub].reset_index(drop=True)
            ev = v["evaluation"]
            reps = ev["cross_n_seeds"] if s in CROSS_SETTINGS else ev["n_repeats"]
            splits = make_splits(meta, s, ev["k"], reps, ev["seed"], data.target, ev.get("scheme", "kfold"))
            for sp in splits:
                fold_indices(meta, sp)
            shapes = {k: list(a.shape) for k, a in data.payload.items()}
            rows.append({"variant": v["name"], "setting": s, "n_subjects": meta["key"].nunique(),
                         "n_windows": len(meta), "n_splits": len(splits), "n_hparam_combos": len(hparam_grid(v)),
                         "shapes": json.dumps(shapes)})
    return pd.DataFrame(rows)
