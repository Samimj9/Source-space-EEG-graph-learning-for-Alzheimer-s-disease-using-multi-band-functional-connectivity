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
                              test windows of every split, each from that split's single fitted model, stored
                              per model under keys ``r<repeat>_f<fold>__{emb,window_id,key,y,dataset,side}``
                              (models of different folds may differ in width and are never concatenated)
    units/, status.csv        checkpoints: one validated unit per outer fold, inner-CV trial and permutation
    _identity.json, final.done.json   checkpoint identity and completion marker of the final outputs
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
from ..graphs.dataset import Dataset, build_dataset, target_column
from ..graphs.store import GraphStore
from ..checkpoint import (CODE_VERSIONS, IncompatibleCheckpoint, StatusTable, atomic_savez, atomic_write_csv,
                          atomic_write_json, atomic_write_text, check_unit, code_identity, identity_diff, stable_hash,
                          supersede, write_marker)
from ..graphs.store import graph_run_id, require_complete_cohort
from ..provenance import compute_environment, git_commit, git_dirty, versions
from .. import progress
from ..storage import check_persistent
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


def run_device(v: dict) -> str:
    """Device the variant's model runs on (classical models: CPU)."""
    if v["input"] == "classical":
        return "cpu"
    if v.get("device"):
        return v["device"]
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


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


# ====================================================================== checkpointing
EVAL_ONLY_KEYS = ("n_repeats", "cross_n_seeds", "n_bootstrap", "permutation")


def training_hash(v: dict) -> str:
    """Hash of everything that affects what a fold trains/predicts.

    Evaluation-only settings (number of repeats, bootstrap/permutation counts) and display
    fields are excluded, so e.g. raising ``n_repeats`` reuses the repeats already computed."""
    vv = {k: val for k, val in v.items() if k not in ("name", "variant", "settings")}
    vv["evaluation"] = {k: val for k, val in v["evaluation"].items() if k not in EVAL_ONLY_KEYS}
    return config_hash(vv)


def split_id(train_keys, test_keys) -> str:
    return stable_hash({"train": sorted(train_keys), "test": sorted(test_keys)})


class ExperimentCheckpoint:
    """Checkpoint directory of one (experiment, variant, setting).

    Units (outer folds, inner-CV trials, permutations) live in ``units/`` and are reused
    only if their identity matches: training-config hash + graph-run id + setting + split id
    + repeat + fold + seed (+ trial parameters / permutation index) + code version.
    """

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
                    raise IncompatibleCheckpoint(f"{out} has outputs without a checkpoint identity (older run); "
                                                 "use force to supersede them")
                found = json.loads(ident_file.read_text())["identity"]
                if json.dumps(found, sort_keys=True) != json.dumps(identity, sort_keys=True):
                    diff = {k: (found.get(k), identity.get(k)) for k in identity_diff(found, identity)}
                    raise IncompatibleCheckpoint(f"{out} was produced with a different configuration, graph run or "
                                                 f"code {diff}; refusing to reuse it (use force to supersede)")
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

    # -- outer folds
    def load_fold(self, name: str, ident: dict):
        if not self._check(name, ident):
            return None
        meta = json.loads((self.units / f"{name}.meta.json").read_text())
        emb = None
        if (self.units / f"{name}.emb.npz").exists():
            z = np.load(self.units / f"{name}.emb.npz", allow_pickle=False)
            emb = {k: z[k] for k in z.files}
        return pd.read_csv(self.units / f"{name}.pred.csv"), meta["folds"], meta["hparams"], emb

    def save_fold(self, name, ident, pred: pd.DataFrame, folds: list, hp: dict, emb: dict | None):
        files = [atomic_write_csv(self.units / f"{name}.pred.csv", pred),
                 atomic_write_json(self.units / f"{name}.meta.json", {"folds": folds, "hparams": hp})]
        if emb is not None:
            files.append(atomic_savez(self.units / f"{name}.emb.npz", **emb))
        write_marker(self.units, name, ident, files)

    # -- inner-CV trials
    def load_trial(self, name, ident):
        return pd.read_csv(self.units / f"{name}.val.csv") if self._check(name, ident) else None

    def save_trial(self, name, ident, val_pred: pd.DataFrame):
        write_marker(self.units, name, ident, [atomic_write_csv(self.units / f"{name}.val.csv", val_pred)])

    # -- permutations
    def load_perm(self, name, ident):
        return json.loads((self.units / f"{name}.json").read_text())["value"] if self._check(name, ident) else None

    def save_perm(self, name, ident, value: float):
        write_marker(self.units, name, ident, [atomic_write_json(self.units / f"{name}.json", {"value": value})])


def select_hparams(data: Dataset, train_idx: np.ndarray, v: dict, setting: str, seed: int,
                   ckpt: ExperimentCheckpoint | None = None, sp_id: str = "",
                   fold_name: str = "") -> tuple[dict, list]:
    """Subject-level inner CV inside the training subjects; returns (best params, score table).

    With a checkpoint, every completed (parameter combination, inner fold) trial is cached,
    so an interrupted search resumes where it stopped (GNNs only: a classical fit takes
    milliseconds, less than writing its checkpoint to Drive, so classical inner trials are
    recomputed on resume; their outer folds are checkpointed as usual). Trials are named after their outer
    fold unit (``fold_name``): the same subject split can recur in another repeat, where it
    is a different unit (different seed)."""
    grid = hparam_grid(v)
    if len(grid) == 1:
        return grid[0], []
    ev = v["evaluation"]
    train_view = data.view(train_idx, "train")
    folds = inner_folds(train_view.meta, ev["inner_k"], seed, _strata(setting) if setting == "pooled" else ("y",),
                        data.target)
    scores = []
    if v["input"] != "graph":
        ckpt = None                                   # inner-trial cache only where fits are expensive
    n_trials, trial = len(grid) * len(folds), 0
    for params in grid:
        vp = apply_hparams(v, params)
        preds = []
        for i, (itr, iva) in enumerate(folds):
            trial += 1
            progress.set(inner=f"{trial}/{n_trials}")
            val = train_view.take(iva, "val")
            name = ident = None
            if ckpt is not None:
                inner_id = split_id(train_view.meta["key"].iloc[itr].unique(), val.meta["key"].unique())
                name = f"inner_{fold_name or sp_id}_{stable_hash(params, 8)}_i{i}"
                ident = ckpt.unit_identity("inner", split_id=sp_id, params=params, inner_fold=i,
                                           inner_split_id=inner_id, seed=seed + i)
                cached = ckpt.load_trial(name, ident)
                if cached is not None:
                    preds.append(cached)
                    continue
            model = make_model(vp, seed + i).fit(train_view.take(itr, "train"))
            out = val.meta[["window_id", "key", "dataset"]].assign(score=model.predict_score(val), y=val.y)
            if ckpt is not None:
                ckpt.save_trial(name, ident, out)
            preds.append(out)
        subj = aggregate_subjects(pd.concat(preds))
        score = binary_metrics(subj["y"], subj["score"], score_threshold(v))[ev["selection_metric"]]
        scores.append({**params, "inner_score": score})
    progress.set(inner="refit")
    best = max(range(len(grid)), key=lambda i: -np.inf if np.isnan(scores[i]["inner_score"])
               else scores[i]["inner_score"])
    return grid[best], scores


def _splits(data_meta: pd.DataFrame, v: dict, setting: str, target: str, n_repeats: int | None = None):
    ev = v["evaluation"]
    sub = setting_subset(data_meta, setting)
    meta = data_meta.iloc[sub].reset_index(drop=True)
    reps = n_repeats if n_repeats is not None else (ev["cross_n_seeds"] if setting in CROSS_SETTINGS
                                                   else ev["n_repeats"])
    return sub, meta, make_splits(meta, setting, ev["k"], reps, ev["seed"], target, ev.get("scheme", "kfold"))


def fold_unit_name(sp) -> str:
    return f"fold_r{sp.repeat:03d}_f{sp.fold:02d}"


def cross_validate(data: Dataset, v: dict, setting: str, n_repeats: int | None = None,
                   collect_band_weights: bool = True, embeddings_out: list | None = None,
                   ckpt: ExperimentCheckpoint | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Outer loop for one setting. Returns window predictions, fold table, hparam table.

    With a checkpoint, each completed outer split is saved immediately and, on restart,
    loaded instead of retrained (identity: config + graph run + split + repeat + fold + seed)."""
    ev = v["evaluation"]
    sub, meta, splits = _splits(data.meta, v, setting, data.target, n_repeats)
    preds, folds, hps = [], [], []
    if ckpt is not None:
        ckpt.status.ensure([fold_unit_name(sp) for sp in splits])
    n_rep, n_fold = len({sp.repeat for sp in splits}), len({sp.fold for sp in splits})
    done_here: list[float] = []                   # durations of folds computed in this session (unit ETA)
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
                pred_f, fold_rows, hp_row, emb = cached
                preds.append(pred_f)
                folds += fold_rows
                hps.append(hp_row)
                if embeddings_out is not None and emb is not None:
                    embeddings_out.append(emb)
                if ckpt.status.get(name) != "completed":
                    ckpt.status.set(name, "completed", "validated existing checkpoint")
                continue                          # already counted as done by the progress bar
            ckpt.status.set(name, "running")
        t_fold = time.perf_counter()
        try:
            params, scores = select_hparams(data, tr_g, v, setting, seed, ckpt, sp_id, name)
            t0 = time.perf_counter()
            model = make_model(apply_hparams(v, params), seed).fit(data.view(tr_g, "train"))
            fit_seconds = time.perf_counter() - t0
            test = data.view(te_g, "test")      # created only after fitting
            out = test.meta[["window_id", "key", "dataset", "subject_id"]].copy()
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
            hp_row = {"repeat": sp.repeat, "fold": sp.fold, "selected": json.dumps(params),
                      "scores": json.dumps(scores), "n_parameters": model.n_parameters(), "fit_seconds": fit_seconds,
                      "compute_fingerprint": compute_environment(run_device(v))["fingerprint"],
                      "split_id": sp_id, "seed": seed}
            emb = None
            if embeddings_out is not None and hasattr(model, "embed"):
                parts = []
                for side, idx in (("train", tr_g), ("test", te_g)):
                    view = data.view(idx, "test")          # read-only use of an already fitted model
                    parts.append({"emb": model.embed(view), "window_id": view.meta["window_id"].to_numpy(),
                                  "key": view.meta["key"].to_numpy(), "y": view.y,
                                  "dataset": view.meta["dataset"].to_numpy(),
                                  "side": np.repeat(side, len(idx)), "repeat": np.repeat(sp.repeat, len(idx)),
                                  "fold": np.repeat(sp.fold, len(idx))})
                emb = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
                emb["window_id"] = emb["window_id"].astype(str)
                for k in ("key", "dataset", "side"):
                    emb[k] = emb[k].astype(str)
                embeddings_out.append(emb)
        except Exception as exc:
            if ckpt is not None:
                ckpt.status.set(name, "failed", f"{type(exc).__name__}: {exc}"[:500])
            raise
        if ckpt is not None:
            ckpt.save_fold(name, ident, out, fold_rows, hp_row, emb)
            ckpt.status.set(name, "completed")
            done_here.append(time.perf_counter() - t_fold)
            remaining = len(splits) - i_sp - 1
            eta = (sum(done_here) / len(done_here) * remaining) if len(done_here) >= 2 and remaining else None
            progress.set(inner=None, **{"unit ETA": progress.fmt_seconds(eta) if eta is not None else None})
            progress.advance(timed=False)
        preds.append(out)
        folds += fold_rows
        hps.append(hp_row)
    if ckpt is not None:
        progress.set(repeat=None, fold=None, inner=None, **{"unit ETA": None})
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
    """Copy of ``data`` with labels permuted across subjects (within each dataset).

    For the dataset-identity probe the label is constant within a dataset, so it is permuted
    across all subjects instead (a within-dataset permutation would leave it unchanged)."""
    meta = data.meta.copy()
    subj = meta.drop_duplicates("key")[["key", "dataset", data.target]]
    perm = {}
    groups = [subj] if data.target == target_column("dataset") else [g for _, g in subj.groupby("dataset")]
    for g in groups:
        perm.update(dict(zip(g["key"], rng.permutation(g[data.target].to_numpy()))))
    meta[data.target] = meta["key"].map(perm)
    return Dataset(meta, data.payload, data.kind, data.info, data.target)


FINAL_FILES = ("predictions_windows.csv", "predictions_subjects.csv", "metrics_repeats.csv", "folds.csv",
               "hparams.csv", "summary.json", "manifest.json")


def result_dir(out_dir: str | Path, v: dict, setting: str) -> Path:
    """``<out>/<experiment>/<variant>/<setting>``; analyses other than the primary one (dataset
    target, LOSO) get their own directory, so they can never collide with primary results."""
    variant = v["variant"]
    if v.get("target", "label") != "label":
        variant += f"__target-{v['target']}"
    if v["evaluation"].get("scheme", "kfold") != "kfold":
        variant += f"__scheme-{v['evaluation']['scheme']}"
    return Path(out_dir) / v["experiment"] / variant / setting


def experiment_identity(store: GraphStore, v: dict, setting: str) -> dict:
    """Identity shared by all units of one variant/setting; ``code`` = git commit + source hash + libraries."""
    return {"kind": "experiment", "code_version": CODE_VERSIONS["experiment"], "training_config_hash": training_hash(v),
            "graph_run_id": graph_run_id(store), "setting": setting, "experiment": v["experiment"],
            "variant": v["variant"], "code": code_identity("experiment")}


def run_variant(store: GraphStore, v: dict, setting: str, out_dir: str | Path, data: Dataset | None = None,
                resume: bool = True, force: bool = False, allow_ephemeral: bool = False) -> dict:
    """Run (or resume) one variant in one setting; returns the summary.

    Already completed results with the same identity are returned without recomputation;
    completed outer folds, inner-CV trials and permutations are reused; results produced with a
    different configuration or graph run are never reused (IncompatibleCheckpoint) unless
    ``force`` supersedes them (moved to ``_superseded/``, never deleted or overwritten).
    For real (non-synthetic) graph runs ``out_dir`` must be persistent storage (mounted Drive
    in Colab), else ``EphemeralStorageError`` unless ``allow_ephemeral``."""
    ev = v["evaluation"]
    require_complete_cohort(store)              # never train on an empty, partial or smoke-test graph run
    if not store.manifest.get("synthetic"):
        check_persistent({"experiment output root": out_dir}, allow_ephemeral=allow_ephemeral)
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; choose from {list(SETTINGS)}")
    if v.get("target", "label") == "dataset" and setting != "pooled":
        raise ValueError("the dataset-identity probe is only defined for the pooled setting")
    if setting in CROSS_SETTINGS:
        fails = gate_failures(store)
        if fails and not ev.get("allow_ungated_cross_dataset", False):
            raise RuntimeError("cross-dataset gate not passed (DESIGN.md §7): " + "; ".join(fails))
    out = result_dir(out_dir, v, setting)
    ck = ExperimentCheckpoint.open(out, experiment_identity(store, v, setting), Path(out_dir), resume, force)
    final_ident = {**ck.identity, "full_config_hash": config_hash(v)}
    state, _ = check_unit(out, "final", final_ident)
    if state == "complete":
        log.info("%s / %s: completed results found, nothing recomputed", v["name"], setting)
        return json.loads((out / "summary.json").read_text())
    if state != "missing":                       # evaluation settings changed: re-evaluate from fold units
        for f in FINAL_FILES + ("embeddings.npz", "final.done.json"):
            supersede(out / f, Path(out_dir))

    data = data or build_dataset(store, v)
    emb = [] if (ev.get("save_embeddings", False) and v["input"] == "graph") else None
    pred, folds, hps = cross_validate(data, v, setting, embeddings_out=emb, ckpt=ck)
    finals = []
    if emb:
        per_model = {}
        for e in emb:
            prefix = f"r{int(e['repeat'][0]):03d}_f{int(e['fold'][0]):02d}__"
            per_model.update({prefix + k: e[k] for k in ("emb", "window_id", "key", "y", "dataset", "side")})
        finals.append(atomic_savez(out / "embeddings.npz", **per_model))
    thr = score_threshold(v)
    subj, per_repeat, summary = evaluate(pred, thr, ev["n_bootstrap"], ev["seed"])
    summary["score"] = {"type": model_score_type(v), "threshold": thr,
                        "subject_aggregation": "mean of window scores"}

    perm_cfg = ev.get("permutation", {}) or {}
    if perm_cfg.get("n", 0) > 0:
        null = []
        for p in range(perm_cfg["n"]):
            name = f"perm_{p:04d}"
            ident = ck.unit_identity("perm", index=p, n_repeats=perm_cfg.get("n_repeats", 1), seed=ev["seed"])
            val = ck.load_perm(name, ident)
            progress.set(permutation=f"{p + 1}/{perm_cfg['n']}")
            if val is None:
                rng = np.random.default_rng([ev["seed"], p])          # independent per permutation (resumable)
                p_pred, _, _ = cross_validate(permute_subject_labels(data, rng), v, setting,
                                              n_repeats=perm_cfg.get("n_repeats", 1), collect_band_weights=False)
                _, p_rep, _ = evaluate(p_pred, thr, n_boot=0)
                val = float(p_rep.loc[p_rep["scope"] == "all", "roc_auc"].mean())
                ck.save_perm(name, ident, val)
                progress.advance(timed=False)
            null.append(val)
        progress.set(permutation=None)
        observed = per_repeat.loc[per_repeat["scope"] == "all", "roc_auc"].mean()
        summary["permutation"] = {"metric": "roc_auc", "observed": observed, "n": len(null),
                                  "p_value": permutation_p_value(observed, np.array(null)), "null": null}

    finals += [atomic_write_csv(out / "predictions_windows.csv", pred),
               atomic_write_csv(out / "predictions_subjects.csv", subj),
               atomic_write_csv(out / "metrics_repeats.csv", per_repeat),
               atomic_write_csv(out / "folds.csv", folds), atomic_write_csv(out / "hparams.csv", hps),
               atomic_write_text(out / "summary.json", json.dumps(summary, indent=2, default=float))]
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "setting": setting, "variant_config": v,
        "variant_config_hash": config_hash(v), "training_config_hash": training_hash(v),
        "checkpoint_identity": ck.identity, "score_type": model_score_type(v),
        "score_threshold": score_threshold(v), "data_info": {k: (list(x) if isinstance(x, tuple) else x)
                                                               for k, x in data.info.items()},
        "graph_run": {"root": str(store.root), "graph_run_id": graph_run_id(store),
                      "config_hash": store.manifest.get("config_hash"), "git_commit": store.manifest.get("git_commit")},
        "cross_dataset_gate_failures": gate_failures(store) if setting in CROSS_SETTINGS else None,
        "git_commit": git_commit(), "git_dirty": git_dirty(), "versions": versions(),
        "compute_environment": compute_environment(run_device(v)),   # needed to compare fit_seconds
    }
    finals.append(atomic_write_json(out / "manifest.json", manifest))
    write_marker(out, "final", final_ident, finals)
    log.info("%s / %s: subject AUC %.3f", v["name"], setting, summary["subject_level"]["all"]["roc_auc"]["mean"])
    return summary


def experiment_status(store: GraphStore, v: dict, setting: str, out_dir: str | Path) -> dict:
    """Completed / pending / failed outer folds of one variant and setting, without running anything."""
    from ..graphs.dataset import _meta, target_column

    out = result_dir(out_dir, v, setting)
    target = target_column(v.get("target", "label"))
    meta = _meta(store, v.get("target", "label"))
    _, _, splits = _splits(meta, v, setting, target)
    perm_cfg = v["evaluation"].get("permutation", {}) or {}
    n_perm = int(perm_cfg.get("n", 0) or 0)
    row = {"experiment": v["experiment"], "variant": v["variant"], "setting": setting, "folds_total": len(splits),
           "folds_completed": 0, "folds_failed": 0, "inner_trials_cached": 0, "perms_total": n_perm,
           "perms_completed": 0, "state": "not started", "changed": ""}
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
        state, _ = check_unit(ck.units, fold_unit_name(sp), uid)
        if state == "complete":
            row["folds_completed"] += 1
        elif ck.status.get(fold_unit_name(sp)) == "failed":
            row["folds_failed"] += 1
    row["inner_trials_cached"] = len(list(ck.units.glob("inner_*.done.json"))) if ck.units.exists() else 0
    for p in range(n_perm):
        pid = ck.unit_identity("perm", index=p, n_repeats=perm_cfg.get("n_repeats", 1), seed=ev["seed"])
        row["perms_completed"] += check_unit(ck.units, f"perm_{p:04d}", pid)[0] == "complete"
    final_state, _ = check_unit(out, "final", {**ident, "full_config_hash": config_hash(v)})
    row["state"] = ("completed" if final_state == "complete" else
                    "failed" if row["folds_failed"] else
                    "partial" if row["folds_completed"] else "not started")
    return row


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
