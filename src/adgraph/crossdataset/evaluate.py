"""Participant-level evaluation of the frozen campaign (design section 8-9). The ONLY place target labels are read,
and only for units of the frozen record at its commit.

Per transfer unit and target stratum (D3 targets: all 65, the 52 development and the 13 holdout participants; D2
targets: the 22 participants of the D2 analysis cohort, amendment X1; D2-19 excluded):
* ROC-AUC (DeLong 95 % CI and participant bootstrap stratified by diagnosis), balanced accuracy, sensitivity (AD),
  specificity (HC) (Wilson and bootstrap CIs), F1, MCC, confusion matrix, n AD / n HC, at the frozen threshold;
* the source-only development estimate (D2: repeated D2 CV; D3: the release / A2 development CV when it matches) and
  the transfer degradation target - source of AUC and balanced accuracy with a CI from independent participant
  bootstraps of both;
* paired exploratory comparisons on the same target participants only: each GNN variant vs the reference baseline
  (Exp 1 LR on source PSD, same direction / inverse / mode), paired DeLong, Holm within each family;
* age: demographics by dataset and diagnosis, the age-only baseline (source fit -> target; within-source CV), the
  association of each unit's scores with age inside each diagnosis, the age-overlap subset, and y ~ age + score vs
  y ~ age (likelihood ratio). All exploratory; never used to select anything.
Window-level numbers are never used as independent observations.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from ..checkpoint import atomic_write_csv, atomic_write_json, code_identity
from ..evaluation.metrics import auc_delong, binary_metrics, delong_paired, wilson_ci
from . import d2store, namespace, plan, transfer

BOOT_METRICS = ("roc_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc")


# ------------------------------------------------------------------------------------------------ statistics
def stratified_bootstrap(y, score, threshold: float, n_boot: int, seed: int, metrics=BOOT_METRICS) -> pd.DataFrame:
    """Bootstrap draws of participant metrics, resampling participants WITH replacement within each diagnosis (the
    class sizes are kept). One row per draw."""
    y, score = np.asarray(y).astype(int), np.asarray(score, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_boot):
            i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
            m = binary_metrics(y[i], score[i], threshold)
            rows.append({k: m[k] for k in metrics})
    return pd.DataFrame(rows, columns=list(metrics))


def ci(draws: pd.Series | np.ndarray, level: float = 0.95) -> tuple[float, float]:
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


def participant_metrics(part: pd.DataFrame, threshold: float, n_boot: int, seed: int) -> tuple[dict, pd.DataFrame]:
    """Metrics of one set of participant scores (columns y, score); returns (metrics, bootstrap draws)."""
    y, s = part["y"].to_numpy(int), part["score"].to_numpy(float)
    m = binary_metrics(y, s, threshold)
    out = {"n": int(len(y)), "n_ad": int(y.sum()), "n_hc": int((y == 0).sum()), "threshold": threshold,
           **{k: m[k] for k in ("roc_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc",
                                "accuracy", "tn", "fp", "fn", "tp")}}
    if out["n_ad"] < 2 or out["n_hc"] < 2:
        return {**out, "note": "fewer than 2 participants in a class: no CI"}, pd.DataFrame()
    d = auc_delong(y, s)
    out.update({"auc_delong_low": d["ci_low"], "auc_delong_high": d["ci_high"], "auc_se": d["se"]})
    draws = stratified_bootstrap(y, s, threshold, n_boot, seed)
    for k in BOOT_METRICS:
        out[f"{k}_boot_low"], out[f"{k}_boot_high"] = ci(draws[k])
    out["sensitivity_wilson_low"], out["sensitivity_wilson_high"] = wilson_ci(int(m["tp"]), int(m["tp"] + m["fn"]))
    out["specificity_wilson_low"], out["specificity_wilson_high"] = wilson_ci(int(m["tn"]), int(m["tn"] + m["fp"]))
    return out, draws


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN kept)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def loglik(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def lr_test_adds_to_age(y, age, score) -> dict:
    """Likelihood-ratio test of y ~ age + score vs y ~ age (unpenalized logistic regressions, standardized
    predictors); exploratory."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y).astype(int)
    a = np.asarray(age, float)
    s = np.asarray(score, float)
    ok = np.isfinite(a) & np.isfinite(s)
    y, a, s = y[ok], a[ok], s[ok]
    if len(np.unique(y)) < 2 or len(y) < 6:
        return {"n": int(len(y)), "lr_stat": float("nan"), "p": float("nan")}
    z = lambda x: (x - x.mean()) / (x.std() or 1.0)  # noqa: E731
    X0, X1 = z(a)[:, None], np.c_[z(a), z(s)]
    m0 = LogisticRegression(C=1e12, max_iter=1000).fit(X0, y)        # effectively unpenalized, any sklearn version
    m1 = LogisticRegression(C=1e12, max_iter=1000).fit(X1, y)
    stat = 2 * (loglik(y, m1.predict_proba(X1)[:, 1]) - loglik(y, m0.predict_proba(X0)[:, 1]))
    return {"n": int(len(y)), "lr_stat": float(max(stat, 0.0)), "p": float(stats.chi2.sf(max(stat, 0.0), 1)),
            "coef_score_given_age": float(m1.coef_[0][1])}


# ------------------------------------------------------------------------------------------------ labels
def label_table(store, d3_partition: bool) -> pd.DataFrame:
    """key, dataset, y, label, age, sex, mmse (+ the D3 partition) of the ok participants of a store."""
    from ..evaluation.splits import subject_table, xu_partition

    s = store.subjects
    s = s[s["status"] == "ok"] if "status" in s else s
    cols = [c for c in ("key", "dataset", "y", "label", "age", "sex", "mmse") if c in s]
    t = s[cols].copy()
    if "sex" in t:
        t["sex"] = t["sex"].astype(str).str.strip().str.upper().str[:1].replace({"N": np.nan, "": np.nan})
    if d3_partition:
        dev, test = xu_partition(subject_table(store.windows))
        t["d3_partition"] = np.where(t["key"].isin(test), "holdout", np.where(t["key"].isin(dev), "development", ""))
    return t.reset_index(drop=True)


def demographics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ds, t in tables.items():
        for lab, g in t.groupby("label"):
            a = pd.to_numeric(g.get("age"), errors="coerce")
            rows.append({"dataset": ds, "diagnosis": lab, "n": int(len(g)), "age_mean": float(a.mean()),
                         "age_sd": float(a.std(ddof=1)), "age_min": float(a.min()), "age_max": float(a.max()),
                         "age_missing": int(a.isna().sum()),
                         "female": int((g.get("sex") == "F").sum()) if "sex" in g else None,
                         "male": int((g.get("sex") == "M").sum()) if "sex" in g else None,
                         "mmse_mean": float(pd.to_numeric(g.get("mmse"), errors="coerce").mean())
                         if "mmse" in g else float("nan")})
    return pd.DataFrame(rows)


def age_overlap(t: pd.DataFrame) -> tuple[float, float]:
    a = pd.to_numeric(t["age"], errors="coerce")
    ad, hc = a[t["y"] == 1], a[t["y"] == 0]
    return float(max(ad.min(), hc.min())), float(min(ad.max(), hc.max()))


def age_baseline(src: pd.DataFrame, tgt: pd.DataFrame, n_boot: int, seed: int, cv: dict) -> list[dict]:
    """Age-only logistic regression: source fit -> target (transfer), and participant CV inside the source."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold

    rows = []
    s = src[pd.to_numeric(src["age"], errors="coerce").notna()]
    t = tgt[pd.to_numeric(tgt["age"], errors="coerce").notna()]
    if s["y"].nunique() < 2 or t["y"].nunique() < 2:
        return [{"analysis": "age_only", "note": "age missing for a class"}]
    fit = LogisticRegression(max_iter=1000).fit(s[["age"]].astype(float), s["y"].astype(int))
    p = fit.predict_proba(t[["age"]].astype(float))[:, 1]
    m, _ = participant_metrics(t.assign(score=p), 0.5, n_boot, seed)
    rows.append({"analysis": "age_only_transfer", "coef_per_year": float(fit.coef_[0][0]), **m})
    rk = RepeatedStratifiedKFold(n_splits=int(cv["folds"]), n_repeats=int(cv["repeats"]), random_state=int(cv["seed"]))
    aucs = []
    X, y = s[["age"]].astype(float).to_numpy(), s["y"].astype(int).to_numpy()
    for r in range(int(cv["repeats"])):
        score = np.full(len(y), np.nan)
        for i, (tr, te) in enumerate(rk.split(X, y)):
            if i // int(cv["folds"]) != r:
                continue
            score[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(binary_metrics(y, score)["roc_auc"])
    rows.append({"analysis": "age_only_source_cv", "n": int(len(y)), "roc_auc": float(np.mean(aucs)),
                 "roc_auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float("nan")})
    return rows


# ------------------------------------------------------------------------------------------------ driver
def _scored(processed, u: dict, c: dict, labels: pd.DataFrame) -> pd.DataFrame | None:
    d = transfer.unit_dir(processed, u, c) / "transfer" / f"seed_{u['seed']}"
    if not (d / "transfer.done.json").exists():
        return None
    p = pd.read_csv(d / "target_participants_blind.csv")
    tr = json.loads((d / "training.json").read_text())
    p = p.merge(labels, on=["key", "dataset"], how="left", validate="one_to_one")
    if p["y"].isna().any():
        raise RuntimeError(f"{d}: target participants without labels {p.loc[p['y'].isna(), 'key'].tolist()}")
    p["threshold"], p["score_type"] = tr["threshold"], tr["score_type"]
    return p


def _source(processed, u, c, frozen, d3) -> tuple[dict, pd.DataFrame | None]:
    if u["direction"] == "D3_to_D2":
        rec = transfer.release_source_cv(processed, u, c, d3)
        if rec["status"] != "available":
            return rec, None
        part = pd.read_csv(rec["predictions"])
        part = part.groupby("key").agg(y=("y", "first"), score=("score", "mean")).reset_index()
        return rec, part
    d = transfer.unit_dir(processed, {**u, "direction": "D2_to_D3"}, c) / "source_cv"
    if not (d / "metrics.json").exists():
        return {"source": "D2", "status": "missing"}, None
    m = json.loads((d / "metrics.json").read_text())
    part = pd.read_csv(d / "predictions_participants.csv")
    part = part.groupby("key").agg(y=("y", "first"), score=("score", "mean")).reset_index()
    return {**m, "auc": m["auc_mean"], "balanced_accuracy": m["balanced_accuracy_mean"]}, part


def evaluate(processed, c: dict, stage: str = "primary") -> dict:
    """Evaluate every completed unit of the frozen campaign; writes ``evaluation/`` (inside the namespace)."""
    frozen = plan.require_frozen(processed, c)
    ev = c["evaluation"]
    nb, seed = int(ev["n_bootstrap"]), int(ev["bootstrap_seed"])
    out_dir = namespace.sub(processed, "evaluation", stage)
    rows, preds, ages, age_assoc = [], [], [], []
    units = plan.units(c, stage)
    stores = {}
    for u in units:
        sk = (u["inverse"], u["mode"])
        if sk not in stores:
            stores.clear()
            d3 = transfer.open_d3(processed, u["inverse"], u["mode"], c)
            d2 = transfer.open_d2(processed, u["inverse"], u["mode"], c)
            transfer.check_frozen_stores(frozen, u["inverse"], u["mode"], d3, d2)
            stores[sk] = (d3, d2, label_table(d3, True), label_table(d2, False))
        d3, d2, l3, l2 = stores[sk]
        d2store.assert_no_excluded(l2["key"], "D2 labels")
        tgt_labels = l2 if u["direction"] == "D3_to_D2" else l3
        p = _scored(processed, u, c, tgt_labels.drop(columns=["d3_partition"], errors="ignore"))
        if p is None:
            continue
        d2store.assert_no_excluded(p["key"], f"{u['direction']} target scores")
        if u["direction"] == "D3_to_D2" and not frozen.get("synthetic"):
            d2store.check_counts(p, d2store.EXPECTED_D2, "D3 -> D2 evaluation denominator")
        if "d3_partition" in tgt_labels:
            p = p.merge(tgt_labels[["key", "d3_partition"]], on="key", how="left")
        thr = float(p["threshold"].iloc[0])
        src_rec, src_part = _source(processed, u, c, frozen, d3) if stage == "primary" else ({}, None)
        if u["direction"] == "D2_to_D3" and src_part is not None:
            d2store.assert_no_excluded(src_part["key"], "D2 source-CV predictions")
            if not frozen.get("synthetic"):
                d2store.check_counts(src_part, d2store.EXPECTED_D2, "D2 source-CV denominator")
        strata = {"all": p}
        if u["direction"] == "D2_to_D3":
            strata.update({"d3_development": p[p["d3_partition"] == "development"],
                           "d3_holdout": p[p["d3_partition"] == "holdout"]})
        src_draws = (stratified_bootstrap(src_part["y"], src_part["score"], thr, nb, seed + 1)
                     if src_part is not None and src_part["y"].nunique() == 2 else None)
        for name, g in strata.items():
            m, draws = participant_metrics(g, thr, nb, seed)
            row = {**{k: u[k] for k in ("direction", "inverse", "mode", "exp", "experiment", "variant", "seed",
                                        "exploratory", "reference")},
                   "source": u["direction"][:2], "target": u["direction"][-2:], "stratum": name,
                   "score_type": p["score_type"].iloc[0], **m,
                   "source_cv_status": src_rec.get("status"), "source_cv_auc": src_rec.get("auc"),
                   "source_cv_balanced_accuracy": src_rec.get("balanced_accuracy")}
            for k, key in (("roc_auc", "auc"), ("balanced_accuracy", "balanced_accuracy")):
                if src_rec.get(key) is not None and np.isfinite(m.get(k, np.nan)):
                    row[f"degradation_{k}"] = m[k] - src_rec[key]
                    if src_draws is not None and len(draws):
                        lo, hi = ci(draws[k].to_numpy() - src_draws[k].to_numpy()[:len(draws)])
                        row[f"degradation_{k}_low"], row[f"degradation_{k}_high"] = lo, hi
            rows.append(row)
        if "age" in p:
            for lab, g in p.groupby("y"):
                a = pd.to_numeric(g["age"], errors="coerce")
                ok = a.notna()
                r = stats.spearmanr(g.loc[ok, "score"], a[ok]) if ok.sum() >= 4 else None
                age_assoc.append({**{k: u[k] for k in ("direction", "inverse", "mode", "experiment", "variant",
                                                       "seed")}, "diagnosis": "AD" if lab == 1 else "HC",
                                  "n": int(ok.sum()), "spearman_score_age": float(r.statistic) if r else np.nan,
                                  "p": float(r.pvalue) if r else np.nan})
            lo, hi = age_overlap(p)
            sub = p[(pd.to_numeric(p["age"], errors="coerce") >= lo) & (pd.to_numeric(p["age"], errors="coerce") <= hi)]
            enough = (sub["y"] == 1).sum() >= c["age"]["min_per_class"] and (sub["y"] == 0).sum() >= c["age"]["min_per_class"]
            mo = binary_metrics(sub["y"], sub["score"], thr) if enough else {}
            ages.append({**{k: u[k] for k in ("direction", "inverse", "mode", "experiment", "variant", "seed")},
                         "overlap_low": lo, "overlap_high": hi, "n_ad": int((sub["y"] == 1).sum()),
                         "n_hc": int((sub["y"] == 0).sum()), "estimable": bool(enough),
                         "roc_auc_overlap": mo.get("roc_auc", np.nan),
                         "balanced_accuracy_overlap": mo.get("balanced_accuracy", np.nan),
                         **{f"score_adds_to_age_{k}": v for k, v in lr_test_adds_to_age(p["y"], p["age"],
                                                                                        p["score"]).items()}})
        keep = ["key", "subject_id", "dataset", "y", "label", "score", "n_windows", "threshold", "score_type"] + \
            [x for x in ("d3_partition", "age", "sex") if x in p]       # on Drive only (participant metadata)
        q = p[keep].rename(columns={"score": "mean_score", "dataset": "target_dataset"})
        q["predicted"] = (q["mean_score"] >= thr).astype(int)
        for k in ("direction", "inverse", "mode", "experiment", "variant", "seed"):
            q[k] = u[k]
        q["source_dataset"] = u["direction"][:2]
        preds.append(q)
    long = pd.DataFrame(rows)
    pairs = paired_comparisons(preds, c)
    code = code_identity("cross_experiment")
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    if len(pred):
        pred["git_commit"], pred["source_sha256"] = code["git_commit"], code["source_sha256"]
    demo = {}
    for (inv, mode), (d3, d2, l3, l2) in stores.items():
        demo = demographics({"D3": l3, "D2": l2})
    age_rows = []
    if stores:
        d3, d2, l3, l2 = next(iter(stores.values()))
        cv = c["source_cv"]["D2"]
        for direction, (s, t) in {"D3_to_D2": (l3[l3["d3_partition"] == "development"], l2),
                                  "D2_to_D3": (l2, l3)}.items():
            age_rows += [{"direction": direction, **r} for r in age_baseline(s, t, nb, seed, cv)]
    files = {"metrics_long.csv": long, "predictions_participants.csv": pred, "paired_comparisons.csv": pairs,
             "age_overlap.csv": pd.DataFrame(ages), "age_score_association.csv": pd.DataFrame(age_assoc),
             "age_only_baseline.csv": pd.DataFrame(age_rows), "demographics.csv": demo}
    for name, df in files.items():
        atomic_write_csv(out_dir / name, df if isinstance(df, pd.DataFrame) else pd.DataFrame(df))
    summary = {"stage": stage, "n_units_evaluated": int(long[["direction", "inverse", "mode", "experiment", "variant",
                                                             "seed"]].drop_duplicates().shape[0]) if len(long) else 0,
               "n_units_planned": len(units), "frozen_campaign_hash": frozen["campaign_hash"],
               "code": code, "evaluated_at": datetime.now(timezone.utc).isoformat(),
               "disclosures": plan.DISCLOSURES, "synthetic": bool(frozen.get("synthetic")),
               "d2_cohort": d2store.cohort_spec()}
    atomic_write_json(out_dir / "summary.json", summary)
    return {"summary": summary, **files}


def paired_comparisons(preds: list[pd.DataFrame], c: dict) -> pd.DataFrame:
    """Each GNN variant vs the reference baseline on the same target participants (paired DeLong; Holm within each
    direction x inverse x mode family). Exploratory."""
    if not preds:
        return pd.DataFrame()
    p = pd.concat(preds, ignore_index=True)
    ref = c["evaluation"]["comparisons"]["reference_baseline"]
    rows = []
    for (d, inv, mode, seed), g in p.groupby(["direction", "inverse", "mode", "seed"]):
        r = g[(g["experiment"] == plan.variants(c)[ref["experiment"]][0]["experiment"])
              & (g["variant"] == ref["variant"])]
        if r.empty:
            continue
        base = r.set_index("key")
        for (exp, var), h in g.groupby(["experiment", "variant"]):
            if (exp, var) == (r["experiment"].iloc[0], ref["variant"]) or h["score_type"].iloc[0] != "probability" \
                    or exp.startswith(("exp0", "exp1")):
                continue
            j = h.set_index("key").join(base[["mean_score"]].rename(columns={"mean_score": "ref"}), how="inner")
            if j["y"].nunique() < 2:
                continue
            t = delong_paired(j["y"], j["mean_score"], j["ref"])
            rows.append({"direction": d, "inverse": inv, "mode": mode, "seed": seed, "experiment": exp,
                         "variant": var, "reference": f"{r['experiment'].iloc[0]}/{ref['variant']}",
                         "n": int(len(j)), "auc": t["auc_a"], "auc_reference": t["auc_b"], "diff": t["diff"],
                         "se": t["se"], "p": t["p"]})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        for _, idx in out.groupby(["direction", "inverse", "mode", "seed"]).groups.items():
            out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].to_numpy())
        out["label"] = "exploratory"
    return out
