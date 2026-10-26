"""Participant-level comparisons from saved predictions (amendment A3; nothing is trained, nothing is written).

Unit of analysis: the participant (one pooled out-of-fold score per development participant; windows are never samples).
Every comparison is on the SAME participants, with the paired DeLong test, a stratified participant bootstrap CI of the
AUC difference (the same resample for both models) and Holm adjustment within its comparison family. A difference is
called *supported* only if the Holm-adjusted p is below alpha AND the bootstrap CI excludes 0; everything else is
descriptive. All of this is conditional on models whose configurations were fixed in advance but whose out-of-fold
scores come from one 52-participant development set; no comparison is a test on new data.

Selection bias: the best of many variants is optimistic. ``selection_check`` estimates the optimism of the winner by a
participant bootstrap (winner chosen on the in-bag participants, scored on the out-of-bag ones) and a max-statistic
permutation test (the best AUC of M variants against label-permuted data, which keeps the correlation between variants).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..evaluation.metrics import auc_delong, delong_paired
from . import inventory

CTX = ["inverse", "mode", "atlas", "graph"]
KEY = CTX + ["experiment", "variant", "setting"]


def participant_scores(processed, rel_dir: str) -> pd.DataFrame:
    p = pd.read_csv(Path(processed) / rel_dir / "predictions_participants.csv")
    return p.groupby("key").agg(y=("y", "first"), score=("score", "mean"))


def _auc_matrix(yp: np.ndarray, S: np.ndarray, idx_pos: np.ndarray, idx_neg: np.ndarray) -> np.ndarray:
    """AUC of every row of S (m x n) for B resamples given as index arrays (B x n_pos), (B x n_neg) -> (B x m)."""
    sp, sn = S[:, idx_pos], S[:, idx_neg]                              # (m, B, np), (m, B, nn)
    out = np.empty((idx_pos.shape[0], S.shape[0]))
    for k in range(S.shape[0]):
        a = sp[k][:, :, None]
        b = sn[k][:, None, :]
        out[:, k] = ((a > b).sum((1, 2)) + 0.5 * (a == b).sum((1, 2))) / (a.shape[1] * b.shape[2])
    return out


def boot_aucs(y: np.ndarray, S: np.ndarray, n_boot: int, seed: int, chunk: int = 200) -> np.ndarray:
    """Stratified participant bootstrap: (n_boot x m) AUCs, the same resamples for every model (paired)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    parts = []
    for s0 in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s0)
        ip = rng.choice(pos, size=(b, len(pos)))
        in_ = rng.choice(neg, size=(b, len(neg)))
        parts.append(_auc_matrix(y, S, ip, in_))
    return np.vstack(parts)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    run, m = 0.0, len(order)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out


def compare(processed, ra: dict, rb: dict, n_boot: int, seed: int) -> dict:
    """Paired comparison of two result rows (A minus B) on their common participants."""
    pa, pb = participant_scores(processed, ra["rel_dir"]), participant_scores(processed, rb["rel_dir"])
    common = sorted(set(pa.index) & set(pb.index))
    same = len(common) == len(pa) == len(pb) and bool((pa.loc[common, "y"] == pb.loc[common, "y"]).all())
    y = pa.loc[common, "y"].to_numpy(int)
    sa, sb = pa.loc[common, "score"].to_numpy(), pb.loc[common, "score"].to_numpy()
    out = {"n_participants": len(common), "same_participants": same, "auc_a": np.nan, "auc_b": np.nan,
           "auc_diff": np.nan, "boot_low": np.nan, "boot_high": np.nan, "delong_se": np.nan, "delong_p": np.nan}
    if len(np.unique(y)) < 2 or not same:
        return out
    t = delong_paired(y, sa, sb)
    d = boot_aucs(y, np.vstack([sa, sb]), n_boot, seed)
    diff = d[:, 0] - d[:, 1]
    out.update(auc_a=t["auc_a"], auc_b=t["auc_b"], auc_diff=t["diff"], delong_se=t["se"], delong_p=t["p"],
               boot_low=float(np.percentile(diff, 2.5)), boot_high=float(np.percentile(diff, 97.5)))
    return out


def finish(rows: list[dict], alpha: float) -> pd.DataFrame:
    """Holm adjustment within each ``family`` and the supported / descriptive label."""
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    t["p_holm"] = np.nan
    for _, idx in t.groupby("family").groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "delong_p"].to_numpy())
    t["n_in_family"] = t.groupby("family")["family"].transform("size")
    t["supported"] = (t["p_holm"] < alpha) & ((t["boot_low"] > 0) | (t["boot_high"] < 0))
    t["conclusion"] = np.where(t["supported"], "supported (Holm p < alpha and bootstrap CI excludes 0)",
                               "descriptive only (not supported after multiplicity correction)")
    return t


# ------------------------------------------------------------------------------------------------ comparison sets
def _lookup(dev: pd.DataFrame) -> dict:
    return {tuple(r[k] for k in KEY): r for r in dev.to_dict("records")}


def _row(lk, inverse, mode, atlas, graph, experiment, variant, setting="xu_dev_cv"):
    return lk.get((inverse, mode, atlas, graph, experiment, variant, setting))


def _conn(variant: str) -> str | None:
    t = variant.split("__")
    return t[0] if t[0] in ("plv", "ciplv") else None


def comparison_sets(processed, table: pd.DataFrame, atlas: str, graph: str, cfg: dict | None = None) -> pd.DataFrame:
    """Every pre-specified paired comparison that has both results (A minus B), Holm-adjusted within its family."""
    from . import namespace as ns

    cfg = cfg or ns.stats_config()
    n_boot, seed, alpha = cfg["n_bootstrap"], cfg["seed"], cfg["alpha"]
    dev = table[table["setting"] == "xu_dev_cv"]
    lk = _lookup(dev)
    rows = []

    def add(family, label, a, b, **extra):
        if a is None or b is None:
            return
        rows.append({"family": family, "comparison": label, "experiment_a": a["experiment"], "variant_a": a["variant"],
                     "experiment_b": b["experiment"], "variant_b": b["variant"], "inverse_a": a["inverse"],
                     "inverse_b": b["inverse"], "mode_a": a["mode"], "mode_b": b["mode"], "atlas_a": a["atlas"],
                     "atlas_b": b["atlas"], **extra, **compare(processed, a, b, n_boot, seed)})

    def g(exp_name):                                         # graph of an experiment's recorded context
        return "none" if exp_name.split("_")[0] in inventory.CLASSICAL else graph

    def rec(inv, mode, experiment, variant):                 # recorded-context lookup
        e = experiment.split("_")[0]
        at = "dk68" if e == "exp0" else atlas
        return _row(lk, "sensor" if e == "exp0" else inv, mode, at, g(experiment), experiment, variant)

    exps = sorted(dev["experiment"].unique()) if len(dev) else []
    base_exp = [e for e in exps if e.startswith(cfg["baselines"]["experiment_prefix"])]
    base_vars = sorted(dev[dev["experiment"].isin(base_exp)]["variant"].unique()) if base_exp else []
    inverses = ("lcmv", "wmne")
    # 1. classical spectral baseline vs each GNN family (the four pre-specified Exp 1 variants, all of them)
    for fam in cfg["gnn_families"]:
        for inv in inverses:
            for v in sorted(dev[dev["experiment"] == fam]["variant"].unique()):
                for be in base_exp:
                    for bv in base_vars:
                        add(f"classical-vs-GNN | {fam}", "GNN minus classical spectral baseline (same inverse, mode B)",
                            rec(inv, "B", fam, v), rec(inv, "B", be, bv))
    # 1b. source space vs sensor space (Exp 1 minus Exp 0, same variant)
    for inv in inverses:
        for v in base_vars:
            add("source-vs-sensor | classical", "source (Exp 1) minus sensor (Exp 0), same variant",
                rec(inv, "B", base_exp[0], v) if base_exp else None, rec(inv, "B", "exp0_sensor_psd_de", v))
    # 2. LCMV vs wMNE, 4. mode B vs A: matched on every other field
    for e in exps:
        if e.startswith("exp0"):
            continue
        for v in sorted(dev[dev["experiment"] == e]["variant"].unique()):
            for mode in ("B", "A"):
                add("LCMV-vs-wMNE", "LCMV minus wMNE (matched participants)", rec("lcmv", mode, e, v),
                    rec("wmne", mode, e, v), context_mode=mode)
            for inv in inverses:
                add("modeB-vs-modeA", "mode B (all windows) minus mode A (10 windows)", rec(inv, "B", e, v),
                    rec(inv, "A", e, v))
    for v in sorted(dev[dev["experiment"] == "exp0_sensor_psd_de"]["variant"].unique()):
        add("modeB-vs-modeA", "mode B (all windows) minus mode A (10 windows)", rec("sensor", "B", "exp0_sensor_psd_de", v),
            rec("sensor", "A", "exp0_sensor_psd_de", v))
    # 3. PLV vs ciPLV (GNN variants whose names differ only in the connectivity token)
    for e in exps:
        if e.split("_")[0] in inventory.CLASSICAL:
            continue
        for v in sorted(dev[dev["experiment"] == e]["variant"].unique()):
            if _conn(v) == "plv":
                for inv in inverses:
                    add("PLV-vs-ciPLV", "PLV minus |ciPLV| (same architecture, features)", rec(inv, "B", e, v),
                        rec(inv, "B", e, "ciplv" + v[3:]))
    # 5. Exp 8a ablations
    e8 = "exp8a_xu_dual_gcn_surface"
    for a, b in cfg["exp8a_pairs"]:
        for inv in inverses:
            add("Exp8a-ablations", f"{a} minus {b}", rec(inv, "B", e8, a), rec(inv, "B", e8, b))
    # 5b. Exp 6 ablations
    e6 = "exp6_psd_strength"
    for a, b in cfg["exp6_pairs"]:
        for inv in inverses:
            add("Exp6-ablations", f"{a} minus {b}", rec(inv, "B", e6, a), rec(inv, "B", e6, b))
    # 6. architecture complexity: variant minus the minimal band-averaged GCN of the same connectivity
    ref = cfg["complexity_reference"]
    ref_exp = next((e for e in exps if e == ref["experiment"]), ref["experiment"])
    for fam in cfg["complexity_families"]:
        for v in sorted(dev[dev["experiment"] == fam]["variant"].unique()):
            c = _conn(v)
            if c is None:
                continue
            for inv in inverses:
                add(f"complexity | {fam}", f"variant minus Exp 4 {ref['variant_template']} (same connectivity)",
                    rec(inv, "B", fam, v), rec(inv, "B", ref_exp, ref["variant_template"].format(conn=c)))
    return finish(rows, alpha)


# ------------------------------------------------------------------------------------------------ selection bias
def _scores_matrix(processed, rows: list[dict]):
    series = [participant_scores(processed, r["rel_dir"]) for r in rows]
    common = sorted(set.intersection(*[set(s.index) for s in series])) if series else []
    y = series[0].loc[common, "y"].to_numpy(int)
    S = np.vstack([s.loc[common, "score"].to_numpy() for s in series])
    return y, S


def selection_check(processed, rows: list[dict], n_boot: int, n_perm: int, seed: int) -> dict:
    """Optimism of the best of ``rows`` (one candidate set) from the saved participant scores."""
    y, S = _scores_matrix(processed, rows)
    m = len(rows)
    ranks = np.vstack([rankdata(s) for s in S])
    npos, nneg = int(y.sum()), int((y == 0).sum())
    auc = (ranks[:, y == 1].sum(1) - npos * (npos + 1) / 2) / (npos * nneg)
    best = int(np.argmax(auc))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    gaps, oob_auc, in_auc = [], [], []
    for _ in range(n_boot):
        ip, in_ = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        oob_p, oob_n = np.setdiff1d(pos, ip), np.setdiff1d(neg, in_)
        if len(oob_p) < 2 or len(oob_n) < 2:
            continue
        a_in = _auc_matrix(y, S, ip[None], in_[None])[0]
        w = int(np.argmax(a_in))
        a_out = _auc_matrix(y, S[w:w + 1], oob_p[None], oob_n[None])[0, 0]
        in_auc.append(a_in[w])
        oob_auc.append(a_out)
        gaps.append(a_in[w] - a_out)
    perm_max = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        k = int(yp.sum())
        perm_max[i] = ((ranks[:, yp == 1].sum(1) - k * (k + 1) / 2) / (k * (len(yp) - k))).max()
    return {"n_candidates": m, "n_participants": int(len(y)), "best_variant": rows[best]["variant"],
            "best_experiment": rows[best]["experiment"], "best_inverse": rows[best]["inverse"],
            "best_auc_observed": float(auc[best]), "best_auc_ci_delong": [float(x) for x in
                                                                          (lambda d: (d["ci_low"], d["ci_high"]))(
                                                                              auc_delong(y, S[best]))],
            "mean_inbag_winner_auc": float(np.mean(in_auc)), "mean_oob_winner_auc": float(np.mean(oob_auc)),
            "optimism_estimate": float(np.mean(gaps)), "n_valid_bootstraps": len(gaps),
            "optimism_corrected_best_auc": float(auc[best] - np.mean(gaps)),
            "max_null_auc_mean": float(perm_max.mean()), "max_null_auc_p95": float(np.percentile(perm_max, 95)),
            "p_best_vs_chance_maxstat": float((1 + np.sum(perm_max >= auc[best])) / (1 + n_perm)),
            "note": "descriptive: the winner is the best of the candidate set; the optimism estimate has high variance "
                    "with 52 participants"}


def winner_difference(processed, set_a: list[dict], set_b: list[dict], n_boot: int, seed: int) -> dict:
    """Selection-aware difference between the best of set A and the best of set B: per participant bootstrap, each
    winner is chosen on the in-bag participants and scored on the out-of-bag ones (same participants for both sets)."""
    rows = set_a + set_b
    y, S = _scores_matrix(processed, rows)
    na = len(set_a)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = []
    for _ in range(n_boot):
        ip, in_ = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        oob_p, oob_n = np.setdiff1d(pos, ip), np.setdiff1d(neg, in_)
        if len(oob_p) < 2 or len(oob_n) < 2:
            continue
        a_in = _auc_matrix(y, S, ip[None], in_[None])[0]
        wa, wb = int(np.argmax(a_in[:na])), na + int(np.argmax(a_in[na:]))
        a_out = _auc_matrix(y, S[[wa, wb]], oob_p[None], oob_n[None])[0]
        diffs.append(a_out[0] - a_out[1])
    d = np.array(diffs)
    full = _auc_matrix(y, S, pos[None], neg[None])[0]
    return {"best_a_observed": float(full[:na].max()), "best_b_observed": float(full[na:].max()),
            "observed_difference_of_maxima": float(full[:na].max() - full[na:].max()),
            "oob_difference_mean": float(d.mean()), "oob_difference_ci95": [float(np.percentile(d, 2.5)),
                                                                           float(np.percentile(d, 97.5))],
            "n_valid_bootstraps": int(len(d)), "n_participants": int(len(y))}


def selection_report(processed, table: pd.DataFrame, atlas: str, graph: str, cfg: dict | None = None) -> dict:
    """Selection checks of the candidate sets that matter for the reported 'best' models, per inverse."""
    from . import namespace as ns

    cfg = cfg or ns.stats_config()
    n_boot, n_perm, seed = cfg["n_bootstrap"], cfg["n_permutation"], cfg["seed"]
    dev = table[(table["setting"] == "xu_dev_cv") & (table["mode"] == "B")]
    on = dev[(dev["atlas"] == atlas) & dev["graph"].isin([graph, "none"])]
    out = {"sets": {}, "winner_differences": {}}

    def cand(inv, exps):
        sub = on[(on["inverse"] == inv) & on["exp"].isin(exps)]
        return sub.to_dict("records")
    sets = {"classical (Exp 1)": ["exp1"], "GNN confirmatory (Exp 2-6)": ["exp2", "exp3", "exp4", "exp5", "exp6"],
            "Exp 8a (amended)": ["exp8a"], "GNN all but Exp 7 (Exp 2-6, 8a)": ["exp2", "exp3", "exp4", "exp5", "exp6",
                                                                                 "exp8a"]}
    for inv in ("lcmv", "wmne"):
        for name, exps in sets.items():
            rows = cand(inv, exps)
            if len(rows) >= 2:
                out["sets"][f"{name} | {inv}"] = selection_check(processed, rows, n_boot, n_perm, seed)
        for name in ("GNN confirmatory (Exp 2-6)", "Exp 8a (amended)", "GNN all but Exp 7 (Exp 2-6, 8a)"):
            a, b = cand(inv, sets["classical (Exp 1)"]), cand(inv, sets[name])
            if a and b:
                out["winner_differences"][f"classical (Exp 1) minus {name} | {inv}"] = winner_difference(
                    processed, a, b, n_boot, seed)
    return out
