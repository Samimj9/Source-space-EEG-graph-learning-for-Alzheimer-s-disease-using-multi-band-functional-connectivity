"""The consolidated development-CV report (Exp 0-8a across both namespaces) and the descriptive holdout report.

Both read saved outputs only and write into a NEW time-stamped directory of the A3 namespace; nothing is overwritten and
nothing outside ``FINAL_v1/amendments/A3_final/`` is written. The development report opens development-CV results only
(``inventory.discover(settings=("xu_dev_cv",))``); the holdout report opens holdout results only and takes the
development AUCs from the frozen records, in a separate, labelled table. The two never share a metrics table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import release
from ..experiments import decisions
from . import inventory, stats
from . import namespace as ns

COLUMNS = ["family", "family_title", "experiment", "variant", "inverse", "mode", "atlas", "graph", "namespace", "role",
           "git_commit", "n_participants", "auc", "auc_ci_low", "auc_ci_high", "auc_boot_low", "auc_boot_high",
           "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc", "tn", "fp", "fn", "tp", "mean_fold_auc",
           "sd_fold_auc", "min_fold_auc", "max_fold_auc", "window_auc", "threshold", "score_type", "rel_dir"]
LEGEND = ("**Roles.** *confirmatory*: pre-specified, recorded atlas and graph, window mode B. *amended (A2)*: Exp 8a, "
          "designed after Exp 0-7 results were seen and before any Exp 8a result (amendment A2). *exploratory*: Exp 7 "
          "(designed after earlier results) and the atlas / graph comparison runs. *sensitivity*: window mode A "
          "(10 windows per participant). Development CV only: the pooled out-of-fold participant AUC with its DeLong "
          "95% CI; fold-level statistics are secondary.")


def tidy(t: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLUMNS if c in t]
    return t[cols].sort_values(["family", "experiment", "variant", "inverse", "mode", "atlas", "graph"],
                               ignore_index=True)


def md_table(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    cols = [c for c in cols if c in df]
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df[cols].itertuples(index=False):
        out += "| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) and not pd.isna(v)
                                 else ("" if pd.isna(v) else str(v)) for v in r) + " |\n"
    return out


# ------------------------------------------------------------------------------------------------ interpretation
def _pairs(cmp: pd.DataFrame, family_prefix: str) -> pd.DataFrame:
    return cmp[cmp["family"].str.startswith(family_prefix)] if len(cmp) else cmp


def _summ(c: pd.DataFrame) -> dict:
    if not len(c):
        return {"n": 0}
    d = c["auc_diff"].dropna()
    return {"n": int(len(c)), "median_diff": float(d.median()) if len(d) else np.nan,
            "min_diff": float(d.min()) if len(d) else np.nan, "max_diff": float(d.max()) if len(d) else np.nan,
            "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
            "n_supported_positive": int((c["supported"] & (c["auc_diff"] > 0)).sum()),
            "n_supported_negative": int((c["supported"] & (c["auc_diff"] < 0)).sum())}


def _fmt(s: dict) -> str:
    if not s.get("n"):
        return "no matched pair has completed results."
    return (f"{s['n']} matched comparisons; AUC difference median {s['median_diff']:+.3f} "
            f"(range {s['min_diff']:+.3f} to {s['max_diff']:+.3f}); first model higher in {s['n_positive']}, lower in "
            f"{s['n_negative']}; supported after Holm correction: {s['n_supported_positive']} higher, "
            f"{s['n_supported_negative']} lower.")


def interpretation(primary: pd.DataFrame, cmp: pd.DataFrame, sel: dict) -> str:
    """Evidence-based reading of the saved tables. Every number is computed here from saved results; a statement is
    *supported* only when the corresponding Holm-adjusted participant-level test and bootstrap CI say so."""
    L = ["# Interpretation of the D3 development results (computed from the saved tables)\n",
         "Descriptive findings and statistically supported conclusions are kept apart. All results are development-CV "
         "results on 52 participants; nothing here is a test on new data, and no holdout result is used.\n"]
    best = primary.sort_values("auc", ascending=False)
    if len(best):
        for inv in ("lcmv", "wmne"):
            for label, sub in (("classical (Exp 1)", best[best["exp"] == "exp1"]),
                               ("GNN, confirmatory + amended (Exp 2-6, 8a)",
                                best[best["exp"].isin(["exp2", "exp3", "exp4", "exp5", "exp6", "exp8a"])])):
                s = sub[sub["inverse"] == inv]
                if len(s):
                    r = s.iloc[0]
                    L.append(f"- *Descriptive.* Highest observed participant AUC, {label}, {inv}: {r['auc']:.3f} "
                             f"(DeLong 95% CI {r['auc_ci_low']:.3f}-{r['auc_ci_high']:.3f}; {r['experiment']} / "
                             f"{r['variant']}). This is the best of {len(s)} variants and therefore optimistic.")
    L.append("\n## Selection bias (best of many variants)\n")
    for k, v in sel.get("sets", {}).items():
        L.append(f"- {k}: best observed AUC {v['best_auc_observed']:.3f} ({v['best_variant']}) of {v['n_candidates']} "
                 f"candidates; bootstrap optimism estimate {v['optimism_estimate']:+.3f} (out-of-bag winner AUC "
                 f"{v['mean_oob_winner_auc']:.3f}); under label permutation the best of {v['n_candidates']} reaches "
                 f"{v['max_null_auc_mean']:.3f} on average (95th percentile {v['max_null_auc_p95']:.3f}); max-statistic "
                 f"p = {v['p_best_vs_chance_maxstat']:.4f}. *Descriptive*: the optimism estimate is noisy at n = 52.")
    for k, v in sel.get("winner_differences", {}).items():
        lo, hi = v["oob_difference_ci95"]
        L.append(f"- {k}: observed difference of the maxima {v['observed_difference_of_maxima']:+.3f}; selection-aware "
                 f"out-of-bag difference {v['oob_difference_mean']:+.3f} (bootstrap 95% CI {lo:+.3f} to {hi:+.3f}). "
                 + ("*Supported*: CI excludes 0." if lo > 0 or hi < 0 else "*Not supported*: CI includes 0."))
    q = [("Do GNNs outperform the classical spectral baselines?", "classical-vs-GNN",
          "GNN variant minus the four pre-specified Exp 1 baselines (same inverse, mode B)"),
         ("Does anatomical adjacency help (Exp 8a)?", "Exp8a-ablations", "full model minus each ablation / alternative"),
         ("Does connectivity-derived information add beyond spectral features (Exp 6 ablations)?",
          "Exp6-ablations", "full minus ablated variant"),
         ("Does a more complicated architecture help consistently?", "complexity",
          "variant minus the minimal band-averaged GCN of the same connectivity"),
         ("PLV versus |ciPLV|", "PLV-vs-ciPLV", "PLV minus |ciPLV|"),
         ("LCMV versus wMNE", "LCMV-vs-wMNE", "LCMV minus wMNE, matched participants"),
         ("Fewer windows (mode B vs mode A)", "modeB-vs-modeA", "all windows minus 10 windows"),
         ("Source space versus sensor space", "source-vs-sensor", "Exp 1 minus Exp 0")]
    for title, prefix, what in q:
        L.append(f"\n## {title}\n\n*{what}.*\n")
        sub = _pairs(cmp, prefix)
        if prefix in ("classical-vs-GNN", "complexity"):
            for fam in sorted(sub["family"].unique()) if len(sub) else []:
                L.append(f"- {fam.split(' | ')[1]}: " + _fmt(_summ(sub[sub["family"] == fam])))
        else:
            L.append("- " + _fmt(_summ(sub)))
        if len(sub):
            sh, sl = int((sub["supported"] & (sub["auc_diff"] > 0)).sum()), int((sub["supported"] & (sub["auc_diff"] < 0)).sum())
            L.append("\n**Statistically supported conclusion:** " + (
                f"{sh} comparison(s) with the first model higher and {sl} with it lower pass Holm-adjusted DeLong "
                "p < alpha with a bootstrap CI excluding 0 (rows with `supported` in `comparisons.csv`); every other "
                "difference is descriptive." if sh + sl else
                "none. No comparison in this set is supported after multiplicity correction; the differences above are "
                "descriptive and do not establish superiority of either model."))
    L.append("""
## Exp 8a and functional connectivity

The two-anatomical-layer variant (`abl_anatomical_both`) removes the functional adjacency from message passing but
still receives node-strength features, which are summaries of the functional-connectivity matrix; it therefore still
uses functional-connectivity information. Only the `plv__psd` / `plv__de` variants (PSD / DE node features only) and the
Exp 6 `no_strength` ablation exclude it, so compare those, not `abl_anatomical_both`, when asking whether connectivity
adds information beyond spectral features.

## Limitations

- 52 development participants (36 AD / 29 HC overall): wide AUC confidence intervals; participant-level paired
  tests have little power, and a non-significant difference is not evidence of equivalence.
- Many configurations were evaluated on the same participants; the best of them is optimistic (see selection bias).
  Comparisons are corrected by Holm within a comparison family, not across families.
- Exp 7 was designed after earlier results, Exp 8a after Exp 0-7 results (amendment A2); atlas and graph decisions were
  recorded after development-CV comparisons. Exp 8a's anatomical prior is a fixed cortical-surface contiguity prior
  (not structural connectivity) on 12 large regions.
- Source localization from 19-channel EEG with a template head model is low-resolution; ROI signals leak into each other
  (median cross-talk 0.38-0.54 for lobes12 and r18 in the exact-BEM analysis, methodology log entries 59a-59b); results depend on the inverse method, the atlas and the
  preprocessing, which were fixed upstream of this analysis.
- The 13-participant holdout was inspected in earlier work; its results are descriptive and not external validation.
""")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ figures
OI = {"lcmv": "#0072B2", "wmne": "#D55E00", "sensor": "#009E73", "B": "#0072B2", "A": "#E69F00"}


def _save(fig, base: Path) -> list[str]:
    out = []
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        p = base.with_suffix("." + ext)
        fig.savefig(p, bbox_inches="tight", **kw)
        out.append(p.name)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out


def figures(primary: pd.DataFrame, allr: pd.DataFrame, cmp: pd.DataFrame, sel: dict, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    made = []
    out.mkdir(parents=True, exist_ok=True)
    p = primary.sort_values(["family", "experiment", "variant"]).reset_index(drop=True)
    if len(p):
        labels = list(dict.fromkeys(zip(p["family"], p["variant"])))
        fig, ax = plt.subplots(figsize=(7, max(4, 0.17 * len(labels))))
        for i, (fam, var) in enumerate(labels):
            for inv, dy in (("lcmv", -0.15), ("wmne", 0.15), ("sensor", 0.0)):
                r = p[(p["family"] == fam) & (p["variant"] == var) & (p["inverse"] == inv)]
                if len(r):
                    r = r.iloc[0]
                    ax.errorbar(r["auc"], i + dy, xerr=[[r["auc"] - r["auc_ci_low"]], [r["auc_ci_high"] - r["auc"]]],
                                fmt="o", ms=3, lw=0.8, color=OI[inv], label=inv)
        ax.set_yticks(range(len(labels)), [f"{a}  {b}" for a, b in labels], fontsize=5)
        ax.invert_yaxis()
        ax.axvline(0.5, ls=":", c="k", lw=0.6)
        ax.set_xlabel("participant ROC-AUC, development CV (DeLong 95% CI)")
        h, l = ax.get_legend_handles_labels()
        ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), frameon=False, loc="lower right")
        ax.set_title("All primary-context variants (mode B, recorded atlas and graph); the best is optimistic",
                     fontsize=8)
        made += _save(fig, out / "fig1_primary_forest")
        fig, ax = plt.subplots(figsize=(5.5, 3))
        fams = list(dict.fromkeys(p["family"]))
        for i, f in enumerate(fams):
            for inv, dx in (("lcmv", -0.12), ("wmne", 0.12), ("sensor", 0)):
                v = p[(p["family"] == f) & (p["inverse"] == inv)]["auc"]
                ax.scatter(np.full(len(v), i + dx) + np.random.default_rng(0).uniform(-.05, .05, len(v)), v, s=9,
                           color=OI[inv], alpha=.8, label=inv)
        ax.set_xticks(range(len(fams)), fams)
        ax.set_ylabel("participant AUC")
        ax.axhline(0.5, ls=":", c="k", lw=0.6)
        h, l = ax.get_legend_handles_labels()
        ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), frameon=False)
        made += _save(fig, out / "fig2_family_summary")
    for fam, name, xl, yl in (("LCMV-vs-wMNE", "fig3_lcmv_vs_wmne", "wMNE AUC", "LCMV AUC"),
                              ("modeB-vs-modeA", "fig4_modeB_vs_modeA", "mode A AUC (10 windows)",
                               "mode B AUC (all windows)"), ("PLV-vs-ciPLV", "fig7_plv_vs_ciplv", "|ciPLV| AUC",
                                                              "PLV AUC")):
        c = cmp[cmp["family"] == fam] if len(cmp) else cmp
        if len(c):
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.scatter(c["auc_b"], c["auc_a"], s=10, c=["#CC79A7" if s else "#555555" for s in c["supported"]])
            lim = [min(c["auc_a"].min(), c["auc_b"].min()) - .02, max(c["auc_a"].max(), c["auc_b"].max()) + .02]
            ax.plot(lim, lim, "k:", lw=0.6)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{fam}: each point one matched variant\n(pink = supported after Holm)", fontsize=7)
            made += _save(fig, out / name)
    c = cmp[cmp["family"].isin(["Exp8a-ablations", "Exp6-ablations"])] if len(cmp) else cmp
    if len(c):
        c = c.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.2 * len(c))))
        for i, r in c.iterrows():
            ax.errorbar(r["auc_diff"], i, xerr=[[r["auc_diff"] - r["boot_low"]], [r["boot_high"] - r["auc_diff"]]],
                        fmt="o", ms=3, lw=0.8, color=OI.get(r["inverse_a"], "k"))
        ax.set_yticks(range(len(c)), [f"{r['inverse_a']}  {r['comparison']}" for _, r in c.iterrows()], fontsize=5)
        ax.invert_yaxis()
        ax.axvline(0, c="k", lw=0.6)
        ax.set_xlabel("AUC difference (paired participant bootstrap 95% CI); uncorrected")
        made += _save(fig, out / "fig5_ablations")
    if sel.get("sets"):
        ks = list(sel["sets"])
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(ks) + 1))
        for i, k in enumerate(ks):
            v = sel["sets"][k]
            ax.plot([v["optimism_corrected_best_auc"], v["best_auc_observed"]], [i, i], c="#999999", lw=1)
            ax.scatter(v["best_auc_observed"], i, c="#D55E00", s=14, zorder=3, label="observed best")
            ax.scatter(v["optimism_corrected_best_auc"], i, c="#0072B2", s=14, zorder=3, label="optimism-corrected")
            ax.scatter(v["max_null_auc_p95"], i, marker="|", c="k", s=40, label="null max 95th pct")
        ax.set_yticks(range(len(ks)), ks, fontsize=5)
        ax.invert_yaxis()
        h, l = ax.get_legend_handles_labels()
        ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), frameon=False, fontsize=6)
        ax.set_xlabel("AUC of the best candidate")
        made += _save(fig, out / "fig6_selection_bias")
    return made


# ------------------------------------------------------------------------------------------------ development report
def decisions_of(processed) -> tuple[str, str]:
    from .freeze import recorded_decisions

    return recorded_decisions(processed)


def development_report(processed, out_dir=None, atlas: str | None = None, graph: str | None = None, cfg=None,
                       make_figures: bool = True) -> dict:
    """Consolidated report of every completed development-CV result of both namespaces."""
    from .. import provenance

    cfg = cfg or ns.stats_config()
    if atlas is None or graph is None:
        atlas, graph = decisions_of(processed)
    t, other = inventory.discover(processed, settings=("xu_dev_cv",))
    t = inventory.with_roles(t, atlas, graph)
    out = Path(out_dir or ns.root(processed) / "reports" / f"development_{ns.stamp()}")
    if out.exists():
        raise FileExistsError(f"{out} exists; reports are never overwritten")
    out.mkdir(parents=True)
    primary = t[(t["mode"] == "B") & t["role"].str.startswith(("confirmatory", "amended"))].copy()
    exploratory = t[t["role"].str.startswith("exploratory")]
    cmp = stats.comparison_sets(processed, t, atlas, graph, cfg) if len(t) else pd.DataFrame()
    sel = stats.selection_report(processed, t, atlas, graph, cfg) if len(t) else {"sets": {}, "winner_differences": {}}
    files = {}

    def csv(name, df):
        p = out / name
        df.to_csv(p, index=False)
        files[name] = ns.sha256_file(p)

    csv("development_results_all.csv", tidy(t) if len(t) else t)
    csv("development_results_primary.csv", tidy(primary) if len(primary) else primary)
    csv("development_results_exploratory.csv", tidy(exploratory) if len(exploratory) else exploratory)
    csv("not_completed_provenance.csv", other if len(other) else pd.DataFrame(
        columns=["namespace", "experiment", "variant", "state", "detail"]))
    csv("comparisons.csv", cmp)
    fam = (primary.groupby(["family", "inverse"]).agg(n_variants=("auc", "size"), auc_median=("auc", "median"),
                                                      auc_min=("auc", "min"), auc_max=("auc", "max")).reset_index()
           if len(primary) else pd.DataFrame())
    csv("family_summary.csv", fam)
    (out / "selection_check.json").write_text(json.dumps(sel, indent=2))
    files["selection_check.json"] = ns.sha256_file(out / "selection_check.json")
    cols = ["variant", "inverse", "mode", "n_participants", "auc", "auc_ci_low", "auc_ci_high", "balanced_accuracy",
            "sensitivity", "specificity", "f1", "mcc", "tn", "fp", "fn", "tp", "mean_fold_auc"]
    md = [f"# D3 development-CV results, Exp 0-8a (consolidated)\n\nRecorded atlas **{atlas}**, graph construction "
          f"**{graph}**. Release namespace at `{ns.part_pin('release')[:10]}`, Exp 8a (amendment A2) at "
          f"`{ns.part_pin('A2')[:10]}`. {len(t)} completed variants are listed in `development_results_all.csv`; "
          f"{len(other)} directories are not completed results and are listed only as provenance "
          "(`not_completed_provenance.csv`, includes the failed original Exp 8 units).\n\n" + LEGEND + "\n\n"
          "No holdout result is used or shown here.\n"]
    for f in dict.fromkeys(primary["family"]) if len(primary) else []:
        g = primary[primary["family"] == f].sort_values(["variant", "inverse"])
        md.append(f"\n## {f} · {g['family_title'].iloc[0]} · {g['role'].iloc[0]}\n\n" + md_table(g, cols))
    if len(exploratory) or len(t):
        md.append("\n## Not in the primary table (sensitivity and exploratory runs)\n\nSee "
                  "`development_results_all.csv` (column `role`) for window mode A, Exp 7 and the atlas / graph "
                  "comparison runs; every completed variant is preserved there.\n")
    (out / "development_results.md").write_text("\n".join(md))
    (out / "interpretation.md").write_text(interpretation(primary, cmp, sel))
    for n in ("development_results.md", "interpretation.md"):
        files[n] = ns.sha256_file(out / n)
    figs = figures(primary, t, cmp, sel, out / "figures") if make_figures and len(t) else []
    man = {"amendment": ns.AID, "kind": "development report", "atlas": atlas, "graph": graph, "n_completed": int(len(t)),
           "n_not_completed": int(len(other)), "n_holdout_rows_read": 0, "settings_read": ["xu_dev_cv"],
           "pins": {"release": ns.part_pin("release"), "A2": ns.part_pin("A2")},
           "namespaces": sorted(t["namespace"].unique()) if len(t) else [], "a3_commit": provenance.git_commit(),
           "git_dirty": provenance.git_dirty(), "versions": provenance.versions(), "statistics_config": cfg,
           "input_results_sha256": ns.sha256_json(sorted(t["metrics_sha256"].tolist())) if len(t) else None,
           "outputs_sha256": files, "figures": figs, "release": release.release_manifest()}
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str))
    return {"dir": str(out), "n_completed": len(t), "n_not_completed": len(other), "comparisons": len(cmp),
            "supported": int(cmp["supported"].sum()) if len(cmp) else 0, "figures": figs}


# ------------------------------------------------------------------------------------------------ holdout report
def holdout_report(processed, out_dir=None, restrict=None, make_figures: bool = True) -> dict:
    """Descriptive holdout report. Needs a verifying A3 freeze; shows holdout metrics only (13 participants) in its main
    table and the frozen development AUCs in a separate, labelled table. No test, no selection."""
    from . import freeze, holdout

    freeze.verify(processed, restrict=restrict, check_development=True)
    t, _ = inventory.discover(processed, settings=("xu_holdout",))
    st = holdout.status(processed, restrict)
    out = Path(out_dir or ns.root(processed) / "reports" / f"holdout_{ns.stamp()}")
    if out.exists():
        raise FileExistsError(f"{out} exists; reports are never overwritten")
    out.mkdir(parents=True)
    frozen = {}
    for p in ns.PARTS:
        for r in ns.read_json(ns.record_path(processed, p))["development_cv"]:
            frozen[(p,) + tuple(str(r[k]) for k in inventory.UNIT_KEY)] = r
    t = tidy(t) if len(t) else t
    t.to_csv(out / "holdout_results.csv", index=False)
    st.to_csv(out / "holdout_status.csv", index=False)
    comp = []
    for r in t.to_dict("records"):
        f = frozen.get((r["namespace"],) + tuple(str(r[k]) for k in inventory.UNIT_KEY))
        if f:
            comp.append({"namespace": r["namespace"], "experiment": r["experiment"], "variant": r["variant"],
                         "inverse": r["inverse"], "dev_auc_frozen_record": f["auc"], "holdout_auc": r["auc"],
                         "holdout_auc_ci_low": r["auc_ci_low"], "holdout_auc_ci_high": r["auc_ci_high"],
                         "holdout_minus_dev": r["auc"] - f["auc"], "holdout_n": r["n_participants"]})
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "dev_vs_holdout_descriptive.csv", index=False)
    done = int((st["holdout_state"] == "completed").sum()) if len(st) else 0
    rho = None
    if len(comp) >= 5:
        from scipy.stats import spearmanr

        rho = float(spearmanr(comp["dev_auc_frozen_record"], comp["holdout_auc"]).statistic)
    cols = ["family", "variant", "inverse", "n_participants", "auc", "auc_ci_low", "auc_ci_high", "balanced_accuracy",
            "sensitivity", "specificity", "f1", "mcc", "tn", "fp", "fn", "tp"]
    md = ["# Descriptive holdout results (13 participants)\n",
          f"**{release.HOLDOUT_POLICY}** Not new external validation; no p-values, no model selection. "
          f"{done} of {len(st)} frozen units completed.\n",
          md_table(t, cols) if len(t) else "No holdout result yet.\n",
          "\n## Development CV (frozen record) versus holdout, descriptive\n\nDifferent participants and sample "
          "sizes (52 pooled out-of-fold vs 13 refit); a difference is not a test.\n",
          md_table(comp, list(comp.columns)) if len(comp) else "",
          (f"\nSpearman rank correlation of development and holdout AUC across the {len(comp)} units: {rho:.2f} "
           "(units share participants and are not independent).\n" if rho is not None else "")]
    (out / "holdout_results.md").write_text("\n".join(md))
    if make_figures and len(comp):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.errorbar(comp["dev_auc_frozen_record"], comp["holdout_auc"],
                    yerr=[comp["holdout_auc"] - comp["holdout_auc_ci_low"],
                          comp["holdout_auc_ci_high"] - comp["holdout_auc"]], fmt="o", ms=3, lw=.6, c="#0072B2")
        ax.plot([0.3, 1], [0.3, 1], "k:", lw=.6)
        ax.set_xlabel("development-CV AUC (frozen record)")
        ax.set_ylabel("holdout AUC (13 participants, DeLong 95% CI)")
        ax.set_title("descriptive; holdout inspected earlier", fontsize=7)
        _save(fig, out / "fig8_dev_vs_holdout")
    return {"dir": str(out), "n_holdout": int(len(t)), "completed_units": done, "planned_units": int(len(st))}
