"""Tables and diagnostic figures of the cross-dataset campaign, from saved evaluation outputs only (design section 11).

``reports/<stage>_<UTC time>/`` (never overwritten):
    table1_transfer.csv        per direction x inverse x mode x experiment x variant (target stratum "all"): source-
                               only CV AUC / balanced accuracy, target AUC [DeLong CI] [bootstrap CI], balanced
                               accuracy [CI], sensitivity, specificity [Wilson], F1, MCC, n AD / n HC, confusion
                               matrix, transfer degradation [CI], exploratory / reference flags
    table2_d3_strata.csv       D2 -> D3 on the 52 development and 13 holdout participants separately (descriptive)
    table3_paired.csv          paired DeLong vs the reference baseline (Holm within family; exploratory)
    table4_age.csv             demographics; age-only baseline; age-overlap subset; score-age association
    fig_forest_<dir>_<inv>_<mode>.png   target AUC with 95 % CI per variant (participants), source CV AUC marked
    fig_age_<dir>.png          participant score vs age by diagnosis for the reference baseline (exploratory)
    README.md                  what each file holds, the disclosures, and the synthetic watermark if applicable
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..checkpoint import atomic_write_csv, atomic_write_text
from . import namespace, plan

MAIN = ["direction", "inverse", "mode", "exp", "experiment", "variant", "seed", "exploratory", "reference",
        "n", "n_ad", "n_hc", "source_cv_status", "source_cv_auc", "source_cv_balanced_accuracy", "roc_auc",
        "auc_delong_low", "auc_delong_high", "roc_auc_boot_low", "roc_auc_boot_high", "balanced_accuracy",
        "balanced_accuracy_boot_low", "balanced_accuracy_boot_high", "sensitivity", "sensitivity_wilson_low",
        "sensitivity_wilson_high", "specificity", "specificity_wilson_low", "specificity_wilson_high", "f1", "mcc",
        "tn", "fp", "fn", "tp", "degradation_roc_auc", "degradation_roc_auc_low", "degradation_roc_auc_high",
        "degradation_balanced_accuracy", "degradation_balanced_accuracy_low", "degradation_balanced_accuracy_high",
        "score_type", "threshold"]


def _watermark(ax, synthetic: bool):
    if synthetic:
        ax.text(0.5, 0.5, "SYNTHETIC - NOT A RESULT", transform=ax.transAxes, ha="center", va="center",
                fontsize=22, color="red", alpha=0.25, rotation=25)


def report(processed, c: dict, stage: str = "primary") -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = namespace.root(processed) / "evaluation" / stage
    summary = json.loads((ev / "summary.json").read_text())
    synthetic = bool(summary.get("synthetic"))
    long = pd.read_csv(ev / "metrics_long.csv")
    out = namespace.sub(processed, "reports", f"{stage}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    main = long[long["stratum"] == "all"]
    atomic_write_csv(out / "table1_transfer.csv", main[[x for x in MAIN if x in main]])
    atomic_write_csv(out / "table2_d3_strata.csv", long[long["stratum"] != "all"][[x for x in MAIN + ["stratum"]
                                                                                    if x in long]])
    atomic_write_csv(out / "table3_paired.csv", pd.read_csv(ev / "paired_comparisons.csv")
                     if (ev / "paired_comparisons.csv").stat().st_size > 2 else pd.DataFrame())
    parts = []
    for name in ("demographics", "age_only_baseline", "age_overlap", "age_score_association"):
        f = ev / f"{name}.csv"
        if f.exists() and f.stat().st_size > 2:
            parts.append(pd.read_csv(f).assign(table=name))
    atomic_write_csv(out / "table4_age.csv", pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
    figs = []
    for (d, inv, mode), g in main.groupby(["direction", "inverse", "mode"]):
        g = g.sort_values(["exp", "variant"]).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7, 0.28 * len(g) + 1.5))
        yy = np.arange(len(g))
        colors = np.where(g["reference"], "tab:gray", np.where(g["exploratory"], "tab:orange", "tab:blue"))
        ax.errorbar(g["roc_auc"], yy, xerr=[g["roc_auc"] - g["auc_delong_low"], g["auc_delong_high"] - g["roc_auc"]],
                    fmt="none", ecolor="0.6", lw=1)
        ax.scatter(g["roc_auc"], yy, c=colors, s=18, zorder=3, label="target (zero-shot)")
        if g["source_cv_auc"].notna().any():
            ax.scatter(g["source_cv_auc"], yy, marker="|", c="k", s=60, zorder=4, label="source-only CV")
        ax.axvline(0.5, color="k", lw=0.8, ls=":")
        ax.set_yticks(yy, [f"{e} {v}" for e, v in zip(g["exp"], g["variant"])], fontsize=6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("participant ROC-AUC (95 % DeLong CI)")
        ax.set_title(f"{d} | {inv} | mode {mode} (n AD {int(g['n_ad'].iloc[0])}, n HC {int(g['n_hc'].iloc[0])})",
                     fontsize=9)
        ax.legend(fontsize=6, loc="lower right")
        _watermark(ax, synthetic)
        fig.tight_layout()
        p = out / f"fig_forest_{d}_{inv}_{mode}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append(p.name)
    pred = pd.read_csv(ev / "predictions_participants.csv") if (ev / "predictions_participants.csv").stat().st_size > 2 \
        else pd.DataFrame()
    ref = c["evaluation"]["comparisons"]["reference_baseline"]
    if len(pred) and "age" in pred:
        for d, g in pred.groupby("direction"):
            r = g[(g["variant"] == ref["variant"]) & g["experiment"].str.startswith(ref["experiment"] + "_")
                  & (g["seed"] == g["seed"].min())]
            if r.empty:
                continue
            n = r[["inverse", "mode"]].drop_duplicates().shape[0]
            fig, axes = plt.subplots(1, n, figsize=(3 * n + 1, 3), squeeze=False)
            for ax, ((inv, mode), h) in zip(axes[0], r.groupby(["inverse", "mode"])):
                for lab, col in (("AD", "tab:red"), ("HC", "tab:blue")):
                    q = h[h["label"] == lab]
                    ax.scatter(q["age"], q["mean_score"], s=12, c=col, label=lab)
                ax.axhline(float(h["threshold"].iloc[0]), color="k", lw=0.6, ls=":")
                ax.set_title(f"{inv} | mode {mode}", fontsize=8)
                ax.set_xlabel("age (years)")
                _watermark(ax, synthetic)
            axes[0][0].set_ylabel("participant mean P(AD)")
            axes[0][0].legend(fontsize=6)
            fig.suptitle(f"{d}: reference baseline {ref['experiment']} {ref['variant']} - score vs age "
                         "(exploratory; participants)", fontsize=8)
            fig.tight_layout()
            p = out / f"fig_age_{d}.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            figs.append(p.name)
    atomic_write_text(out / "README.md", readme(summary, figs, synthetic, ref))
    return {"dir": str(out), "figures": figs}


def readme(summary: dict, figs: list[str], synthetic: bool, ref: dict) -> str:
    lines = [f"# Cross-dataset D3 <-> D2 report ({summary['stage']})", ""]
    if synthetic:
        lines += ["**SYNTHETIC DATA: smoke-test output, not a result.**", ""]
    lines += [f"Units evaluated: {summary['n_units_evaluated']} of {summary['n_units_planned']} planned; frozen "
              f"campaign {summary['frozen_campaign_hash']}; code {summary['code']['git_commit']}.", "",
              "Participant-level metrics only (one mean P(AD) per participant, threshold 0.5; SVM margin 0). CIs: "
              "DeLong for AUC, participant bootstrap stratified by diagnosis, Wilson for sensitivity / specificity. "
              f"Paired comparisons: vs {ref['experiment']} / {ref['variant']} on the same target participants, Holm "
              "within direction x inverse x mode; exploratory.", "", "## Disclosures", ""]
    lines += [f"- {d}" for d in summary["disclosures"]]
    lines += ["", "## Files", "", "- table1_transfer.csv, table2_d3_strata.csv, table3_paired.csv, table4_age.csv"]
    lines += [f"- {f}" for f in figs]
    return "\n".join(lines) + "\n"
