"""Subject-level performance figures (families 1-4, 10, 12, 14).

Statistical unit: every plotted value is a **subject-level** metric. One point =
one CV repeat, whose metric is computed from the out-of-fold predictions of all
subjects (``metrics_repeats.csv``). Per-fold metrics are not plotted: a fold holds
only 4-5 test subjects, so its AUC is unstable or undefined. Error bars are the mean
± SD across repeats, or (``uncertainty: bootstrap``) the 95 % subject-bootstrap CI
of the repeat-averaged subject scores stored in ``summary.json``. Repeats share
subjects, so their spread shows split variability, not independent replication.

PLV and |ciPLV| are always separate conditions (colour + marker + fill); they are
never pooled into one value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import stats as st
from .results import read_csv, read_manifest, read_summary, same_splits, select, select_unique
from .style import (CONDITION_STYLE, METRIC_LABEL, SETTING_LABEL, condition_style, figure_size, marker_kwargs)

CHANCE = {"balanced_accuracy": 0.5, "roc_auc": 0.5}
CROSS = ("cross_D1_to_D2", "cross_D2_to_D1")


# ------------------------------------------------------------------ data collection
def collect_metrics(index, entries, settings, metrics, scope="all", with_ci=False) -> pd.DataFrame:
    """Long table: label, experiment, variant, setting, condition, repeat, metric, value [, ci_lo, ci_hi]."""
    rows = []
    for e in entries:
        for _, r in select_unique(index, e, settings).iterrows():
            mr = read_csv(r, "metrics_repeats.csv")
            mr = mr[mr["scope"] == scope]
            summ = read_summary(r) if with_ci else {}
            ci = summ.get("bootstrap_ci_95", {})
            point = summ.get("subject_level_repeat_averaged", {})
            for metric in metrics:
                for _, m in mr.iterrows():
                    rows.append({"label": e["label"], "experiment": r["experiment"], "variant": r["variant"],
                                 "setting": r["setting"], "condition": r["condition"], "repeat": int(m["repeat"]),
                                 "metric": metric, "value": m[metric],
                                 "ci_lo": ci.get(metric, [np.nan, np.nan])[0] if with_ci else np.nan,
                                 "ci_hi": ci.get(metric, [np.nan, np.nan])[1] if with_ci else np.nan,
                                 "ci_point": point.get(metric, np.nan) if with_ci else np.nan,
                                 "path": r["path"]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no saved results matched the figure entries")
    return df


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = df.groupby(keys, sort=False)["value"]
    out = g.agg(mean="mean", sd=lambda v: v.std(ddof=1) if v.notna().sum() > 1 else np.nan,
                n_repeats=lambda v: int(v.notna().sum())).reset_index()
    if "ci_lo" in df:
        ci = df.groupby(keys, sort=False)[["ci_point", "ci_lo", "ci_hi"]].first().reset_index()
        out = out.merge(ci, on=keys, how="left")
    return out


# ------------------------------------------------------------------ drawing core
def strip_panel(ax, df, x_col, x_order, style, hue_col="condition", hue_order=None, uncertainty="repeats",
                rng=None, ylim=None, chance=None):
    """Repeat points (jittered) + mean marker with error bar, per x and hue. Returns x positions."""
    rng = rng or np.random.default_rng(0)
    hue_order = hue_order or [h for h in CONDITION_STYLE if h in set(df[hue_col])] or sorted(set(df[hue_col]))
    hue_order = [h for h in hue_order if h in set(df[hue_col])]
    width = 0.7 / max(len(hue_order), 1)
    pos = {}
    for xi, x in enumerate(x_order):
        for hi, h in enumerate(hue_order):
            d = df[(df[x_col] == x) & (df[hue_col] == h)]
            if d.empty:
                continue
            cx = xi + (hi - (len(hue_order) - 1) / 2) * width
            pos[(x, h)] = cx
            spec = condition_style(h)
            vals = d["value"].to_numpy(float)
            jit = rng.uniform(-style["jitter"], style["jitter"], len(vals)) * width
            ax.scatter(cx + jit, vals, s=style["point_size"] ** 2, alpha=style["point_alpha"], linewidths=0.4,
                       marker=spec["marker"], facecolors=spec["color"] if spec["fill"] else "none",
                       edgecolors=spec["color"], zorder=2)
            mean = np.nanmean(vals)
            if uncertainty == "bootstrap" and "ci_lo" in d and np.isfinite(d["ci_lo"].iloc[0]):
                # centre = metric of repeat-averaged subject scores, i.e. the quantity the CI describes
                mean = d["ci_point"].iloc[0]
                yerr = [[max(mean - d["ci_lo"].iloc[0], 0)], [max(d["ci_hi"].iloc[0] - mean, 0)]]
            elif np.isfinite(vals).sum() > 1:
                sd = np.nanstd(vals, ddof=1)
                yerr = [[sd], [sd]]
            else:
                yerr = None
            ax.errorbar(cx, mean, yerr=yerr, capsize=style["errorbar_capsize"], zorder=3,
                        elinewidth=style["line_width"], **marker_kwargs(spec, style["marker_size"] * 1.4))
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order, rotation=35, ha="right")
    ax.set_xlim(-0.6, len(x_order) - 0.4)
    if ylim:
        ax.set_ylim(*ylim)
    if chance is not None:
        ax.axhline(chance, color="0.5", lw=0.6, ls=":", zorder=1)
    return pos, hue_order


def condition_legend(fig, hue_order, style):
    handles = [plt.Line2D([], [], linestyle="none", **marker_kwargs(condition_style(h), style["marker_size"] * 1.4),
                          label=condition_style(h)["label"]) for h in hue_order]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))


def annotate_from_stats(ax, positions, stats, prefix, ymax):
    """Brackets with stars from a valid statistics file. Comparison ids:
    ``<prefix><labelA>[<condA>]~<labelB>[<condB>]``. Returns the comparisons drawn."""
    if stats is None:
        return []
    drawn, y = [], ymax
    for _, row in stats[stats["comparison"].str.startswith(prefix)].iterrows():
        a, b = row["comparison"][len(prefix):].split("~")
        pa = next((p for (x, h), p in positions.items() if f"{x}[{h}]" == a), None)
        pb = next((p for (x, h), p in positions.items() if f"{x}[{h}]" == b), None)
        if pa is None or pb is None:
            continue
        y += 0.04
        ax.plot([pa, pa, pb, pb], [y - 0.01, y, y, y - 0.01], color="k", lw=0.5)
        ax.text((pa + pb) / 2, y, st.stars(row["p_corrected"]) or "n.s.", ha="center", va="bottom", fontsize=6)
        drawn.append(row["comparison"])
    return drawn


def _meta(kind, unit, notes, extra=None):
    return {"kind": kind, "statistical_unit": unit, "notes": notes, **(extra or {})}


UNIT_REPEATS = ("subject-level metric per CV repeat (all subjects' out-of-fold predictions); "
                "repeats are not independent samples")


# ------------------------------------------------------------------ family 1
def fig_performance(index, spec, style, stats=None):
    settings, metrics = spec["settings"], spec["metrics"]
    df = collect_metrics(index, spec["entries"], settings, metrics, spec.get("scope", "all"),
                         with_ci=spec.get("uncertainty") == "bootstrap")
    labels = [e["label"] for e in spec["entries"] if e["label"] in set(df["label"])]
    fig, axes = plt.subplots(len(metrics), len(settings), squeeze=False, sharey="row",
                             figsize=figure_size(style, spec.get("width", "double"),
                                                 height_mm=spec.get("height_mm", 38 * len(metrics) + 20)))
    drawn, hue_order = [], []
    for i, metric in enumerate(metrics):
        for j, s in enumerate(settings):
            ax = axes[i, j]
            d = df[(df["metric"] == metric) & (df["setting"] == s)]
            pos, hue_order = strip_panel(ax, d, "label", labels, style, uncertainty=spec.get("uncertainty", "repeats"),
                                         ylim=(0, 1.05), chance=CHANCE.get(metric))
            drawn += annotate_from_stats(ax, pos, stats, f"{s}:{metric}:", 1.0)
            if i == 0:
                ax.set_title(SETTING_LABEL.get(s, s))
            if j == 0:
                ax.set_ylabel(METRIC_LABEL.get(metric, metric))
            if i < len(metrics) - 1:
                ax.set_xticklabels([])
    condition_legend(fig, sorted(set(df["condition"]), key=list(CONDITION_STYLE).index), style)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    data = summarize(df, ["label", "condition", "setting", "metric"]).merge(
        df[["label", "condition", "setting", "metric", "repeat", "value"]], on=["label", "condition", "setting", "metric"])
    meta = _meta("inferential" if drawn else "descriptive", UNIT_REPEATS,
                 "points = CV repeats; " +
                 ("marker = metric of repeat-averaged subject scores; bars = its 95% subject-bootstrap CI"
                  if spec.get("uncertainty") == "bootstrap" else "marker = mean over repeats; bars = SD across repeats"),
                 {"significance_from_stats_file": drawn})
    return fig, data, meta, sorted(set(df["path"]))


# ------------------------------------------------------------------ family 2
def _pair_label(variant: str) -> str:
    return "__".join(t for t in variant.split("__") if t not in ("plv", "ciplv")) or "default"


def matched_condition_pairs(index, experiments, settings):
    """(plv_row, ciplv_row) with identical configs except graph.connectivity and identical subject folds."""
    pairs, refused = [], []
    for exp in experiments:
        sel = select(index, exp, settings)
        for _, a in sel[sel["condition"] == "plv"].iterrows():
            for _, b in sel[(sel["condition"] == "ciplv") & (sel["setting"] == a["setting"])].iterrows():
                ca = {k: v for k, v in a["config"].items() if k not in ("variant", "name")}
                cb = {k: v for k, v in b["config"].items() if k not in ("variant", "name")}
                cb = {**cb, "graph": {**cb["graph"], "connectivity": "plv"}}
                if ca != cb:
                    continue
                if same_splits(a, b):
                    pairs.append((a, b))
                else:
                    refused.append((a["variant"], b["variant"], a["setting"]))
    return pairs, refused


def fig_paired_conditions(index, spec, style, stats=None):
    metric, settings = spec["metric"], spec["settings"]
    pairs, refused = matched_condition_pairs(index, spec["experiments"], settings)
    if not pairs:
        raise ValueError("no PLV/|ciPLV| pairs with identical configuration and identical subject folds")
    rows = []
    for a, b in pairs:
        ma, mb = (read_csv(r, "metrics_repeats.csv").query("scope == 'all'").set_index("repeat")[metric] for r in (a, b))
        for rep in ma.index.intersection(mb.index):
            rows.append({"experiment": a["experiment"], "pair": f"{a['experiment']}:{_pair_label(a['variant'])}",
                         "setting": a["setting"], "repeat": int(rep), "metric": metric, "plv": ma[rep],
                         "ciplv": mb[rep], "diff_ciplv_minus_plv": mb[rep] - ma[rep],
                         "path_plv": a["path"], "path_ciplv": b["path"]})
    df = pd.DataFrame(rows)
    pair_order = list(dict.fromkeys(df["pair"]))
    fig, axes = plt.subplots(1, len(settings), squeeze=False, sharey=True,
                             figsize=figure_size(style, spec.get("width", "double"),
                                                 height_mm=spec.get("height_mm", 70)))
    sp, sc = CONDITION_STYLE["plv"], CONDITION_STYLE["ciplv"]
    for j, s in enumerate(settings):
        ax = axes[0, j]
        for xi, p in enumerate(pair_order):
            d = df[(df["pair"] == p) & (df["setting"] == s)]
            for _, r in d.iterrows():
                ax.plot([xi - 0.15, xi + 0.15], [r["plv"], r["ciplv"]], color="0.6", lw=0.4, zorder=1)
            ax.plot(np.full(len(d), xi - 0.15), d["plv"], linestyle="none", alpha=0.8, **marker_kwargs(sp))
            ax.plot(np.full(len(d), xi + 0.15), d["ciplv"], linestyle="none", alpha=0.8, **marker_kwargs(sc))
        ax.set_xticks(range(len(pair_order)))
        ax.set_xticklabels(pair_order, rotation=40, ha="right")
        ax.set_title(SETTING_LABEL.get(s, s))
        ax.axhline(CHANCE.get(metric, np.nan), color="0.5", lw=0.6, ls=":")
        if j == 0:
            ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    condition_legend(fig, ["plv", "ciplv"], style)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    meta = _meta("descriptive", "paired subject-level metric per CV repeat (same model, band, subjects and folds)",
                 "lines join PLV and |ciPLV| of the same repeat; only pairs with identical configuration and "
                 "identical saved subject folds are shown", {"refused_unmatched_pairs": refused})
    return fig, df, meta, sorted(set(df["path_plv"]) | set(df["path_ciplv"]))


# ------------------------------------------------------------------ family 3
def fig_band_comparison(index, spec, style, stats=None):
    metric, settings, bands = spec["metric"], spec["settings"], spec.get("bands", ["theta", "alpha1", "alpha2", "beta"])
    rows = []
    for panel in spec["panels"]:
        sel = select(index, panel["experiment"], settings, panel.get("where"))
        sel = sel[sel["band_mode"] == "single"]
        for _, r in sel.iterrows():
            mr = read_csv(r, "metrics_repeats.csv").query("scope == 'all'")
            for _, m in mr.iterrows():
                rows.append({"panel": panel["label"], "band": r["bands"], "setting": r["setting"],
                             "condition": r["condition"], "repeat": int(m["repeat"]), "metric": metric,
                             "value": m[metric], "path": r["path"]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no single-band results matched")
    dup = df.duplicated(["panel", "band", "setting", "condition", "repeat"])
    if dup.any():
        raise ValueError("several variants per panel/band/condition; refine panel 'where' (e.g. model.encoder)")
    panels = [p["label"] for p in spec["panels"]]
    fig, axes = plt.subplots(len(panels), len(settings), squeeze=False, sharey=True,
                             figsize=figure_size(style, spec.get("width", "double"),
                                                 height_mm=spec.get("height_mm", 42 * len(panels) + 15)))
    for i, p in enumerate(panels):
        for j, s in enumerate(settings):
            ax = axes[i, j]
            strip_panel(ax, df[(df["panel"] == p) & (df["setting"] == s)], "band", bands, style, ylim=(0, 1.05),
                        chance=CHANCE.get(metric))
            if i == 0:
                ax.set_title(SETTING_LABEL.get(s, s))
            if j == 0:
                ax.set_ylabel(f"{p}\n{METRIC_LABEL.get(metric, metric)}")
    condition_legend(fig, [c for c in CONDITION_STYLE if c in set(df["condition"])], style)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    data = summarize(df, ["panel", "band", "setting", "condition"]).merge(
        df[["panel", "band", "setting", "condition", "repeat", "value"]], on=["panel", "band", "setting", "condition"])
    return fig, data, _meta("descriptive", UNIT_REPEATS, "points = CV repeats; bars = SD across repeats"), \
        sorted(set(df["path"]))


# ------------------------------------------------------------------ family 4
def _ablation_name(variant: str) -> str:
    toks = [t for t in variant.split("__") if t not in ("plv", "ciplv")]
    return toks[-1] if toks else "full"


def fig_ablation(index, spec, style, stats=None):
    metric, settings, order = spec["metric"], spec["settings"], spec["order"]
    sel = select(index, spec["experiment"], settings)
    rows, refused = [], []
    for _, r in sel.iterrows():
        name = _ablation_name(r["variant"])
        if name not in order:
            continue
        full = sel[(sel["setting"] == r["setting"]) & (sel["condition"] == r["condition"])
                   & (sel["variant"].map(_ablation_name) == "full")]
        mr = read_csv(r, "metrics_repeats.csv").query("scope == 'all'").set_index("repeat")[metric]
        mf = None
        if not full.empty:
            if same_splits(r, full.iloc[0]):
                mf = read_csv(full.iloc[0], "metrics_repeats.csv").query("scope == 'all'").set_index("repeat")[metric]
            else:
                refused.append((r["variant"], r["setting"]))
        for rep, val in mr.items():
            rows.append({"ablation": name, "setting": r["setting"], "condition": r["condition"], "repeat": int(rep),
                         "metric": metric, "value": val,
                         "delta_vs_full": (val - mf[rep]) if mf is not None and rep in mf.index else np.nan,
                         "path": r["path"]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no ablation results matched")
    order = [o for o in order if o in set(df["ablation"])]
    conds = [c for c in ("plv", "ciplv") if c in set(df["condition"])]
    fig, axes = plt.subplots(2 * len(conds), len(settings), squeeze=False, sharey="row",
                             figsize=figure_size(style, spec.get("width", "double"),
                                                 height_mm=spec.get("height_mm", 40 * 2 * len(conds) + 15)))
    for ci, c in enumerate(conds):
        for j, s in enumerate(settings):
            d = df[(df["condition"] == c) & (df["setting"] == s)]
            ax = axes[2 * ci, j]
            strip_panel(ax, d, "ablation", order, style, ylim=(0, 1.05), chance=CHANCE.get(metric))
            ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"{CONDITION_STYLE[c]['label']}\n{METRIC_LABEL.get(metric, metric)}")
            if ci == 0:
                ax.set_title(SETTING_LABEL.get(s, s))
            ax2 = axes[2 * ci + 1, j]
            strip_panel(ax2, d.assign(value=d["delta_vs_full"]), "ablation", order, style, chance=0.0)
            if j == 0:
                ax2.set_ylabel(f"Δ vs full\n(paired repeats)")
    fig.tight_layout()
    data = df.drop(columns=["path"])
    meta = _meta("descriptive", UNIT_REPEATS, "Δ vs full is computed per repeat between results with identical "
                 "saved subject folds", {"refused_unmatched": refused})
    return fig, data, meta, sorted(set(df["path"]))


# ------------------------------------------------------------------ family 10
def fig_generalization(index, spec, style, stats=None):
    settings = spec.get("settings", ["within_D1", "within_D2", "pooled", *CROSS])
    metrics = spec["metrics"]
    df = collect_metrics(index, spec["entries"], settings, metrics, with_ci=True)
    labels = [e["label"] for e in spec["entries"] if e["label"] in set(df["label"])]
    x_order = [SETTING_LABEL[s] for s in settings]
    df["setting_label"] = df["setting"].map(SETTING_LABEL)
    fig, axes = plt.subplots(len(metrics), len(labels), squeeze=False, sharey="row",
                             figsize=figure_size(style, spec.get("width", "double"),
                                                 height_mm=spec.get("height_mm", 45 * len(metrics) + 15)))
    for i, metric in enumerate(metrics):
        for j, lab in enumerate(labels):
            ax = axes[i, j]
            strip_panel(ax, df[(df["metric"] == metric) & (df["label"] == lab)], "setting_label", x_order, style,
                        uncertainty="bootstrap", ylim=(0, 1.05), chance=CHANCE.get(metric))
            n_within = sum(s not in CROSS for s in settings)
            if n_within < len(settings):
                ax.axvline(n_within - 0.5, color="0.3", lw=0.6, ls="--")
                ax.axvspan(n_within - 0.5, len(settings) - 0.4, color="0.92", zorder=0)
            if i == 0:
                ax.set_title(lab)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL.get(metric, metric))
            if i < len(metrics) - 1:
                ax.set_xticklabels([])
    condition_legend(fig, [c for c in CONDITION_STYLE if c in set(df["condition"])], style)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    data = summarize(df, ["label", "condition", "setting", "metric"])
    meta = _meta("descriptive", "subject-level; marker = metric of repeat-averaged subject scores, bars = its 95% "
                 "subject-bootstrap CI; points = CV repeats (cross-dataset: model seeds on one fixed split)",
                 "shaded region = cross-dataset transfer (train on one dataset, test on all subjects of the other)")
    return fig, data, meta, sorted(set(df["path"]))


# ------------------------------------------------------------------ family 12
AGE_TYPES = ["Age only", "EEG only", "EEG + age"]


def fig_age_sensitivity(index, spec, style, stats=None):
    metric, settings = spec["metric"], spec["settings"]
    rows = []
    for g in spec["groups"]:
        for mtype, exp, where in (("Age only", spec.get("age_experiment", "baseline_age"), {}),
                                  ("EEG only", g["experiment"], g.get("where")),
                                  ("EEG + age", g["experiment"] + "_plus_age", g.get("where"))):
            for _, r in select(index, exp, settings, where).iterrows():
                for _, m in read_csv(r, "metrics_repeats.csv").query("scope == 'all'").iterrows():
                    rows.append({"group": g["label"], "model_type": mtype, "setting": r["setting"],
                                 "condition": r["condition"], "repeat": int(m["repeat"]), "metric": metric,
                                 "value": m[metric], "path": r["path"]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no age-sensitivity results matched")
    groups = [g["label"] for g in spec["groups"]]
    fig, axes = plt.subplots(len(groups), len(settings), squeeze=False, sharey=True,
                             figsize=figure_size(style, spec.get("width", "double"),
                                                 height_mm=spec.get("height_mm", 42 * len(groups) + 15)))
    for i, g in enumerate(groups):
        for j, s in enumerate(settings):
            ax = axes[i, j]
            strip_panel(ax, df[(df["group"] == g) & (df["setting"] == s)], "model_type", AGE_TYPES, style,
                        ylim=(0, 1.05), chance=CHANCE.get(metric))
            if i == 0:
                ax.set_title(SETTING_LABEL.get(s, s))
            if j == 0:
                ax.set_ylabel(f"{g}\n{METRIC_LABEL.get(metric, metric)}")
    condition_legend(fig, [c for c in CONDITION_STYLE if c in set(df["condition"])], style)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    data = summarize(df, ["group", "model_type", "setting", "condition"])
    return fig, data, _meta("descriptive", UNIT_REPEATS, "age-only, EEG-only and EEG + age on identical splits"), \
        sorted(set(df["path"]))


# ------------------------------------------------------------------ family 14
class IncomparableTiming(ValueError):
    pass


def fig_complexity(index, spec, style, stats=None):
    """Performance vs parameter count (always comparable) or fit time (only within ONE environment).

    Fit times are compared only if every result carries the same ``compute_environment``
    fingerprint (CPU/GPU, device, threads, platform, library versions); otherwise refused.
    """
    metric, setting, quantity = spec["metric"], spec["setting"], spec.get("quantity", "n_parameters")
    rows, envs = [], {}
    for e in spec["entries"]:
        for _, r in select_unique(index, e, [setting]).iterrows():
            hp = read_csv(r, "hparams.csv")
            if quantity not in hp:
                raise ValueError(f"{r['path']}/hparams.csv has no {quantity!r} (older run)")
            mr = read_csv(r, "metrics_repeats.csv").query("scope == 'all'")[metric]
            env = read_manifest(r).get("compute_environment")
            if quantity == "fit_seconds":
                if not env:
                    raise IncomparableTiming(f"{r['path']}: no compute_environment saved; fit times not comparable")
                envs.setdefault(env["fingerprint"], env)
            rows.append({"label": e["label"], "condition": r["condition"], quantity: hp[quantity].mean(),
                         "environment": env.get("fingerprint") if env else None,
                         f"{quantity}_sd": hp[quantity].std(ddof=1) if len(hp) > 1 else np.nan,
                         "mean": mr.mean(), "sd": mr.std(ddof=1) if len(mr) > 1 else np.nan,
                         "n_repeats": len(mr), "path": r["path"]})
    df = pd.DataFrame(rows)
    if len(envs) > 1:
        raise IncomparableTiming(f"fit times come from {len(envs)} different computational environments "
                                 f"{sorted(envs)}; rerun in one environment or plot n_parameters")
    fig, ax = plt.subplots(figsize=figure_size(style, spec.get("width", "single")))
    for _, r in df.iterrows():
        ax.errorbar(r[quantity], r["mean"], yerr=r["sd"] if np.isfinite(r["sd"]) else None,
                    capsize=style["errorbar_capsize"], linestyle="none", **marker_kwargs(condition_style(r["condition"])))
        ax.annotate(r["label"], (r[quantity], r["mean"]), xytext=(3, 3), textcoords="offset points", fontsize=5)
    ax.set_xscale("log")
    ax.set_xlabel({"n_parameters": "Trainable parameters", "fit_seconds": "Fit time per split (s)"}.get(quantity, quantity))
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    fig.tight_layout()
    if quantity == "fit_seconds":
        ax.set_title(f"one environment: {next(iter(envs.values()))['device']}, "
                     f"{next(iter(envs.values()))['cpu_model'][:40]}", fontsize=5)
    return fig, df.drop(columns=["path"]), _meta(
        "descriptive", UNIT_REPEATS, f"x = mean {quantity} over splits; y = mean ± SD across repeats",
        {"compute_environment": next(iter(envs.values())) if envs else None}), sorted(set(df["path"]))
