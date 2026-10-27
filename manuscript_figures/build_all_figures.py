#!/usr/bin/env python3
"""Manuscript figure builder for the D3 source-space EEG-GNN study (reads saved outputs only).

NOTHING here trains, infers or evaluates a model. Every plotted number is read from a saved file:
  development_results_all.csv, comparisons.csv, selection_check.json, resolution_by_atlas.csv (or recommendation.json),
  the per-unit predictions_participants.csv files (one out-of-fold score per participant), the single-model holdout
  predictions, and the committed atlas tables (src/adgraph/source/atlas_tables.json).
Descriptive statistics recomputed from saved participant predictions (AUC, DeLong CI, paired DeLong) are labelled as
"recomputed from saved predictions" and are cross-checked against the saved table (the run aborts on a mismatch).

Scope rule of the manuscript: Exp0 / Exp1 (sensor / source classical baselines) are excluded from every figure.
Output rule: figures are written ONLY to --out (default ./generated); the script refuses any output path inside a
FINAL_v1 results tree or an amendment namespace and never opens a result file for writing.

Usage (Colab, Drive mounted):  python build_all_figures.py --processed /content/drive/MyDrive/AD-LEB/processed
       (local test):          python build_all_figures.py --results-csv R.csv --comparisons-csv C.csv --pred-dir P ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CLASSICAL = ("exp0_sensor_psd_de", "exp1_source_psd_de")
OKABE = {"lcmv": "#0072B2", "wmne": "#D55E00", "grey": "#6b6b6b", "green": "#009E73", "purple": "#CC79A7",
         "yellow": "#E69F00", "sky": "#56B4E9", "black": "#000000"}
LOBE_COLOR = {"frontal": "#0072B2", "parietal": "#009E73", "temporal": "#D55E00", "occipital": "#E69F00",
              "cingulate": "#CC79A7", "insula": "#56B4E9"}
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
FAMILIES = ("exp2_conn_minimal", "exp3_single_band", "exp4_band_mean", "exp5_multibranch", "exp6_psd_strength",
            "exp7_earm_gnn", "exp8a_xu_dual_gcn_surface")
FAM_LABEL = dict(zip(FAMILIES, ("Exp2", "Exp3", "Exp4", "Exp5", "Exp6", "Exp7", "Exp8a")))
REQUIRED_COLS = ["experiment", "variant", "inverse", "mode", "atlas", "graph", "namespace", "n_participants", "auc",
                 "auc_ci_low", "auc_ci_high", "balanced_accuracy", "sensitivity", "specificity", "tn", "fp", "fn",
                 "tp", "mean_fold_auc", "sd_fold_auc", "rel_dir"]
CMP_COLS = ["family", "experiment_a", "variant_a", "experiment_b", "variant_b", "inverse_a", "mode_a", "auc_diff",
            "boot_low", "boot_high", "p_holm", "supported"]


# ------------------------------------------------------------------------------------------------ models of interest
@dataclass(frozen=True)
class Unit:
    """Complete identity of one saved development-CV unit (never matched by variant name alone)."""
    key: str
    label: str
    experiment: str
    variant: str
    inverse: str
    mode: str
    atlas: str
    graph: str
    fallback_file: str          # name inside --pred-dir when the Drive tree is not mounted


PRIMARY = Unit("exp8a_primary", "Exp8a plv__psd (LCMV, lobes12)", "exp8a_xu_dual_gcn_surface", "plv__psd", "lcmv", "B",
               "lobes12", "dense", "exp8a_lcmv_plv_psd_B.csv")
EXP5_R18 = Unit("exp5_r18", "Exp5 separate/mean (LCMV, r18)", "exp5_multibranch", "plv__separate__mean", "lcmv", "B",
                "r18", "dense", "exp5_lcmv_r18_plv_sep_mean_B.csv")
EXP4_REF = Unit("exp4_ref", "Exp4 band-mean GCN reference (LCMV, lobes12)", "exp4_band_mean", "plv__psd", "lcmv", "B",
                "lobes12", "dense", "exp4_lcmv_lobes12_plv_psd_B.csv")
HOLDOUT_UNIT = Unit("holdout", "Exp8a plv__psd_de_strength__abl_anatomical_both (wMNE, lobes12)",
                    "exp8a_xu_dual_gcn_surface", "plv__psd_de_strength__abl_anatomical_both", "wmne", "B", "lobes12",
                    "dense", "holdout_wmne_anatomical_both.csv")


class MissingInput(RuntimeError):
    """A required saved file / column is absent. Raised instead of substituting any value."""


# ------------------------------------------------------------------------------------------------ configuration / IO
@dataclass
class Config:
    out: Path
    processed: Path | None = None
    results_csv: Path | None = None
    comparisons_csv: Path | None = None
    selection_json: Path | None = None
    resolution_csv: Path | None = None
    pred_dir: Path | None = None
    holdout_pred: Path | None = None
    atlas_tables: Path = REPO / "src" / "adgraph" / "source" / "atlas_tables.json"
    existing_fig_dir: Path | None = None
    dpi: int = 600
    inputs: dict = field(default_factory=dict)         # name -> path actually read (for the manifest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard_output(out: Path) -> None:
    parts = {p.lower() for p in out.resolve().parts}
    if "final_v1" in parts or "amendments" in parts or "results" in parts and "processed" in parts:
        raise PermissionError(f"refusing to write inside a frozen results tree: {out}")


def find_one(root: Path | None, name: str) -> Path | None:
    if root is None or not root.exists():
        return None
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def resolve(cfg: Config, attr: str, name: str, *, required: bool = True) -> Path | None:
    p = getattr(cfg, attr)
    if p is None:
        p = find_one(cfg.processed / "FINAL_v1" if cfg.processed else None, name)
    if p is None or not Path(p).exists():
        if required:
            raise MissingInput(f"{name}: not found (pass --{attr.replace('_', '-')} or mount Drive and pass --processed)")
        return None
    cfg.inputs[name] = Path(p)
    return Path(p)


def load_results(cfg: Config) -> pd.DataFrame:
    p = resolve(cfg, "results_csv", "development_results_all.csv")
    d = pd.read_csv(p)
    miss = [c for c in REQUIRED_COLS if c not in d.columns]
    if miss:
        raise MissingInput(f"development_results_all.csv lacks columns {miss}")
    d = d.copy()
    d["accuracy"] = (d.tn + d.tp) / (d.tn + d.fp + d.fn + d.tp)
    d["scope"] = np.where(d.experiment.isin(CLASSICAL), "classical (excluded)", "GNN")
    return d


def validate_inventory(d: pd.DataFrame) -> dict:
    g = d[d.scope == "GNN"]
    info = {"n_all": len(d), "n_gnn": len(g), "n_classical_excluded": int((d.scope != "GNN").sum()),
            "participants": sorted(g.n_participants.unique().tolist()),
            "families": g.experiment.value_counts().to_dict()}
    if info["participants"] != [52]:
        raise ValueError(f"expected n_participants == 52 for every GNN row, got {info['participants']}")
    return info


def load_comparisons(cfg: Config) -> pd.DataFrame:
    p = resolve(cfg, "comparisons_csv", "comparisons.csv")
    c = pd.read_csv(p)
    miss = [x for x in CMP_COLS if x not in c.columns]
    if miss:
        raise MissingInput(f"comparisons.csv lacks columns {miss}")
    return c[~c.experiment_a.isin(CLASSICAL) & ~c.experiment_b.isin(CLASSICAL)].copy()


def pick(d: pd.DataFrame, u: Unit) -> pd.Series:
    m = d[(d.experiment == u.experiment) & (d.variant == u.variant) & (d.inverse == u.inverse) & (d["mode"] == u.mode)
          & (d.atlas == u.atlas) & (d.graph == u.graph)]
    if len(m) != 1:
        raise ValueError(f"unit {u.key}: expected exactly one saved result, found {len(m)}")
    return m.iloc[0]


# ------------------------------------------------------------------------------------------- predictions + statistics
def load_predictions(cfg: Config, d: pd.DataFrame, u: Unit, *, holdout: bool = False) -> pd.DataFrame:
    """Saved participant-level out-of-fold (or holdout) predictions of a unit; validated against the saved table."""
    row = pick(d, u)
    path = None
    if holdout and cfg.holdout_pred and cfg.holdout_pred.exists():
        path = cfg.holdout_pred
    elif cfg.processed and not holdout:
        cand = cfg.processed / str(row.rel_dir) / "predictions_participants.csv"
        path = cand if cand.exists() else None
    if path is None and cfg.pred_dir and (cfg.pred_dir / u.fallback_file).exists():
        path = cfg.pred_dir / u.fallback_file
    if path is None:
        raise MissingInput(f"predictions for {u.key}: expected {row.rel_dir}/predictions_participants.csv "
                           f"or {u.fallback_file} in --pred-dir")
    cfg.inputs[f"pred:{u.key}"] = path
    p = pd.read_csv(path)
    need = {"key", "y", "score"}
    if not need <= set(p.columns):
        raise MissingInput(f"{path}: needs columns {sorted(need)}")
    if p.key.duplicated().any():
        raise ValueError(f"{path}: more than one prediction per participant (windows must be aggregated)")
    if "repeat" in p.columns and p.repeat.nunique() != 1:
        raise ValueError(f"{path}: several CV repeats; one prediction per participant is required")
    if holdout:
        if len(p) != 13 or int(p.y.sum()) != 7:
            raise ValueError(f"{path}: holdout must be 13 participants (7 AD / 6 HC)")
        return p
    if len(p) != 52 or int(p.y.sum()) != 29:
        raise ValueError(f"{path}: expected 52 participants (29 AD / 23 HC), got {len(p)} / {int(p.y.sum())}")
    auc = roc_auc_score(p.y, p.score)
    if abs(auc - float(row.auc)) > 1e-6:
        raise ValueError(f"{path}: recomputed AUC {auc:.6f} != saved {float(row.auc):.6f}")
    pred = (p.score >= 0.5).astype(int)
    tp, tn = int(((pred == 1) & (p.y == 1)).sum()), int(((pred == 0) & (p.y == 0)).sum())
    if (tp, tn) != (int(row.tp), int(row.tn)):
        raise ValueError(f"{path}: confusion at threshold 0.5 does not match the saved table")
    return p


def _placements(y: np.ndarray, s: np.ndarray):
    pos, neg = s[y == 1], s[y == 0]
    psi = lambda a, b: (a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])
    m = psi(pos, neg)
    return m.mean(1), m.mean(0)


def delong_ci(y, s, alpha=0.05):
    v10, v01 = _placements(np.asarray(y), np.asarray(s, float))
    auc = v10.mean()
    var = v10.var(ddof=1) / len(v10) + v01.var(ddof=1) / len(v01)
    z = norm.ppf(1 - alpha / 2)
    return auc, max(0.0, auc - z * np.sqrt(var)), min(1.0, auc + z * np.sqrt(var))


def delong_paired(y, s1, s2):
    y = np.asarray(y)
    a1, b1 = _placements(y, np.asarray(s1, float))
    a2, b2 = _placements(y, np.asarray(s2, float))
    s10, s01 = np.cov(np.vstack([a1, a2])), np.cov(np.vstack([b1, b2]))
    S = s10 / len(a1) + s01 / len(b1)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    diff = a1.mean() - a2.mean()
    return diff, float(2 * norm.sf(abs(diff) / np.sqrt(var)))


def bootstrap_diff(y, s1, s2, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, s1, s2 = np.asarray(y), np.asarray(s1), np.asarray(s2)
    out = []
    while len(out) < n:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    return np.percentile(out, [2.5, 97.5])


# ----------------------------------------------------------------------------------------------------------- plotting
def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
                         "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 150})


def save(fig, cfg: Config, name: str, manifest: list, caption_key: str, inputs: list[str], note: str = "") -> None:
    cfg.out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ("pdf", "svg", "png"):
        f = cfg.out / f"{name}.{ext}"
        fig.savefig(f, dpi=cfg.dpi if ext == "png" else None, bbox_inches="tight", facecolor="white")
        files.append(f.name)
    plt.close(fig)
    manifest.append({"figure": name, "files": ";".join(files), "caption_key": caption_key,
                     "inputs": ";".join(inputs), "note": note})


def panel(ax, letter):
    ax.text(-0.10, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def box(ax, x, y, w, h, title, body="", fc="#F2F2F2", ec="#444444", fs=5.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h - 0.025, title, ha="center", va="top", fontsize=fs + 0.6, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + (h - 0.07) / 2, body, ha="center", va="center", fontsize=fs, linespacing=1.35)


def arrow(ax, p0, p1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#333333",
                                 connectionstyle=f"arc3,rad={rad}"))


# --------------------------------------------------------------------------------------------------- FIGURE 1 pipeline
def fig_pipeline(cfg, d, man):
    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    W, H = 0.222, 0.36
    xs = [0.005, 0.26, 0.515, 0.77]
    top = [("1  Resting-state EEG", "OpenNeuro ds004504\n19 electrodes, 500 Hz\neyes closed\nAD n = 36, HC n = 29\nprovider: 0.5-45 Hz, ASR, ICA"),
           ("2  Segmentation", "split at boundary events\n250 Hz, average reference\n10 s windows, 1 s step\n5 min selected (mode B)\nmode A: first 10 windows"),
           ("3  Source reconstruction", "fsaverage ico-5, 3-layer BEM\nLCMV (per participant)\nor depth-weighted wMNE\n68 x 19 ROI operator W"),
           ("4  Atlas aggregation", "dk68 -> r18 -> lobes12\nunions of DK labels\nlabel-blind resolution rule\nrecorded atlas: lobes12")]
    bot = [("5  Features per window", "5 bands (delta ... gamma)\nPLV (or |ciPLV|) per band\nPSD (log, relative),\nDE, node strength"),
           ("6  Weighted graphs", "one graph per band\nA_func,b: 12 x 12\nX_b: 12 x F\nfixed A_anat (contiguity)"),
           ("7  Graph network", "Exp8a: dual-layer GCN\nExp5: multiband GCN\n60 epochs, Adam,\nno early stopping"),
           ("8  Participant decision", "window probability\nmean over windows\nthreshold 0.5 -> AD / HC\n5-fold participant CV, n = 52")]
    for i, (t, b) in enumerate(top):
        box(ax, xs[i], 0.58, W, H, t, b, fc="#FDF0E6" if i == 2 else "#EAF2FA")
    for i, (t, b) in enumerate(bot):
        box(ax, xs[3 - i], 0.14, W, H, t, b, fc="#FDF0E6" if i == 2 else "#EAF6F0")
    for i in range(3):
        arrow(ax, (xs[i] + W + 0.004, 0.76), (xs[i + 1] - 0.004, 0.76))
    arrow(ax, (xs[3] + W / 2, 0.58), (xs[3] + W / 2, 0.51))
    for i in range(3):
        arrow(ax, (xs[3 - i] - 0.004, 0.32), (xs[2 - i] + W + 0.004, 0.32))
    ax.text(0.5, 0.03, "LCMV and wMNE differ only in step 3; all downstream code is identical. Classical baselines are outside the scope of this study.",
            ha="center", fontsize=5.8, style="italic", color="#444")
    save(fig, cfg, "fig01_pipeline", man, "fig01", [])


# ----------------------------------------------------------------------------------------------- FIGURE 2 architecture
def fig_architecture(cfg, d, man):
    fig, ax = plt.subplots(figsize=(7.16, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.985, "Per band b = 1..5 (delta, theta, alpha, beta, gamma); N(A) = D^(-1/2) (A + I) D^(-1/2)", ha="center", va="top", fontsize=6.3, style="italic")
    box(ax, 0.01, 0.62, 0.17, 0.25, "A_func,b", "PLV of band b\n(5, 12, 12) per window\nsymmetric, zero diagonal", fc="#EAF2FA")
    box(ax, 0.01, 0.34, 0.17, 0.25, "X_b", "[log PSD, relative PSD]\n(5, 12, 2) per window\nz-scored per band/ROI/feature", fc="#EAF2FA")
    box(ax, 0.01, 0.06, 0.17, 0.25, "A_anat (fixed)", "(12, 12), 22 edges\ncortical-surface contiguity\nno data, no labels", fc="#FFF4D6")
    box(ax, 0.25, 0.46, 0.19, 0.36, "GCN layer 1\nfunctional", "H1_b = ReLU(N(A_func,b)\n X_b W1_b + c1_b)\nW1_b: 2 -> 32\nout (B, 5, 12, 32)", fc="#EAF2FA")
    box(ax, 0.49, 0.46, 0.19, 0.36, "GCN layer 2\nanatomical", "H2_b = ReLU(N(A_anat)\n H1_b W2_b + c2_b)\nW2_b: 32 -> 2\nout (B, 5, 12, 2)", fc="#FFF4D6")
    box(ax, 0.73, 0.46, 0.17, 0.36, "Flatten", "5 x 12 x 2 = 120\ndropout 0.3\n(B, 120)", fc="#F2F2F2")
    box(ax, 0.30, 0.06, 0.14, 0.27, "FC1", "120 -> 32\nReLU", fc="#F2F2F2")
    box(ax, 0.47, 0.06, 0.14, 0.27, "FC2", "32 -> 32\nReLU", fc="#F2F2F2")
    box(ax, 0.64, 0.06, 0.12, 0.27, "Output", "32 -> 1\nlogit", fc="#F2F2F2")
    box(ax, 0.80, 0.06, 0.195, 0.27, "Decision", "sigmoid -> window prob.\nmean over windows =\nparticipant score\nthreshold 0.5", fc="#EAF6F0")
    arrow(ax, (0.185, 0.74), (0.245, 0.70)); arrow(ax, (0.185, 0.46), (0.245, 0.58))
    arrow(ax, (0.185, 0.18), (0.485, 0.50), 0.25)
    arrow(ax, (0.445, 0.64), (0.485, 0.64)); arrow(ax, (0.685, 0.64), (0.725, 0.64))
    arrow(ax, (0.815, 0.455), (0.40, 0.34), 0.2)
    arrow(ax, (0.445, 0.195), (0.465, 0.195)); arrow(ax, (0.615, 0.195), (0.635, 0.195)); arrow(ax, (0.765, 0.195), (0.795, 0.195))
    ax.text(0.5, 0.005, "5,771 trainable parameters (counted by instantiating XuDualGCN: in_dim 2, 5 bands, 12 ROIs; no data used). B = batch of windows (32). "
            "Source: src/adgraph/models/xu_dual.py.", ha="center", fontsize=5.4, color="#444")
    save(fig, cfg, "fig02_exp8a_architecture", man, "fig02", [])


# -------------------------------------------------------------------------------------------- FIGURE 3 atlas resolution
def _read_resolution(cfg) -> pd.DataFrame | None:
    p = resolve(cfg, "resolution_csv", "resolution_by_atlas.csv", required=False)
    return pd.read_csv(p) if p is not None else None


def fig_atlas(cfg, d, man):
    t = json.loads(cfg.atlas_tables.read_text())
    cfg.inputs["atlas_tables.json"] = cfg.atlas_tables
    res = _read_resolution(cfg)
    if res is None:
        raise MissingInput("resolution_by_atlas.csv not found (pass --resolution-csv); Fig. 3 needs the saved resolution table")
    fig = plt.figure(figsize=(7.16, 5.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1, 1], hspace=0.95, wspace=0.7)
    # (a) atlas geometry from saved centroids (fsaverage white surface, mm): axial (x, y) projection
    for j, a in enumerate(("dk68", "r18", "lobes12")):
        ax = fig.add_subplot(gs[0, j])
        ta = t[a]
        c = np.array(ta["centroids_mm"]); lobes = ta["lobes"]
        names = ta["regions"]
        if a == "lobes12":
            idx = {n: i for i, n in enumerate(names)}
            for x, y in ta["contiguous_pairs"]:
                ax.plot([c[idx[x], 0], c[idx[y], 0]], [c[idx[x], 1], c[idx[y], 1]], color="#999", lw=0.8, zorder=1)
        cols = [LOBE_COLOR[lobes[n]] if isinstance(lobes, dict) else LOBE_COLOR[lobes[i]] for i, n in enumerate(names)]
        ax.scatter(c[:, 0], c[:, 1], c=cols, s=14 if a != "dk68" else 8, edgecolor="k", linewidth=0.3, zorder=2)
        ax.set_aspect("equal"); ax.set_xlim(-75, 75); ax.set_ylim(-110, 80)
        ax.set_title(f"{a}: {len(names)} regions", fontsize=7, pad=6)
        ax.set_xlabel("x (mm), L - R"); ax.set_ylabel("y (mm), posterior - anterior" if j == 0 else "")
        panel(ax, "abc"[j])
    axl = fig.add_subplot(gs[0, 3]); axl.axis("off")
    for k, (lobe, col) in enumerate(LOBE_COLOR.items()):
        axl.scatter([0], [0.9 - k * 0.14], c=col, s=18, edgecolor="k", linewidth=0.3); axl.text(0.12, 0.9 - k * 0.14, lobe, va="center", fontsize=6.5)
    axl.text(0.0, 0.02, "Region centroids on the fsaverage\nwhite surface (saved atlas tables).\nLines (lobes12): 22 surface-\ncontiguity edges = A_anat.", fontsize=5.8, va="top")
    axl.set_xlim(-0.1, 1); axl.set_ylim(-0.4, 1)
    # (d-g) resolution metrics
    order = ["dk68", "dk34_gyral", "r18", "lobes12"]
    short = ["dk68", "dk34", "r18", "lobes12"]
    def bars(ax, col, title, thr, higher_better, letter):
        for k, inv in enumerate(("lcmv", "wmne")):
            v = [float(res[(res.atlas == a) & (res.inverse == inv)][col].iloc[0]) for a in order]
            ax.bar(np.arange(4) + (k - 0.5) * 0.36, v, 0.34, color=OKABE[inv], label=inv.upper())
        if thr is not None:
            ax.axhline(thr, color="k", ls="--", lw=0.7)
            ax.text(3.45, thr, " rule", fontsize=5.5, va="center")
        ax.set_xticks(range(4)); ax.set_xticklabels(short, rotation=35, ha="right"); ax.set_title(title, fontsize=7)
        panel(ax, letter)
    ax = fig.add_subplot(gs[1, 0]); bars(ax, "crosstalk_fraction_median", "median cross-talk\n(rule <= 0.60)", 0.60, False, "d"); pass
    ax = fig.add_subplot(gs[1, 1]); bars(ax, "rois_dominated_by_own_region", "own-region dominance\n(rule >= 0.80)", 0.80, True, "e")
    ax = fig.add_subplot(gs[1, 2]); bars(ax, "participation_ratio", "participation ratio", None, True, "f"); ax.axhline(12.8, color=OKABE["green"], ls=":", lw=0.8); ax.text(0, 13.2, "effective rank of the leadfield 12.8", fontsize=5.3, color=OKABE["green"]); ax.legend(frameon=False, loc="center right", fontsize=6)
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    ax.text(0, 0.95, "19-electrode leadfield\n(exact fsaverage BEM)\nnumerical rank 18\nentropy-effective rank 12.8\n(dotted line, panel f; no rule\nthreshold on participation ratio)\n\nredundant ROI signals:\ndk68 50 | dk34_gyral 16\nr18 0 | lobes12 0", fontsize=6, va="top")
    # (g) classification by atlas: paired lines, Exp4 and Exp5 reference models, mode B dense
    ax = fig.add_subplot(gs[2, :2]); panel(ax, "g")
    xs = {"dk68": 0, "lobes12": 1, "r18": 2}
    spec = [("exp4_band_mean", "plv__psd", "Exp4"), ("exp5_multibranch", "plv__separate__mean", "Exp5")]
    for exp, var, lab in spec:
        for inv in ("lcmv", "wmne"):
            rows = d[(d.experiment == exp) & (d.variant == var) & (d.inverse == inv) & (d["mode"] == "B") & (d.graph == "dense")]
            if set(rows.atlas) != set(xs):
                raise MissingInput(f"{exp}/{var}/{inv}: atlas rows {sorted(rows.atlas)} (need dk68, lobes12, r18)")
            r = rows.set_index("atlas").loc[list(xs)]
            ax.plot(range(3), r.auc.values, marker="o" if exp == "exp4_band_mean" else "s", ms=3.5, lw=1,
                    ls="-" if exp == "exp5_multibranch" else "--", color=OKABE[inv], label=f"{lab} {inv.upper()}")
    ax.set_xticks(range(3)); ax.set_xticklabels(["dk68 (68)", "lobes12 (12, recorded)", "r18 (18, exploratory)"])
    ax.set_ylabel("participant ROC-AUC"); ax.set_ylim(0.70, 0.90); ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.set_title("classification AUC by atlas (dev CV, n = 52)", fontsize=7)
    ax = fig.add_subplot(gs[2, 2:]); panel(ax, "h")
    for exp, var, lab in spec:
        for inv in ("lcmv", "wmne"):
            rows = d[(d.experiment == exp) & (d.variant == var) & (d.inverse == inv) & (d["mode"] == "B") & (d.graph == "dense")].set_index("atlas").loc[list(xs)]
            ax.plot(range(3), rows.balanced_accuracy.values, marker="o" if exp == "exp4_band_mean" else "s", ms=3.5, lw=1,
                    ls="-" if exp == "exp5_multibranch" else "--", color=OKABE[inv])
    ax.set_xticks(range(3)); ax.set_xticklabels(["dk68", "lobes12", "r18"]); ax.set_ylabel("balanced accuracy"); ax.set_ylim(0.60, 0.90)
    ax.set_title("same models, balanced accuracy", fontsize=7)
    save(fig, cfg, "fig03_atlas_resolution", man, "fig03", ["resolution_by_atlas.csv", "atlas_tables.json", "development_results_all.csv"],
         "dk34_gyral: analytic resolution only (no classifier trained). Cortical-surface mesh rendering: see render_cortical_surface().")


def render_cortical_surface(cfg: Config, atlas_name: str = "lobes12"):
    """Optional: render an atlas on the fsaverage inflated surface with MNE (needs mne, fsaverage and the repo's
    adgraph package; run in Colab after notebook 02 has cached fsaverage). Not executed in the offline test build."""
    try:
        import mne
        from adgraph.source import atlas as A
    except Exception as e:                                   # pragma: no cover
        raise MissingInput(f"surface rendering needs mne + adgraph + fsaverage: {e}")
    subjects_dir = Path(mne.datasets.fetch_fsaverage(verbose=False)).parent
    labels = mne.read_labels_from_annot("fsaverage", "aparc", subjects_dir=subjects_dir, verbose="error")
    labels = [l for l in labels if not l.name.startswith("unknown")]
    merged = A.merged_labels(sorted(labels, key=lambda l: l.name), atlas_name)
    lobes = A.lobe_of(atlas_name)
    brain = mne.viz.Brain("fsaverage", hemi="split", surf="inflated", subjects_dir=subjects_dir, size=(1200, 600), views=["lat", "med"])
    for lab in merged:
        brain.add_label(lab, color=LOBE_COLOR[lobes[lab.name]], hemi=lab.hemi, borders=True)
    out = cfg.out / f"fig03_surface_{atlas_name}.png"
    brain.save_image(out); brain.close()
    return out


# ------------------------------------------------------------------------------------------- FIGURE 4 landscape / bands
def fig_landscape(cfg, d, man):
    g = d[(d.scope == "GNN") & (d["mode"] == "B")]
    fig = plt.figure(figsize=(7.16, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0]); panel(ax, "a")
    rng = np.random.default_rng(7)
    for i, f in enumerate(FAMILIES):
        for inv, off in (("lcmv", -0.17), ("wmne", 0.17)):
            v = g[(g.experiment == f) & (g.inverse == inv)].auc.values
            ax.scatter(i + off + rng.uniform(-0.08, 0.08, len(v)), v, s=7, color=OKABE[inv], alpha=0.75, lw=0)
            ax.hlines(np.median(v), i + off - 0.14, i + off + 0.14, color=OKABE[inv], lw=1.6)
    ax.axhline(0.5, color="#888", ls="--", lw=0.6)
    ax.set_xticks(range(len(FAMILIES))); ax.set_xticklabels([FAM_LABEL[f] for f in FAMILIES])
    ax.set_ylabel("participant ROC-AUC (dev CV, n = 52)"); ax.set_ylim(0.35, 0.92)
    ax.scatter([], [], color=OKABE["lcmv"], s=7, label="LCMV"); ax.scatter([], [], color=OKABE["wmne"], s=7, label="wMNE")
    ax.legend(frameon=False, loc="lower right", title="bar: median", title_fontsize=6)
    ax.set_title("GNN families, mode B (dots: configurations)", fontsize=7)
    ax = fig.add_subplot(gs[1]); panel(ax, "b")
    e3 = g[(g.experiment == "exp3_single_band") & (g.atlas == "lobes12") & (g.graph == "dense")]
    cols = [("lcmv", "plv", "LCMV\nPLV"), ("lcmv", "ciplv", "LCMV\n|ciPLV|"), ("wmne", "plv", "wMNE\nPLV"), ("wmne", "ciplv", "wMNE\n|ciPLV|")]
    M = np.full((5, 4), np.nan)
    for j, (inv, conn, _) in enumerate(cols):
        for i, b in enumerate(BANDS):
            r = e3[(e3.inverse == inv) & (e3.variant == f"{conn}__{b}")]
            if len(r) != 1:
                raise MissingInput(f"Exp3 {inv} {conn}__{b}: expected one saved result, found {len(r)}")
            M[i, j] = r.auc.iloc[0]
    im = ax.imshow(M, vmin=0.4, vmax=0.9, cmap="viridis", aspect="auto")
    for i in range(5):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if M[i, j] < 0.68 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels([c[2] for c in cols]); ax.set_yticks(range(5)); ax.set_yticklabels(BANDS)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("participant ROC-AUC", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    ax.set_title("Exp3 single-band GCN, observed AUC", fontsize=7)
    save(fig, cfg, "fig04_family_and_band_performance", man, "fig04", ["development_results_all.csv"])
    # supplementary: balanced-accuracy version of panel b
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    M2 = np.zeros((5, 4))
    for j, (inv, conn, _) in enumerate(cols):
        for i, b in enumerate(BANDS):
            M2[i, j] = e3[(e3.inverse == inv) & (e3.variant == f"{conn}__{b}")].balanced_accuracy.iloc[0]
    im = ax.imshow(M2, vmin=0.4, vmax=0.9, cmap="viridis", aspect="auto")
    for i in range(5):
        for j in range(4):
            ax.text(j, i, f"{M2[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if M2[i, j] < 0.68 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels([c[2] for c in cols]); ax.set_yticks(range(5)); ax.set_yticklabels(BANDS)
    ax.spines[:].set_visible(False); fig.colorbar(im, ax=ax, fraction=0.05).set_label("balanced accuracy (threshold 0.5)", fontsize=6.5)
    ax.set_title("Exp3 single-band GCN, balanced accuracy", fontsize=7)
    save(fig, cfg, "figS_exp3_band_heatmap_balanced_accuracy", man, "figS_bands", ["development_results_all.csv"])


# ------------------------------------------------------------------------------------------------ FIGURE 5 ROC + CM
def fig_roc(cfg, d, man):
    units = [PRIMARY, EXP5_R18, EXP4_REF]
    preds = {u.key: load_predictions(cfg, d, u) for u in units}
    y = preds[PRIMARY.key].y.values
    for u in units[1:]:
        if not (preds[u.key].key.values == preds[PRIMARY.key].key.values).all():
            raise ValueError("participants / order differ between prediction files")
    fig = plt.figure(figsize=(7.16, 3.9))
    gs = fig.add_gridspec(2, 5, width_ratios=[2.0, 0.05, 1, 1, 1], hspace=0.9, wspace=0.7)
    ax = fig.add_subplot(gs[:, 0]); panel(ax, "a")
    colors = {PRIMARY.key: OKABE["lcmv"], EXP5_R18.key: OKABE["wmne"], EXP4_REF.key: OKABE["grey"]}
    ls = {PRIMARY.key: "-", EXP5_R18.key: "--", EXP4_REF.key: ":"}
    stats = {}
    for u in units:
        p = preds[u.key]
        fpr, tpr, _ = roc_curve(p.y, p.score)
        auc, lo, hi = delong_ci(p.y.values, p.score.values)
        row = pick(d, u)
        if abs(lo - float(row.auc_ci_low)) > 5e-3 or abs(hi - float(row.auc_ci_high)) > 5e-3:
            raise ValueError(f"{u.key}: recomputed DeLong CI deviates from the saved CI")
        stats[u.key] = (auc, lo, hi)
        ax.step(fpr, tpr, where="post", color=colors[u.key], ls=ls[u.key], lw=1.4,
                label=f"{u.label}\nAUC {auc:.3f} (DeLong 95% CI {lo:.3f}-{hi:.3f})")
    ax.plot([0, 1], [0, 1], color="#999", lw=0.6); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity")
    ax.set_title("participant-level ROC (out-of-fold, n = 52)", fontsize=7); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=5.4)
    ax.set_aspect("equal")
    for k, u in enumerate(units):
        axc = fig.add_subplot(gs[0, 2 + k])
        row = pick(d, u)
        cm = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        axc.imshow(cm, cmap="Blues", vmin=0, vmax=29)
        for i in range(2):
            for j in range(2):
                axc.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 15 else "black")
        axc.set_xticks([0, 1]); axc.set_xticklabels(["HC", "AD"]); axc.set_yticks([0, 1]); axc.set_yticklabels(["HC", "AD"])
        axc.set_xlabel("predicted"); axc.set_ylabel("true" if k == 0 else "")
        axc.set_title(["Exp8a primary", "Exp5 r18", "Exp4 ref."][k] + f"\nacc {row.accuracy:.3f}\nsens {row.sensitivity:.3f}, spec {row.specificity:.3f}", fontsize=5.3)
        if k == 0:
            panel(axc, "b")
    # paired comparisons (recomputed from saved predictions, uncorrected, descriptive)
    axt = fig.add_subplot(gs[1, 2:]); axt.axis("off")
    lines = ["Paired comparisons recomputed from the saved predictions (descriptive, uncorrected):"]
    for a, b in ((PRIMARY, EXP5_R18), (PRIMARY, EXP4_REF)):
        diff, pv = delong_paired(y, preds[a.key].score.values, preds[b.key].score.values)
        lo, hi = bootstrap_diff(y, preds[a.key].score.values, preds[b.key].score.values)
        lines.append(f"{a.label.split(' (')[0]} - {b.label.split(' (')[0]}: dAUC {diff:+.3f}, paired DeLong p = {pv:.2f}, bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
    axt.text(0, 0.9, "\n".join(lines), fontsize=5.4, va="top", wrap=True)
    man_note = "; ".join(f"{k}: AUC {v[0]:.4f}" for k, v in stats.items())
    save(fig, cfg, "fig05_roc_confusion", man, "fig05", [f"pred:{u.key}" for u in units] + ["development_results_all.csv"], man_note)


# -------------------------------------------------------------------------------- FIGURE 6 ablations + robustness
def fig_ablation(cfg, d, c, man):
    fig = plt.figure(figsize=(7.16, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0]); panel(ax, "a")
    ab = c[c.family == "Exp8a-ablations"].copy()
    if ab.empty:
        raise MissingInput("comparisons.csv has no 'Exp8a-ablations' family")
    ab["label"] = ab.variant_b.str.replace("plv__psd_de_strength__", "", regex=False).str.replace("plv__", "feat: ", regex=False)
    ab = ab.sort_values(["label", "inverse_a"]).reset_index(drop=True)
    for i, r in ab.iterrows():
        col = OKABE[r.inverse_a]
        ax.plot([r.boot_low, r.boot_high], [i, i], color=col, lw=1.2)
        ax.plot(r.auc_diff, i, "o", color=col, ms=3.5)
        ax.text(0.305, i, f"p_Holm = {r.p_holm:.2f}", fontsize=5.4, va="center")
    ax.axvline(0, color="k", lw=0.6); ax.set_xlim(-0.2, 0.42)
    ax.set_yticks(range(len(ab))); ax.set_yticklabels([f"{l} ({inv.upper()})" for l, inv in zip(ab.label, ab.inverse_a)], fontsize=5.6)
    ax.invert_yaxis(); ax.set_xlabel("AUC difference: full model (plv__psd_de_strength) minus variant\n(paired participant bootstrap 95% CI, uncorrected; Holm p from comparisons.csv)")
    ax.set_title("Exp8a ablations: none supported after Holm correction", fontsize=7)
    ax = fig.add_subplot(gs[1]); panel(ax, "b")
    sets = [("PLV - |ciPLV|", "PLV-vs-ciPLV"), ("LCMV - wMNE", "LCMV-vs-wMNE"), ("mode B - mode A", "modeB-vs-modeA")]
    rng = np.random.default_rng(3); rows = []
    for k, (lab, fam) in enumerate(sets):
        x = c[c.family == fam]
        if x.empty:
            raise MissingInput(f"comparisons.csv has no '{fam}' family")
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(x)), x.auc_diff, s=6, color=[OKABE["purple"] if s else "#555555" for s in x.supported], alpha=0.8, lw=0)
        ax.hlines(x.auc_diff.median(), k - 0.28, k + 0.28, color="k", lw=1.4)
        rows.append(f"{lab}: n={len(x)} (GNN only), median {x.auc_diff.median():+.3f}, higher {int((x.auc_diff > 0).sum())}/lower {int((x.auc_diff < 0).sum())}, supported {int(x.supported.sum())}")
    ax.axhline(0, color="#888", lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels([s[0] for s in sets], rotation=15, ha="right")
    ax.set_ylabel("AUC difference of matched pair (each dot one pair)")
    ax.set_title("matched robustness comparisons", fontsize=7)
    ax.text(0.0, -0.22, "\n".join(rows), transform=ax.transAxes, fontsize=5.3, va="top")
    save(fig, cfg, "fig06_ablation_and_robustness", man, "fig06", ["comparisons.csv"], "GNN-only subsets of the saved paired comparisons (Exp0/Exp1 rows removed); Holm p-values are those saved for the full family.")


# ------------------------------------------------------------------------------------------------- supplementary
def fig_selection(cfg, d, man):
    p = resolve(cfg, "selection_json", "selection_check.json")
    s = json.loads(Path(p).read_text())["sets"]
    keep = [(k, v) for k, v in s.items() if k.startswith(("GNN", "Exp 8a"))]
    if not keep:
        raise MissingInput("selection_check.json has no GNN candidate sets")
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    for i, (k, v) in enumerate(keep):
        ax.plot([v["mean_oob_winner_auc"], v["best_auc_observed"]], [i, i], color="#999", lw=1)
        ax.plot(v["best_auc_observed"], i, "o", color=OKABE["wmne"], ms=4)
        ax.plot(v["mean_oob_winner_auc"], i, "s", color=OKABE["lcmv"], ms=4)
        ax.plot(v["optimism_corrected_best_auc"], i, "D", color=OKABE["green"], ms=3.5, mfc="none")
        ax.plot([v["max_null_auc_p95"]] * 2, [i - 0.25, i + 0.25], color="k", lw=1.4)
        ax.text(0.93, i, f"n_cand = {v['n_candidates']}, optimism +{v['optimism_estimate']:.3f}", fontsize=5.4, va="center")
    ax.set_yticks(range(len(keep))); ax.set_yticklabels([k for k, _ in keep], fontsize=6); ax.invert_yaxis(); ax.set_xlim(0.65, 1.08)
    ax.set_xlabel("AUC of the best candidate (n = 52 participants)")
    for m, lab, col in (("o", "observed best", OKABE["wmne"]), ("s", "bootstrap out-of-bag winner (mean)", OKABE["lcmv"]), ("D", "observed - optimism", OKABE["green"])):
        ax.plot([], [], m, color=col, label=lab, mfc="none" if m == "D" else col)
    ax.plot([], [], color="k", lw=1.4, label="95th percentile of best-of-m under permuted labels"); ax.legend(frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set_title("Selection optimism (participants resampled; winner chosen in-bag, scored on out-of-bag participants)", fontsize=7)
    save(fig, cfg, "figS_selection_optimism", man, "figS_selection", ["selection_check.json"], "Out-of-bag estimate is a within-development resampling estimate, not an independent test.")


def fig_holdout(cfg, d, man):
    pr = load_predictions(cfg, d, HOLDOUT_UNIT, holdout=True)
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.7), gridspec_kw={"width_ratios": [1.3, 1]})
    fpr, tpr, _ = roc_curve(pr.y, pr.score); auc, lo, hi = delong_ci(pr.y.values, pr.score.values)
    axs[0].step(fpr, tpr, where="post", color=OKABE["wmne"], lw=1.4); axs[0].plot([0, 1], [0, 1], color="#999", lw=0.6)
    axs[0].set_xlabel("1 - specificity"); axs[0].set_ylabel("sensitivity"); axs[0].set_aspect("equal")
    axs[0].set_title(f"AUC {auc:.3f} (DeLong 95% CI {lo:.3f}-{hi:.3f})\n13 participants: 7 AD / 6 HC", fontsize=6.5)
    pred = (pr.score >= 0.5).astype(int); tp = int(((pred == 1) & (pr.y == 1)).sum()); tn = int(((pred == 0) & (pr.y == 0)).sum())
    fp = int(((pred == 1) & (pr.y == 0)).sum()); fn = int(((pred == 0) & (pr.y == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]]); axs[1].imshow(cm, cmap="Blues", vmin=0, vmax=7)
    for i in range(2):
        for j in range(2):
            axs[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["HC", "AD"]); axs[1].set_yticks([0, 1]); axs[1].set_yticklabels(["HC", "AD"]); axs[1].set_xlabel("predicted"); axs[1].set_ylabel("true")
    fig.suptitle("DESCRIPTIVE, previously inspected holdout.\nwMNE anatomical-both variant only; NOT the LCMV plv__psd primary model.", fontsize=6, color="#8a1f11", y=1.12)
    save(fig, cfg, "figS_holdout_descriptive", man, "figS_holdout", ["pred:holdout"], f"recomputed from saved holdout predictions: AUC {auc:.4f}")


def fig_graph_sparsity(cfg, d, man):
    g = d[(d.scope == "GNN") & (d["mode"] == "B") & (d.atlas == "lobes12")]
    order = ["dense", "knn_union_f10", "knn_union_f20", "knn_union_f30", "knn_mutual_f10", "knn_mutual_f20", "knn_mutual_f30", "proportional_f20", "contiguity", "contiguity_knn_union_f20", "feature_similarity_knn_union_f20"]
    fig, ax = plt.subplots(figsize=(7.16, 2.8))
    for k, (exp, var, mk) in enumerate((("exp4_band_mean", "plv__psd", "o"), ("exp5_multibranch", "plv__separate__mean", "s"))):
        for inv, off in (("lcmv", -0.1), ("wmne", 0.1)):
            v = []
            for o in order:
                r = g[(g.experiment == exp) & (g.variant == var) & (g.inverse == inv) & (g.graph == o)]
                if len(r) != 1:
                    raise MissingInput(f"{exp}/{var}/{inv}/{o}: expected one saved result, found {len(r)}")
                v.append(r.auc.iloc[0])
            ax.plot(np.arange(len(order)) + off, v, marker=mk, ms=3.5, lw=0.8, ls="-" if k else "--", color=OKABE[inv], label=f"{FAM_LABEL[exp]} {inv.upper()}")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace("_", " ") for o in order], rotation=30, ha="right", fontsize=5.8)
    ax.set_ylabel("participant ROC-AUC"); ax.legend(frameon=False, ncol=4, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0.74, 0.88)
    ax.set_title("Graph construction on lobes12 (dense kept by the pre-specified rule)", fontsize=7, pad=14)
    save(fig, cfg, "figS_graph_construction", man, "figS_graph", ["development_results_all.csv"])


def fig_exp6(cfg, d, c, man):
    g = d[(d.experiment == "exp6_psd_strength") & (d["mode"] == "B")]
    order = ["plv__full", "plv__no_psd", "plv__no_strength", "plv__no_learned_fusion", "plv__drop_delta", "plv__drop_theta", "plv__drop_alpha", "plv__drop_beta", "plv__drop_gamma", "ciplv__full"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for inv, off in (("lcmv", -0.17), ("wmne", 0.17)):
        r = g[g.inverse == inv].set_index("variant").loc[order]
        ax.errorbar(r.auc, np.arange(len(order)) + off, xerr=[r.auc - r.auc_ci_low, r.auc_ci_high - r.auc], fmt="o", ms=3, lw=0.8, color=OKABE[inv], label=inv.upper())
    ax.set_yticks(range(len(order))); ax.set_yticklabels([("|ciPLV| full" if o == "ciplv__full" else o.replace("plv__", "")) for o in order], fontsize=6); ax.invert_yaxis(); ax.axvline(0.5, color="#888", lw=0.6, ls="--")
    ax.set_xlabel("participant ROC-AUC with DeLong 95% CI (n = 52)"); ax.legend(frameon=False, fontsize=6); ax.set_title("Exp6 feature and band ablations (mode B)", fontsize=7)
    save(fig, cfg, "figS_exp6_ablations", man, "figS_exp6", ["development_results_all.csv"])


# ------------------------------------------------------------------------------------------------ inventory / manifest
def existing_figure_inventory(cfg: Config) -> pd.DataFrame:
    rows = []
    if cfg.existing_fig_dir and cfg.existing_fig_dir.exists():
        for f in sorted(cfg.existing_fig_dir.glob("*")):
            if f.is_file():
                rows.append({"file": f.name, "path": str(f), "bytes": f.stat().st_size, "sha256": sha256(f)})
    return pd.DataFrame(rows)


def write_manifests(cfg: Config, man: list, info: dict) -> None:
    pd.DataFrame(man).to_csv(cfg.out.parent / "figure_manifest.csv", index=False)
    inv = [{"name": k, "path": str(v), "sha256": sha256(Path(v)), "bytes": Path(v).stat().st_size} for k, v in cfg.inputs.items()]
    ex = existing_figure_inventory(cfg)
    for _, r in ex.iterrows():
        inv.append({"name": "existing_figure:" + r.file, "path": r.path, "sha256": r.sha256, "bytes": r.bytes})
    pd.DataFrame(inv).to_csv(cfg.out.parent / "source_inventory.csv", index=False)
    (cfg.out.parent / "run_info.json").write_text(json.dumps(info, indent=2, default=str))


FIGS = ("pipeline", "architecture", "atlas", "landscape", "roc", "ablation", "selection", "holdout", "graph", "exp6")


def build(cfg: Config, only: tuple[str, ...] = FIGS) -> list:
    guard_output(cfg.out)
    style()
    d = load_results(cfg)
    info = validate_inventory(d)
    man: list = []
    c = None
    skipped = {}
    for name in only:
        try:
            if name == "pipeline": fig_pipeline(cfg, d, man)
            elif name == "architecture": fig_architecture(cfg, d, man)
            elif name == "atlas": fig_atlas(cfg, d, man)
            elif name == "landscape": fig_landscape(cfg, d, man)
            elif name == "roc": fig_roc(cfg, d, man)
            elif name == "ablation":
                c = c if c is not None else load_comparisons(cfg); fig_ablation(cfg, d, c, man)
            elif name == "selection": fig_selection(cfg, d, man)
            elif name == "holdout": fig_holdout(cfg, d, man)
            elif name == "graph": fig_graph_sparsity(cfg, d, man)
            elif name == "exp6":
                c = c if c is not None else load_comparisons(cfg); fig_exp6(cfg, d, c, man)
            print(f"[ok]      {name}")
        except MissingInput as e:
            skipped[name] = str(e); print(f"[MISSING] {name}: {e}")
    info["skipped_missing_input"] = skipped
    write_manifests(cfg, man, info)
    return man


def parse(argv=None) -> Config:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--processed", type=Path, help="AD-LEB/processed (contains FINAL_v1)")
    a.add_argument("--out", type=Path, default=HERE / "generated")
    for n in ("results-csv", "comparisons-csv", "selection-json", "resolution-csv", "pred-dir", "holdout-pred", "existing-fig-dir", "atlas-tables"):
        a.add_argument("--" + n, type=Path)
    a.add_argument("--dpi", type=int, default=600)
    a.add_argument("--only", nargs="*", choices=FIGS)
    ns = a.parse_args(argv)
    cfg = Config(out=ns.out, processed=ns.processed, dpi=ns.dpi)
    for n in ("results_csv", "comparisons_csv", "selection_json", "resolution_csv", "pred_dir", "holdout_pred", "existing_fig_dir"):
        setattr(cfg, n, getattr(ns, n))
    if ns.atlas_tables:
        cfg.atlas_tables = ns.atlas_tables
    cfg._only = tuple(ns.only) if ns.only else FIGS
    return cfg


if __name__ == "__main__":
    cfg = parse()
    build(cfg, cfg._only)
