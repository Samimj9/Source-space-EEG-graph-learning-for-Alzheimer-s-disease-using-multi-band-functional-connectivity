"""Cohort QC report of one store and the gate every experiment passes before training (docs/final/qc.md).

    <processed>/FINAL_v1/qc/<inverse>_<mode>_<atlas>/
        participant_summaries.csv   label-blind summaries per participant (``participants.participant_summaries``)
        window_summaries.csv.gz     the per-window summaries they come from
        technical_problems.csv      category A findings (empty when clean)
        robust_flags.csv            category B: per participant, flagged-feature count, max |modified z|, robust
                                    distance, multivariate outlier, PCA coordinates
        flagged_features.csv        category B: every flagged (participant, feature, modified z, value)
        qc_report.md                human-readable summary, incl. flags by diagnosis (description only)
        pca.png                     exploratory 2-D projection (reference: development participants)
        qc.done.json                completion marker (identity: store graph_run_id, reference participants, QC code)
    <processed>/FINAL_v1/qc/decisions.yaml   the author's decisions on category A findings (never written by code)

``require_qc`` loads (or computes once) the report and refuses to start an experiment while a category A finding has
no decision, or when a decision asks to exclude a participant: an exclusion changes the frozen participant split and
is therefore a protocol amendment that must be implemented and logged deliberately, never applied by a QC rule.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .. import release
from ..checkpoint import atomic_write_csv, atomic_write_json, atomic_write_text, check_unit, write_marker
from ..graphs.store import GraphStore, graph_run_id
from ..provenance import source_fingerprint
from .participants import Z_THRESHOLD, participant_summaries, robust_flags, technical_problems, window_summaries

QC_VERSION = "qc-1"


class QCUnresolved(RuntimeError):
    """Category A findings without a recorded decision, or a requested exclusion."""


def qc_dir(processed: str | Path, inverse: str, mode: str, atlas: str) -> Path:
    return release.root(processed) / "qc" / f"{inverse}_{mode}_{atlas}"


def decisions_path(processed: str | Path) -> Path:
    return release.root(processed) / "qc" / "decisions.yaml"


def load_decisions(processed: str | Path) -> dict:
    p = decisions_path(processed)
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def development_keys(store: GraphStore) -> list[str]:
    split = json.loads((Path(store.root) / "xu_split.json").read_text()) if store.root else None
    if split:
        return list(split["development"])
    from ..evaluation.splits import subject_table, xu_partition

    return xu_partition(subject_table(store.windows))[0]


def identity(store: GraphStore, reference: list[str]) -> dict:
    return {"kind": "participant_qc", "qc_version": QC_VERSION, "graph_run_id": graph_run_id(store),
            "reference": sorted(reference), "z_threshold": Z_THRESHOLD, "code": source_fingerprint(("qc",))}


def compute(store: GraphStore, out: Path, reference: list[str] | None = None, label: str = "") -> dict:
    """Run every check on ``store`` and write the report to ``out`` (atomic files, marker last)."""
    reference = reference or development_keys(store)
    ident = identity(store, reference)
    if check_unit(out, "qc", ident)[0] == "complete":
        return load(out)
    win = window_summaries(store)
    summ = participant_summaries(store, win)
    tech = technical_problems(store)
    flags, long = robust_flags(summ, reference)
    labels = store.windows.groupby("key")["y"].first().reindex(summ.index)
    flags.insert(0, "y", labels)
    flags["category_A"] = flags.index.isin(set(tech["key"]))
    files = [atomic_write_csv(out / "participant_summaries.csv", summ.reset_index()),
             atomic_write_csv(out / "technical_problems.csv", tech),
             atomic_write_csv(out / "robust_flags.csv", flags.reset_index()),
             atomic_write_csv(out / "flagged_features.csv", long)]
    win.to_csv(out / "window_summaries.csv.gz", index=False)
    files.append(out / "window_summaries.csv.gz")
    summary = {"label": label, "n_participants": int(len(summ)), "n_reference": len(reference),
               "category_A_participants": sorted(set(tech["key"])), "n_category_A_findings": int(len(tech)),
               "category_B_participants": sorted(flags.index[flags["category_B"]].tolist()),
               "multivariate_outliers": sorted(flags.index[flags["multivariate_outlier"]].tolist()),
               "pca_explained": flags.attrs.get("pca_explained"), "mcd_cutoff": flags.attrs.get("mcd_cutoff"),
               "computed_at": datetime.now(timezone.utc).isoformat()}
    files.append(atomic_write_json(out / "qc_summary.json", summary))
    files.append(atomic_write_text(out / "qc_report.md", markdown(summary, flags, long, tech)))
    try:
        files.append(_pca_plot(flags, reference, out / "pca.png"))
    except Exception:                                  # a figure must never block the QC itself
        pass
    write_marker(out, "qc", ident, files)
    return load(out)


def load(out: Path) -> dict:
    return {"summary": json.loads((out / "qc_summary.json").read_text()),
            "flags": pd.read_csv(out / "robust_flags.csv").set_index("key"),
            "technical": pd.read_csv(out / "technical_problems.csv"),
            "summaries": pd.read_csv(out / "participant_summaries.csv").set_index("key"), "dir": str(out)}


def markdown(summary: dict, flags: pd.DataFrame, long: pd.DataFrame, tech: pd.DataFrame) -> str:
    lines = [f"# Participant QC: {summary['label']}", "",
             f"{summary['n_participants']} participants; reference distribution: {summary['n_reference']} "
             "development participants (label-blind summaries; the diagnosis column only describes the flags).", "",
             "No participant is excluded by this report. Category A needs a decision in `FINAL_v1/qc/decisions.yaml`; "
             "category B is reviewed and kept; category C (difficult to classify) comes from out-of-fold predictions "
             "after the experiments (`scripts/final.py difficult`).", "",
             f"## Category A: technical corruption ({len(summary['category_A_participants'])} participants)", ""]
    if len(tech):
        lines += ["| participant | check | detail |", "|---|---|---|"]
        lines += [f"| {r.key} | {r.check} | {r.detail} |" for r in tech.itertuples()]
    else:
        lines.append("None.")
    b = flags[flags["category_B"]]
    lines += ["", f"## Category B: statistically unusual, kept ({len(b)} participants; |modified z| > {Z_THRESHOLD} or "
                  "multivariate MCD outlier)", ""]
    if len(b):
        lines += ["| participant | diagnosis | features flagged | max abs z | multivariate | top features |",
                  "|---|---|---|---|---|---|"]
        for k, r in b.iterrows():
            top = long[long["key"] == k].reindex(long[long["key"] == k]["modified_z"].abs()
                                                 .sort_values(ascending=False).index).head(3)
            lines.append(f"| {k} | {'AD' if r['y'] == 1 else 'HC'} | {int(r['n_features_flagged'])} | "
                         f"{r['max_abs_modified_z']:.1f} | {bool(r['multivariate_outlier'])} | "
                         + ", ".join(f"{f} ({z:+.1f})" for f, z in zip(top["feature"], top["modified_z"])) + " |")
        by = b.groupby("y").size().to_dict()
        lines += ["", f"Flags by diagnosis: AD {by.get(1, 0)}, HC {by.get(0, 0)} (description only)."]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def _pca_plot(flags: pd.DataFrame, reference: list[str], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "pc1" not in flags:
        raise ValueError("no PCA")
    fig, ax = plt.subplots(figsize=(5, 4))
    ref = flags.index.isin(reference)
    for y, mk in ((1, "o"), (0, "s")):
        m = flags["y"].to_numpy() == y
        ax.scatter(flags.loc[m & ref, "pc1"], flags.loc[m & ref, "pc2"], marker=mk, s=18, alpha=0.7,
                   label=f"{'AD' if y else 'HC'} (development)")
        ax.scatter(flags.loc[m & ~ref, "pc1"], flags.loc[m & ~ref, "pc2"], marker=mk, s=18, facecolors="none",
                   edgecolors="k", label=f"{'AD' if y else 'HC'} (holdout)")
    for k in flags.index[flags["category_B"]]:
        ax.annotate(k, (flags.loc[k, "pc1"], flags.loc[k, "pc2"]), fontsize=6)
    ax.set_xlabel("PC 1 (robust z of participant summaries)")
    ax.set_ylabel("PC 2")
    ax.set_title("Participant QC: exploratory projection (not a result)", fontsize=8)
    ax.legend(fontsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def require_qc(store: GraphStore, processed: str | Path, inverse: str, mode: str, atlas: str) -> dict:
    """The gate of every experiment: the store's QC exists (computed once), and every category A finding has a
    decision ``keep``. Raises ``QCUnresolved`` otherwise. Returns the loaded report."""
    out = qc_dir(processed, inverse, mode, atlas)
    rep = compute(store, out, label=f"{inverse} / mode {mode} / {atlas}")
    decisions = load_decisions(processed)
    unresolved, excluded = [], []
    for k in rep["summary"]["category_A_participants"]:
        d = (decisions.get(k) or {}).get("decision")
        if d is None:
            unresolved.append(k)
        elif d != "keep":
            excluded.append(k)
    if unresolved:
        raise QCUnresolved(f"category A (technical) QC findings without a decision for {unresolved}; review "
                           f"{out / 'qc_report.md'} and record a decision in {decisions_path(processed)}")
    if excluded:
        raise QCUnresolved(f"decisions request excluding {excluded}: an exclusion changes the frozen participant "
                           "split and must be implemented as a logged protocol amendment, not by the QC gate")
    return rep


def fold_reference_summaries(rep: dict) -> pd.DataFrame:
    """Participant summaries of a loaded report, for the fold-level QC (``qc.fold``)."""
    return rep["summaries"].drop(columns=[c for c in ("y",) if c in rep["summaries"]]).astype(float)


def describe(rep: dict) -> str:
    s = rep["summary"]
    return (f"QC {s['label']}: {s['n_participants']} participants, category A {len(s['category_A_participants'])}, "
            f"category B {len(s['category_B_participants'])} (kept), multivariate outliers "
            f"{len(s['multivariate_outliers'])}; report {rep['dir']}")


def _np(x):
    return x.tolist() if isinstance(x, np.ndarray) else x
