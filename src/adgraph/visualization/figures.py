"""Figure registry and driver: config -> figures, each with data, metadata and a manifest.

Every family reads only saved files: experiment output directories (``results``) and/or
a graph run (``graphs``). Nothing is recomputed from raw EEG, and no model is trained.
"""

from __future__ import annotations

from pathlib import Path

from . import connectivity, embedding, fusion, performance, spectral, subjects
from .performance import IncomparableTiming
from .results import MissingInputs, index_results, reset_missing
from .stats import InvalidStatistics, load_stats
from .style import load_style, styled
from .writer import FigureWriter

# family -> (function, input kind)
FAMILIES = {
    "performance": (performance.fig_performance, "results"),
    "paired_conditions": (performance.fig_paired_conditions, "results"),
    "band_comparison": (performance.fig_band_comparison, "results"),
    "ablation": (performance.fig_ablation, "results"),
    "generalization": (performance.fig_generalization, "results"),
    "age_sensitivity": (performance.fig_age_sensitivity, "results"),
    "complexity": (performance.fig_complexity, "results"),
    "roc": (subjects.fig_roc, "results"),
    "confusion": (subjects.fig_confusion, "results"),
    "subject_scores": (subjects.fig_subject_scores, "results"),
    "fusion_weights": (fusion.fig_fusion_weights, "results"),
    "graph_embedding": (embedding.fig_graph_embedding, "results"),
    "psd_heatmap": (spectral.fig_psd_heatmap, "graphs"),
    "psd_lobes": (spectral.fig_psd_lobes, "graphs"),
    "connectivity_matrices": (connectivity.fig_connectivity_matrices, "graphs"),
    "connectivity_network": (connectivity.fig_connectivity_network, "graphs"),
    "psd_embedding": (embedding.fig_psd_embedding, "graphs"),
}


def make_figures(config: dict, out_dir: str | Path, results_root: str | Path | None = None, store=None,
                 stats_path: str | Path | None = None, style_path: str | Path | None = None,
                 synthetic: bool = False, only: list[str] | None = None, on_missing: str = "skip"):
    """Generate the figures listed in ``config['figures']``.

    on_missing: ``skip`` records figures whose inputs are absent (in the manifest) and
    continues; ``error`` raises. Results that a drawn figure asked for but that are not
    saved (yet) are listed in its metadata (``missing_inputs``) and in ``writer.incomplete``.
    Any other exception is a code error: the remaining figures are still written, then a
    RuntimeError lists the errors. Results from more than one graph run, or from a graph run
    other than ``store``, are refused (every figure must come from one graph run).
    Returns the FigureWriter (figures, skipped, incomplete).
    """
    style = load_style(style_path, config.get("style"))
    stats = load_stats(stats_path) if stats_path else None
    index = index_results(results_root, config.get("duplicates", "error")) if results_root else None
    if index is not None and len(index):
        runs = sorted(set(index["graph_run_id"]) - {""})
        if len(runs) > 1:
            raise ValueError(f"results under {results_root} come from {len(runs)} graph runs {runs}; "
                             "figures must use one graph run (pass a results root of one run)")
        if runs and store is not None:
            from ..graphs.store import graph_run_id

            if graph_run_id(store) != runs[0]:
                raise ValueError(f"results were computed on graph run {runs[0]}, but --graphs is "
                                 f"{graph_run_id(store)}; use the graph run the experiments used")
    writer = FigureWriter(out_dir, style, {"figures_config": config, "results_root": str(results_root),
                                           "graphs": str(getattr(store, "root", None)),
                                           "stats": str(stats_path) if stats_path else None}, synthetic)
    writer.skipped, writer.incomplete, errors = [], [], []
    for spec in config["figures"]:
        if only and spec["id"] not in only:
            continue
        fn, kind = FAMILIES[spec["family"]]
        source = index if kind == "results" else store
        reset_missing()
        try:
            if source is None:
                raise MissingInputs(f"figure needs {kind} input")
            with styled(style):
                fig, data, meta, sources = fn(source, spec, style, stats)
        except (MissingInputs, IncomparableTiming, InvalidStatistics) as exc:   # inputs absent or refused
            if on_missing == "error":
                raise
            writer.skipped.append({"id": spec["id"], "reason": f"{type(exc).__name__}: {exc}"})
            continue
        except Exception as exc:                       # noqa: BLE001 - a code error, never a silent skip
            if on_missing == "error":
                raise
            errors.append({"id": spec["id"], "error": f"{type(exc).__name__}: {exc}"})
            writer.skipped.append({"id": spec["id"], "reason": f"ERROR {type(exc).__name__}: {exc}"})
            continue
        missing = reset_missing()
        if missing:
            writer.incomplete.append({"id": spec["id"], "status": spec.get("status", "supplementary"),
                                      "missing_inputs": missing})
        writer.save(fig, spec["id"], data,
                    {"family": spec["family"], "status": spec.get("status", "supplementary"),
                     "title": spec.get("title", ""), "spec": spec, "missing_inputs": missing, **meta}, sources)
    writer.config["skipped"] = writer.skipped
    writer.config["incomplete"] = writer.incomplete
    writer.write_manifest()
    if errors:
        raise RuntimeError(f"{len(errors)} figure(s) failed with code errors (other figures were written to "
                           f"{writer.out_dir}): {errors}")
    return writer
