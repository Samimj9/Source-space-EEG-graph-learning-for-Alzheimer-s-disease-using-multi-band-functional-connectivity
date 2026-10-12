"""Figure registry and driver: config -> figures, each with data, metadata and a manifest.

Every family reads only saved files: experiment output directories (``results``) and/or
a graph run (``graphs``). Nothing is recomputed from raw EEG, and no model is trained.
"""

from __future__ import annotations

from pathlib import Path

from . import connectivity, embedding, fusion, performance, spectral, subjects
from .results import index_results
from .stats import load_stats
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
    continues; ``error`` raises. Returns the FigureWriter (figures + skipped list).
    """
    style = load_style(style_path, config.get("style"))
    stats = load_stats(stats_path) if stats_path else None
    index = index_results(results_root, config.get("duplicates", "error")) if results_root else None
    writer = FigureWriter(out_dir, style, {"figures_config": config, "results_root": str(results_root),
                                           "graphs": str(getattr(store, "root", None)),
                                           "stats": str(stats_path) if stats_path else None}, synthetic)
    writer.skipped = []
    for spec in config["figures"]:
        if only and spec["id"] not in only:
            continue
        fn, kind = FAMILIES[spec["family"]]
        source = index if kind == "results" else store
        try:
            if source is None:
                raise ValueError(f"figure needs {kind} input")
            with styled(style):
                fig, data, meta, sources = fn(source, spec, style, stats)
        except Exception as exc:                       # noqa: BLE001 - recorded, or re-raised
            if on_missing == "error":
                raise
            writer.skipped.append({"id": spec["id"], "reason": f"{type(exc).__name__}: {exc}"})
            continue
        writer.save(fig, spec["id"], data,
                    {"family": spec["family"], "status": spec.get("status", "supplementary"),
                     "title": spec.get("title", ""), "spec": spec, **meta}, sources)
    writer.config["skipped"] = writer.skipped
    writer.write_manifest()
    return writer
