"""Synthetic inputs for testing/demonstrating the figure framework (never real results).

Builds a synthetic graph run and runs the *real* experiment runner on it with a tiny
protocol, so figures are exercised on the exact file formats they will read later.
All outputs are synthetic; figures made from them are watermarked by the writer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..experiments.config import expand_variants, load_experiment, set_dotted
from ..experiments.runner import run_variant
from ..graphs.dataset import build_dataset
from ..synthetic import make_synthetic_store

EXP_DIR = Path(__file__).resolve().parents[3] / "configs" / "experiments"
ALL = ["within_D1", "within_D2", "pooled", "cross_D1_to_D2", "cross_D2_to_D1"]
WITHIN = ["within_D1", "within_D2"]
FAST = {"evaluation.n_repeats": 2, "evaluation.k": 3, "evaluation.inner_k": 2, "evaluation.n_bootstrap": 50,
        "evaluation.cross_n_seeds": 2, "training.epochs": 2, "model.hidden": 8, "model.mlp_hidden": 8}


def _fast(v: dict, **extra) -> dict:
    for k, val in {**FAST, **extra}.items():
        if k.startswith("evaluation") or k.split(".")[0] in v:
            v = set_dotted(v, k, val)
    v["hparams"] = {}
    return v


def plan() -> list[tuple[str, dict, list[str], dict]]:
    """(config, where-filter on variant config, settings, extra overrides)."""
    full6 = {"node_features.components": ["psd", "strength"], "model.fusion": "attention",
             "graph.bands": ["theta", "alpha1", "alpha2", "beta"]}
    return [
        ("exp0_sensor_psd.yaml", {}, ALL, {}),
        ("exp1_source_psd.yaml", {}, ALL, {}),
        ("exp2_conn_minimal_fixedhp.yaml", {"model.encoder": "gcn"}, WITHIN, {}),
        ("exp2_conn_minimal_fixedhp.yaml", {"model.encoder": "gcn", "graph.bands": ["alpha1"]},
         ["pooled", "cross_D1_to_D2", "cross_D2_to_D1"], {}),
        ("exp3_conn_psd_fixedhp.yaml", {"model.encoder": "gcn"}, WITHIN, {}),
        ("exp3_conn_psd_fixedhp.yaml", {"model.encoder": "gcn", "graph.bands": ["alpha1"]},
         ["pooled", "cross_D1_to_D2", "cross_D2_to_D1"], {}),
        ("exp4_band_mean_fixedhp.yaml", {}, ALL, {}),
        ("exp5_multiband_fixedhp.yaml", {"model.shared_encoder": True, "model.fusion": "attention"}, ALL, {}),
        ("exp6_multiband_psd_strength_fixedhp.yaml", {}, WITHIN, {}),
        ("exp6_multiband_psd_strength_fixedhp.yaml", full6, ["pooled", "cross_D1_to_D2", "cross_D2_to_D1"],
         {"evaluation.save_embeddings": True}),
        ("baseline_age.yaml", {}, ["within_D1", "within_D2", "pooled"], {}),
        ("age/exp1_source_psd_plus_age.yaml", {"model.name": "svm"}, ["within_D1", "within_D2", "pooled"], {}),
        ("age/exp6_multiband_psd_strength_fixedhp_plus_age.yaml", full6, ["within_D1", "within_D2", "pooled"], {}),
    ]


def _match(v, where):
    from ..visualization.results import matches

    return matches(v, where)


def build_synthetic_results(root: str | Path, cohort=None, seed: int = 0):
    """Write a synthetic graph run to ``root/graphs`` and experiment outputs to ``root/results``.

    Returns (GraphStore loaded from disk, results root)."""
    from ..graphs.store import GraphStore

    root = Path(root)
    store = make_synthetic_store(cohort or {"D1": (6, 6), "D2": (6, 7)}, windows_per_subject=(3, 4),
                                 psd_effect=0.6, conn_effect=0.6, dataset_shift=0.3, seed=seed)
    store.save(root / "graphs")
    store = GraphStore.load(root / "graphs")
    results = root / "results"
    for cfg, where, settings, extra in plan():
        for v in expand_variants(load_experiment(EXP_DIR / cfg)):
            if not _match(v, where):
                continue
            v = _fast(v, **extra)
            data = build_dataset(store, v)
            for s in settings:
                run_variant(store, v, s, results, data=data)
    return store, results


def synthetic_stats(path: str | Path, comparisons: list[str]) -> Path:
    """A *synthetic* statistics file in the required format (for testing the annotation path only)."""
    p = np.linspace(0.001, 0.2, len(comparisons))
    pd.DataFrame({"comparison": comparisons, "metric": "synthetic", "test": "synthetic-test", "unit": "subject",
                  "n_subjects": 25, "p_value": p, "p_corrected": np.minimum(1, p * len(p)),
                  "correction_method": "holm", "family": "synthetic"}).to_csv(path, index=False)
    return Path(path)
