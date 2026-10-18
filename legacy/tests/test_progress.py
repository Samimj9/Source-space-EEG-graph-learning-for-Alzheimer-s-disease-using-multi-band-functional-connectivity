"""Retired from tests/test_progress.py by the final framework (docs/final/inventory.md): they test D1/D2 or
earlier-suite functionality that is no longer part of the code base. Kept verbatim; not collected."""

def test_experiment_progress_counts_recovered_folds(tmp_path, bars, fit_counter):
    from adgraph.experiments.plan import run_plan
    from adgraph.legacy.synthetic import make_synthetic_store

    made, depth = bars
    store = make_synthetic_store({"D1": (6, 6), "D2": (6, 7)}, windows_per_subject=(3, 3))
    entries = [{"config": _tiny_config(tmp_path, "exp1_source_psd.yaml"), "settings": ["within_D1"]}]
    total_units = 2 * (2 * 3 + 2)                            # 2 variants x (6 folds + 2 permutations)
    fit_counter.update(fits=0, crash_at=40)                  # disconnect part-way
    with pytest.raises(SimulatedCrash):
        run_plan(entries, store, tmp_path / "res", config_root="/")
    first = made[-1]
    done_before = first.n
    assert first.total == total_units and 0 < done_before < total_units

    made.clear()
    fit_counter.update(fits=0, crash_at=None)
    run_plan(entries, store, tmp_path / "res", config_root="/")
    bar = made[-1]
    assert bar.total == total_units and bar.initial == done_before      # recovered work counted as done
    assert bar.n == total_units and bar.counts["units done"] + bar.counts["resumed"] == 2
    assert {"variant", "setting", "repeat", "fold", "inner", "permutation"} <= bar.seen
    assert depth["max"] == 1

    made.clear()
    fit_counter["fits"] = 0
    run_plan(entries, store, tmp_path / "res", config_root="/")
    bar = made[-1]
    assert bar.initial == bar.total and fit_counter["fits"] == 0 and bar.counts["resumed"] == 2


def test_gnn_epochs_are_shown(tmp_path, bars):
    from adgraph.experiments.plan import run_plan
    from adgraph.legacy.synthetic import make_synthetic_store

    made, _ = bars
    store = make_synthetic_store({"D1": (5, 5), "D2": (5, 5)}, windows_per_subject=(3, 3))
    cfg = {"extends": str(REPO / "configs/experiments/exp4_band_mean.yaml"),
           "evaluation": {"n_repeats": 1, "k": 3, "inner_k": 2, "n_bootstrap": 20},
           "training": {"epochs": 2}, "hparams": {"model.hidden": [8, 16]}}
    p = tmp_path / "tiny_exp4.yaml"
    p.write_text(yaml.safe_dump(cfg))
    run_plan([{"config": str(p), "settings": ["within_D1"], "variants": ["plv"]}], store, tmp_path / "r",
             config_root="/")
    assert {"epoch", "inner", "fold"} <= made[-1].seen


def test_figure_progress(tmp_path, bars, synthetic_results):
    from adgraph.visualization.figures import make_figures

    made, _ = bars
    store, res = synthetic_results
    paper = yaml.safe_load((REPO / "configs/figures/paper.yaml").read_text())
    only = ["Fig3_performance_overall", "Fig2a_source_psd_relative", "S20_graph_embedding_pca"]
    make_figures(paper, tmp_path / "f", res, store, synthetic=True, only=only)
    bar = next(b for b in made if b.desc == "figures")
    assert bar.total == 3 and bar.n == 3
    assert bar.counts["made"] + bar.counts["skipped"] == 3 and "figure" in bar.seen


