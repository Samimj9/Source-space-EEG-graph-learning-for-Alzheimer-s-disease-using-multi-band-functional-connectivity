"""Retired from tests/test_incomplete_graph_run.py by the final framework (docs/final/inventory.md): they test D1/D2 or
earlier-suite functionality that is no longer part of the code base. Kept verbatim; not collected."""

def test_experiments_refuse_an_empty_graph_run(tmp_path):
    store = make_synthetic_store().subset(np.array([], dtype=int))
    v = [x for x in expand_variants(load_experiment(Path("configs/experiments/exp1_source_psd.yaml")))
         if x["model"]["name"] == "logreg"][0]
    with pytest.raises(IncompleteGraphRun, match="empty cohort"):
        runner.run_variant(store, v, "within_D1", tmp_path / "exp")
    assert not (tmp_path / "exp").exists()                         # nothing written


