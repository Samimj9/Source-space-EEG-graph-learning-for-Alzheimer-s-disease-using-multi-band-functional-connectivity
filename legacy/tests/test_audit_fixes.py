"""Retired from tests/test_audit_fixes.py by the final framework (docs/final/inventory.md): they test D1/D2 or
earlier-suite functionality that is no longer part of the code base. Kept verbatim; not collected."""

def test_recurring_outer_split_does_not_collide_with_cached_inner_trials(store, tmp_path):
    """D1 has only 45 x 45 possible 5-fold test folds, so splits recur across repeats; each
    recurrence is a different unit (different seed) and must not be mistaken for another."""
    v = _variant("exp1_source_psd.yaml", "logreg", **{"evaluation.n_repeats": 4, "evaluation.n_bootstrap": 20})
    data = build_dataset(store, v)
    meta = data.meta[data.meta["dataset"] == "D1"].drop_duplicates("key")
    _, _, splits = runner._splits(data.meta, v, "within_D1", "y")
    keys = [runner.split_id(sp.train_keys, sp.test_keys) for sp in splits]
    assert len(keys) > len(set(keys)), "fixture must contain a recurring outer split"
    assert len(meta) == 20
    runner.run_variant(store, v, "within_D1", tmp_path, data=data)   # used to raise IncompatibleCheckpoint
    again = runner.run_variant(store, v, "within_D1", tmp_path, data=data)
    assert np.isfinite(again["subject_level"]["all"]["roc_auc"]["mean"])


def test_dataset_probe_permutation_changes_labels(store):
    v = _variant("probe_dataset.yaml", "default")
    data = build_dataset(store, v)
    perm = runner.permute_subject_labels(data, np.random.default_rng(0))
    before = data.meta.drop_duplicates("key").set_index("key")[data.target]
    after = perm.meta.drop_duplicates("key").set_index("key")[data.target]
    assert (before != after.loc[before.index]).any()
    assert sorted(after) == sorted(before)                           # a permutation, same class sizes


def test_result_dir_separates_non_primary_analyses(tmp_path):
    v = _variant("exp6_multiband_psd_strength.yaml", "plv__full")
    assert runner.result_dir(tmp_path, v, "pooled") == tmp_path / "exp6_multiband_psd_strength" / "plv__full" / "pooled"
    probe = set_dotted(v, "target", "dataset")
    loso = set_dotted(v, "evaluation.scheme", "loso")
    assert runner.result_dir(tmp_path, probe, "pooled").parent.name == "plv__full__target-dataset"
    assert runner.result_dir(tmp_path, loso, "pooled").parent.name == "plv__full__scheme-loso"


def test_variant_filter_matches_whole_tokens():
    vs = expand_variants(load_experiment(EXP / "exp6_multiband_psd_strength.yaml"))
    plv = [v["variant"] for v in vs if variant_matches(v, ["plv"])]
    assert plv and all(n.startswith("plv__") for n in plv)
    assert [v["variant"] for v in vs if variant_matches(v, ["plv__full"])] == ["plv__full"]
    assert {v["variant"] for v in vs if variant_matches(v, ["full"])} == {"plv__full", "ciplv__full"}


def test_exp6_full_saves_embeddings_for_figure_s20():
    vs = {v["variant"]: v for v in expand_variants(load_experiment(EXP / "exp6_multiband_psd_strength.yaml"))}
    assert vs["plv__full"]["evaluation"]["save_embeddings"] and not vs["plv__no_psd"]["evaluation"]["save_embeddings"]


def test_missing_results_are_reported_not_silent(results, tmp_path):
    from adgraph.visualization.figures import make_figures

    store, res = results
    partial = tmp_path / "partial"
    shutil.copytree(res, partial)
    for d in partial.glob("exp5_multiband/*"):                      # Exp5 "still running"
        shutil.rmtree(d)
    w = make_figures(_paper(), tmp_path / "figs", partial, store, synthetic=True, only=["Fig3_performance_overall"])
    assert [f["id"] for f in w.figures] == ["Fig3_performance_overall"]
    inc = {f["id"]: f for f in w.incomplete}
    missing = inc["Fig3_performance_overall"]["missing_inputs"]
    assert {m["entry"] for m in missing} == {"Exp5 multi-band · attn"}
    meta = json.loads((tmp_path / "figs" / "Fig3_performance_overall_meta.json").read_text())
    assert meta["missing_inputs"] == missing


def test_code_errors_are_not_reported_as_missing_inputs(results, tmp_path, monkeypatch):
    from adgraph.visualization import figures

    store, res = results
    fn, kind = figures.FAMILIES["performance"]
    monkeypatch.setitem(figures.FAMILIES, "performance", (lambda *a: (_ for _ in ()).throw(IndexError("bug")), kind))
    with pytest.raises(RuntimeError, match="code errors.*IndexError"):
        figures.make_figures(_paper(), tmp_path / "f", res, store, synthetic=True,
                             only=["Fig3_performance_overall", "Fig2a_source_psd_relative"])
    assert any((tmp_path / "f").rglob("Fig2a*.png"))                 # other figures still written


def test_figures_refuse_results_from_another_graph_run(results, tmp_path):
    from adgraph.visualization.figures import make_figures

    store, res = results
    other = make_synthetic_store({"D1": (5, 5), "D2": (5, 6)}, windows_per_subject=(3, 3), seed=7)
    with pytest.raises(ValueError, match="graph run"):
        make_figures(_paper(), tmp_path / "f", res, other, synthetic=True, only=["Fig3_performance_overall"])


def test_index_excludes_non_primary_runs(results, tmp_path):
    from adgraph.visualization.results import index_results

    _, res = results
    root = tmp_path / "r"
    shutil.copytree(res, root)
    idx = index_results(root)
    d = Path(idx["path"].iloc[0])
    man = json.loads((d / "manifest.json").read_text())
    man["variant_config"]["target"] = "dataset"
    (d / "manifest.json").write_text(json.dumps(man))
    assert len(index_results(root)) == len(idx) - 1
    assert len(index_results(root, primary_only=False)) == len(idx)
    assert set(idx["graph_run_id"]) != {""}


