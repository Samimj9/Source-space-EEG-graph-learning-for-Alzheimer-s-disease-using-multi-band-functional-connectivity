"""Retired from tests/test_checkpoint.py by the final framework (docs/final/inventory.md): they test D1/D2 or
earlier-suite functionality that is no longer part of the code base. Kept verbatim; not collected."""

def test_experiment_resumes_from_unfinished_folds(small_store, tmp_path, fit_counter):
    v = lr_variant()
    full = runner.run_variant(small_store, v, "within_D1", tmp_path / "ref")    # uninterrupted reference
    total = fit_counter["fits"]               # 2 repeats x 3 folds x (2 C x 2 inner + 1 outer) = 30
    assert total == 30
    fit_counter.update(fits=0, crash_at=13)
    with pytest.raises(SimulatedCrash):
        runner.run_variant(small_store, v, "within_D1", tmp_path / "run")
    st = runner.experiment_status(small_store, v, "within_D1", tmp_path / "run")
    # classical inner trials are not cached (a fit is cheaper than a Drive write); outer folds are
    assert st["state"] == "partial" and st["folds_completed"] == 2 and st["inner_trials_cached"] == 0
    fit_counter.update(fits=0, crash_at=None)
    resumed = runner.run_variant(small_store, v, "within_D1", tmp_path / "run")
    assert fit_counter["fits"] == total - 10                       # the 2 completed folds (5 fits each) reused
    a = pd.read_csv(tmp_path / "ref/exp1_source_psd/logreg/within_D1/predictions_windows.csv")
    b = pd.read_csv(tmp_path / "run/exp1_source_psd/logreg/within_D1/predictions_windows.csv")
    assert np.allclose(a["score"], b["score"])
    assert json.dumps(resumed["subject_level"], sort_keys=True) == json.dumps(full["subject_level"], sort_keys=True)
    fit_counter["fits"] = 0
    runner.run_variant(small_store, v, "within_D1", tmp_path / "run")
    assert fit_counter["fits"] == 0                                  # complete: nothing retrained
    assert runner.experiment_status(small_store, v, "within_D1", tmp_path / "run")["state"] == "completed"


def test_more_repeats_reuse_existing_folds(small_store, tmp_path, fit_counter):
    runner.run_variant(small_store, lr_variant(2), "within_D2", tmp_path)
    fit_counter["fits"] = 0
    s3 = runner.run_variant(small_store, lr_variant(3), "within_D2", tmp_path)
    assert fit_counter["fits"] == 15                                 # only the new repeat (3 folds x 5 fits)
    assert s3["n_repeats"] == 3 and (tmp_path / "_superseded").exists()


def test_incompatible_results_are_never_reused(small_store, tmp_path, fit_counter):
    v = lr_variant()
    runner.run_variant(small_store, v, "within_D1", tmp_path)
    changed = set_dotted(v, "hparams", {"model.params.C": [0.01, 10.0]})
    with pytest.raises(IncompatibleCheckpoint, match="different configuration"):
        runner.run_variant(small_store, changed, "within_D1", tmp_path)
    other_graphs = make_synthetic_store({"D1": (5, 5), "D2": (5, 6)}, windows_per_subject=(3, 4), seed=7)
    with pytest.raises(IncompatibleCheckpoint, match="graph_run_id"):
        runner.run_variant(other_graphs, v, "within_D1", tmp_path)
    assert runner.experiment_status(other_graphs, v, "within_D1", tmp_path)["state"] == "incompatible"
    fit_counter["fits"] = 0
    runner.run_variant(small_store, changed, "within_D1", tmp_path, force=True)
    assert fit_counter["fits"] == 30 and list((tmp_path / "_superseded").rglob("final.done.json"))
    from adgraph.visualization.results import index_results

    assert len(index_results(tmp_path)) == 1                         # superseded results never indexed


def test_corrupted_fold_unit_is_recomputed_alone(small_store, tmp_path, fit_counter):
    v = lr_variant()
    runner.run_variant(small_store, v, "within_D1", tmp_path)
    out = tmp_path / "exp1_source_psd/logreg/within_D1"
    p = out / "units/fold_r001_f02.pred.csv"
    p.write_text(p.read_text().replace("0.", "9.", 1))
    (out / "final.done.json").unlink()                               # e.g. crash before the final marker
    fit_counter["fits"] = 0
    runner.run_variant(small_store, v, "within_D1", tmp_path)
    assert fit_counter["fits"] == 5                                  # that fold only (classical: inner search redone)


def test_permutations_resume(small_store, tmp_path, fit_counter):
    v = set_dotted(lr_variant(1), "evaluation.permutation", {"n": 3, "n_repeats": 1})
    v["hparams"] = {}
    fit_counter.update(fits=0, crash_at=3 + 3 + 2)                   # 3 outer fits, perm 0, crash in perm 1
    with pytest.raises(SimulatedCrash):
        runner.run_variant(small_store, v, "pooled", tmp_path)
    fit_counter.update(fits=0, crash_at=None)
    s = runner.run_variant(small_store, v, "pooled", tmp_path)
    assert fit_counter["fits"] == 6 and s["permutation"]["n"] == 3   # perms 1 and 2 only


def test_gnn_inner_trials_and_folds_resume(small_store, tmp_path, fit_counter):
    pytest.importorskip("torch")
    v = expand_variants(load_experiment(EXP / "exp6_multiband_psd_strength.yaml"))[0]
    for k, val in {"evaluation.n_repeats": 1, "evaluation.k": 3, "evaluation.inner_k": 2, "training.epochs": 1,
                   "model.hidden": 8, "evaluation.save_embeddings": True}.items():
        v = set_dotted(v, k, val)
    v["hparams"] = {"model.hidden": [8, 16]}
    fit_counter.update(fits=0, crash_at=8)
    with pytest.raises(SimulatedCrash):
        runner.run_variant(small_store, v, "within_D2", tmp_path)
    fit_counter.update(fits=0, crash_at=None)
    runner.run_variant(small_store, v, "within_D2", tmp_path)
    assert fit_counter["fits"] == 15 - 7
    z = np.load(tmp_path / v["experiment"] / v["variant"] / "within_D2" / "embeddings.npz")
    assert {f"r000_f{f:02d}__emb" for f in range(3)} <= set(z.files)   # reused + new folds, one model each


def test_figures_never_trigger_training(small_store, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from adgraph.visualization.figures import make_figures

    v = lr_variant(1)
    for s in ("within_D1", "within_D2"):
        runner.run_variant(small_store, v, s, tmp_path / "res")
    for name in ("make_model", "cross_validate", "run_variant", "select_hparams"):
        monkeypatch.setattr(runner, name, lambda *a, **k: pytest.fail("figures must not train"))
    cfg = {"figures": [{"id": "perf", "family": "performance", "settings": ["within_D1", "within_D2"],
                        "metrics": ["roc_auc"], "entries": [{"label": "LR", "experiment": "exp1_source_psd",
                                                             "where": {"model.name": "logreg"}}]}]}
    w = make_figures(cfg, tmp_path / "figs", tmp_path / "res", synthetic=True, on_missing="error")
    assert [f["id"] for f in w.figures] == ["perf"]


def test_cli_resume_and_status(tmp_path):
    base = [sys.executable, "scripts/run_experiment.py", "--config", str(EXP / "exp0_sensor_psd.yaml"),
            "--synthetic", "--out", str(tmp_path), "--settings", "within_D1", "--variants", "logreg",
            "--override", "evaluation.n_repeats=1", "--override", "evaluation.k=3", "--override", "hparams={}"]
    st = subprocess.run(base + ["--status"], capture_output=True, text=True, check=True).stdout
    assert "not started" in st
    subprocess.run(base, capture_output=True, text=True, check=True)
    st = subprocess.run(base + ["--status"], capture_output=True, text=True, check=True).stdout
    assert "completed" in st
    again = subprocess.run(base, capture_output=True, text=True, check=True)
    assert "nothing recomputed" in again.stderr
    bad = subprocess.run(base[:-1] + ["hparams={model.params.C: [5.0]}"], capture_output=True, text=True)
    assert bad.returncode != 0 and "IncompatibleCheckpoint" in bad.stderr


