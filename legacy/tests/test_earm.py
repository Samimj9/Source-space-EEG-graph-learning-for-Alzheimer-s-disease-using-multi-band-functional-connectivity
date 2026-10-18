"""Retired from tests/test_earm.py by the final framework (docs/final/inventory.md): they test D1/D2 or
earlier-suite functionality that is no longer part of the code base. Kept verbatim; not collected."""

def test_config_variants():
    vs = {v["variant"]: v for v in expand_variants(load_experiment(CFG))}
    assert set(vs) == {f"{c}__{n}" for c in ("plv", "ciplv") for n in ("full", "fixed_adjacency", "mean_fusion")}
    for v in vs.values():
        assert hparam_grid(v) == [{}] and v["evaluation"]["n_repeats"] == 10
        assert v["graph"]["band_mode"] == "multi" and v["graph"]["bands"] == ["theta", "alpha1", "alpha2", "beta"]
        assert v["node_features"]["components"] == ["psd", "strength"] and v["sparsify"]["method"] == "none"
    assert vs["plv__fixed_adjacency"]["model"]["adaptive_adjacency"] is False
    assert vs["plv__mean_fusion"]["model"]["fusion"] == "mean" and vs["plv__full"]["model"]["fusion"] == "attention"


def test_experiment_one_fit_per_fold_saves_state(tmp_path, monkeypatch):
    from adgraph.models import trainer

    store = make_synthetic_store({"D1": (5, 5), "D2": (5, 6)}, windows_per_subject=(3, 4), psd_effect=0.8,
                                 conn_effect=0.8, seed=1)
    fitted = []
    real_fit = trainer.GNNModel.fit
    monkeypatch.setattr(trainer.GNNModel, "fit", lambda self, view: (fitted.append(set(view.meta["key"])),
                                                                     real_fit(self, view))[1])
    v = expand_variants(load_experiment(CFG))[0]
    for k, val in {"evaluation.n_repeats": 1, "evaluation.k": 3, "evaluation.n_bootstrap": 50,
                   "training.epochs": 2}.items():
        v = set_dotted(v, k, val)
    run_variant(store, v, "within_D2", tmp_path)
    out = tmp_path / "exp7_earm_gnn" / "plv__full" / "within_D2"
    folds = pd.read_csv(out / "folds.csv")
    pred = pd.read_csv(out / "predictions_windows.csv")
    assert len(fitted) == 3
    for f, g in folds.groupby("fold"):
        train, test = set(g.loc[g.side == "train", "key"]), set(g.loc[g.side == "test", "key"])
        assert not train & test and fitted[f] == train and set(pred.query("fold == @f")["key"]) == test
    assert [c for c in pred if c.startswith("band_weight_")] == [f"band_weight_{b}" for b in
                                                                 ("theta", "alpha1", "alpha2", "beta")]
    states = sorted((out / "units").glob("fold_*.model.npz"))
    assert len(states) == 3
    s = np.load(states[0])
    assert s["edge_gates"].shape == (4, 68, 68) and len(s["history_train_loss"]) == 2
    assert json.loads((out / "manifest.json").read_text())["variant_config"]["model"]["architecture"] == "earm_gnn"


def test_comparison_and_dataset_diagnostic_from_saved_outputs(tmp_path):
    from adgraph.evaluation.dataset_shift import diagnose
    from adgraph.experiments.compare import write_comparison

    store = make_synthetic_store({"D1": (4, 4), "D2": (4, 5)}, windows_per_subject=(3, 3), psd_effect=0.8,
                                 conn_effect=0.6, seed=2)
    fast = {"evaluation.n_repeats": 2, "evaluation.k": 3, "evaluation.inner_k": 2, "evaluation.n_bootstrap": 50,
            "training.epochs": 1}

    def run(cfg, keep):
        for v in expand_variants(load_experiment(cfg)):
            if keep(v):
                for k, val in fast.items():
                    v = set_dotted(v, k, val)
                for s in ("within_D1", "within_D2"):
                    run_variant(store, v, s, tmp_path / "results")

    run(Path("configs/experiments/exp1_source_psd.yaml"), lambda v: v["model"]["name"] == "svm")
    run(CFG, lambda v: v["graph"]["connectivity"] == "plv" and v["variant"] != "plv__mean_fusion")
    res = write_comparison(tmp_path / "results", tmp_path / "cmp")
    t = res["table"]
    assert {"EARM-GNN full", "EARM-GNN fixed adjacency (abl. A)", "Exp1 source PSD SVM"} == set(t["model"])
    d = res["differences"]
    assert set(d["comparator"]) == {"EARM-GNN fixed adjacency (abl. A)", "Exp1 source PSD SVM"}
    assert (d["n_shared_repeats"] == 2).all()
    assert any(f.startswith("E7_roc_within_D1_plv") for f in res["figures"])
    diag = diagnose(store, tmp_path / "results", tmp_path / "diag")
    sep = diag["separability"]
    assert {"auc_D1", "auc_D2", "smd_D1_minus_D2_HC"} <= set(sep) and len(sep) == 4 * 4
    assert set(diag["predictions"]["setting"]) == {"within_D1", "within_D2"}
    assert (tmp_path / "diag" / "report.md").exists()


