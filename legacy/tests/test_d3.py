"""Retired from tests/test_d3.py by the final framework (docs/final/inventory.md): they test D1/D2 or
earlier-suite functionality that is no longer part of the code base. Kept verbatim; not collected."""

def test_d3_settings_and_configs(tmp_path):
    from adgraph.experiments.config import expand_variants, load_experiment
    from adgraph.experiments.runner import run_variant
    from adgraph.experiments.config import set_dotted
    from adgraph.legacy.synthetic import make_synthetic_store

    v = expand_variants(load_experiment(REPO / "configs/experiments/d3/d3_exp1_source_psd.yaml"))[1]
    bad = dict(v, settings=["within_D3", "within_D1"])
    with pytest.raises(ValueError, match="never mixed"):
        from adgraph.experiments.config import validate
        validate(bad)
    store = make_synthetic_store({"D3": (5, 5)}, windows_per_subject=(3, 3), psd_effect=0.8, seed=3)
    for k, val in {"evaluation.n_repeats": 2, "evaluation.k": 3, "evaluation.inner_k": 2,
                   "evaluation.n_bootstrap": 20}.items():
        v = set_dotted(v, k, val)
    s = run_variant(store, v, "within_D3", tmp_path)
    assert s["n_subjects"] == 10 and "roc_auc" in s["subject_level"]["all"]


def test_experiment_cli_uses_d3_family(tmp_path):
    from adgraph.legacy.synthetic import make_synthetic_store

    processed = tmp_path / "processed"
    for name, cohort in (("graphs_v4_synth", {"D1": (4, 4), "D2": (4, 5)}), ("graphsD3_v4_synth", {"D3": (5, 5)})):
        g = processed / name
        make_synthetic_store(cohort, windows_per_subject=(3, 3), psd_effect=0.6).save(g)
        (g / "assembly.done.json").write_text("{}")
        man = json.loads((g / "manifest.json").read_text())
        man["assembled_at"] = "2026-01-01T00:00:00+00:00"
        (g / "manifest.json").write_text(json.dumps(man))
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump({"drive_root": str(tmp_path), "processed_dir": str(processed),
                                                         "cache_dir": str(processed / "cache")}))
    cfg = tmp_path / "age.yaml"
    cfg.write_text(yaml.safe_dump({"extends": str(REPO / "configs/experiments/d3/d3_baseline_age.yaml"),
                                   "evaluation": {"n_repeats": 1, "k": 3, "inner_k": 2, "n_bootstrap": 20}}))
    r = subprocess.run([sys.executable, "scripts/experiment.py", "run", "--config", str(cfg), "--paths",
                        str(tmp_path / "paths.yaml"), "--allow-ephemeral-storage"], capture_output=True, text=True,
                       cwd=REPO)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2000:]
    assert "graphsD3_v4_synth" in r.stdout
    assert list((processed / "experiments_D3" / "d3_baseline_age").rglob("final.done.json"))
    assert not (processed / "experiments").exists()


@pytest.mark.parametrize("name, ref, n_variants", [
    ("d3_exp2_conn_minimal_fixedhp", "exp2_conn_minimal_fixedhp", 8),
    ("d3_exp3_conn_psd_fixedhp", "exp3_conn_psd_fixedhp", 8),
    ("d3_exp4_band_mean_fixedhp", "exp4_band_mean_fixedhp", 2),
    ("d3_exp5_multiband_fixedhp", "exp5_multiband_fixedhp", 4),
])
def test_d3_gnn_experiments_reuse_d1d2_definitions_and_run_on_d3_only(tmp_path, name, ref, n_variants):
    """D3 Exp 2-5 (methodology log entries 48-49): the Tier 1 D1/D2 variants unchanged (graph, node features,
    model, training, evaluation, protocol), within_D3 only, results only in experiments_D3/<experiment>/."""
    from adgraph.experiments.config import expand_variants, load_experiment, validate
    from adgraph.legacy.synthetic import make_synthetic_store

    d3 = expand_variants(load_experiment(REPO / f"configs/experiments/d3/{name}.yaml"))
    base = expand_variants(load_experiment(REPO / f"configs/experiments/{ref}.yaml"))
    assert [v["variant"] for v in d3] == [v["variant"] for v in base] and len(d3) == n_variants
    for a, b in zip(d3, base):
        validate(a)
        assert a["experiment"] == name and a["graph_family"] == "D3" and a["settings"] == ["within_D3"]
        for k in ("graph", "node_features", "model", "training", "evaluation", "protocol", "input"):
            assert a[k] == b[k], k
        assert a["protocol"]["name"] == "gnn_fixed_hparams_v1" and a["evaluation"]["n_repeats"] == 10

    processed = tmp_path / "processed"
    for gname, cohort in (("graphs_v4_synth", {"D1": (4, 4), "D2": (4, 5)}), ("graphsD3_v4_synth", {"D3": (5, 5)})):
        g = processed / gname
        make_synthetic_store(cohort, windows_per_subject=(3, 3), psd_effect=0.6).save(g)
        (g / "assembly.done.json").write_text("{}")
        man = json.loads((g / "manifest.json").read_text())
        man["assembled_at"] = "2026-01-01T00:00:00+00:00"
        (g / "manifest.json").write_text(json.dumps(man))
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump({"drive_root": str(tmp_path), "processed_dir": str(processed),
                                                         "cache_dir": str(processed / "cache")}))
    cfg = tmp_path / "small.yaml"
    cfg.write_text(yaml.safe_dump({"extends": str(REPO / f"configs/experiments/d3/{name}.yaml"),
                                   "evaluation": {"n_repeats": 1, "k": 3, "n_bootstrap": 20},
                                   "training": {"epochs": 2}}))
    r = subprocess.run([sys.executable, "scripts/experiment.py", "run", "--config", str(cfg), "--paths",
                        str(tmp_path / "paths.yaml"), "--allow-ephemeral-storage"], capture_output=True, text=True,
                       cwd=REPO)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2000:]
    assert "graphsD3_v4_synth" in r.stdout
    out = processed / "experiments_D3" / name
    done = sorted(p.parent.parent.name for p in out.rglob("within_D3/final.done.json"))
    assert done == sorted(v["variant"] for v in d3)
    assert not (processed / "experiments").exists()                      # never the D1/D2 results root
    assert sorted(p.name for p in (processed / "experiments_D3").iterdir()) == [name]   # no other D3 experiment
    for f in out.rglob("within_D3/predictions_subjects.csv"):
        subj = pd.read_csv(f)
        assert subj["key"].str.startswith("D3-").all() and subj["key"].nunique() == 10     # participant-level


