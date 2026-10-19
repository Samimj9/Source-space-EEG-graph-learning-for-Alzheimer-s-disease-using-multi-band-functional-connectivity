"""The final framework end to end on synthetic stores at the release paths (docs/final/framework.md): configurations,
store verification and lock, the QC gate, runs of classical and GNN variants in both settings with every required
output and metric, resume, reporting, the development-only selection rule, the run plan, the notebooks and the
single release pin. No EEG; synthetic results are never results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from adgraph import release
from adgraph.experiments.config import (FIXED_EPOCHS, experiment_configs, expand_variants, graph_constructions,
                                        load_experiment, validate, with_context)
from adgraph.experiments.plan import EmptySchedule, experiment_variants, run_experiment, status_table
from adgraph.stores import StoreError, open_store, verify_store

REPO = Path(__file__).resolve().parents[1]
N_VARIANTS = {"exp0": 4, "exp1": 4, "exp2": 10, "exp3": 10, "exp4": 4, "exp5": 8, "exp6": 10, "exp7": 6, "exp8": 10}


# ------------------------------------------------------------------------------------------- configurations
def test_one_config_per_experiment_exp0_to_exp8():
    cfgs = experiment_configs()
    assert sorted(cfgs) == [f"exp{i}" for i in range(9)]
    for exp, path in cfgs.items():
        vs = expand_variants(load_experiment(path))
        assert len(vs) == N_VARIANTS[exp], exp
        for v in vs:
            assert v["settings"] == ["xu_dev_cv", "xu_holdout"] and v["target"] == "label"
            if v["input"] == "graph":
                assert v["training"]["epochs"] == FIXED_EPOCHS == 60 and not v.get("hparams")
                assert not (v["training"].get("early_stopping") or {}).get("enabled")
                assert v["protocol"]["name"] == "gnn_fixed60_v1"
            else:
                assert v["hparams"] == {"model.params.C": [0.001, 0.01, 0.1, 1.0, 10.0]}


def test_gnn_training_blocks_are_identical_across_experiments():
    blocks = {json.dumps(v["training"], sort_keys=True) for p in experiment_configs().values()
              for v in expand_variants(load_experiment(p)) if v["input"] == "graph"}
    assert blocks == {json.dumps({"lr": 0.001, "weight_decay": 0.0001, "epochs": 60, "batch_size": 32},
                                 sort_keys=True)}


def test_protocol_violations_are_refused():
    v = expand_variants(load_experiment(experiment_configs()["exp3"]))[0]
    for bad in ({"training": {**v["training"], "epochs": 61}},
                {"training": {**v["training"], "early_stopping": {"enabled": True}}},
                {"hparams": {"model.hidden": [16, 32]}}, {"settings": ["within_D1"]}, {"target": "age"},
                {"graph": {**v["graph"], "connectivity": "ciplv_signed"}}):
        with pytest.raises(ValueError):
            validate({**v, **bad})


def test_run_context():
    v = expand_variants(load_experiment(experiment_configs()["exp5"]))[0]
    c = with_context(v, "wmne", "A")
    assert c["context"] == {"inverse": "wmne", "mode": "A", "atlas": "dk68", "graph": "dense"}
    assert with_context(v, "lcmv", "B", graph="knn_union_f20")["graph_construction"]["method"] == "knn"
    s = with_context(expand_variants(load_experiment(experiment_configs()["exp0"]))[0], "wmne", "B")
    assert s["context"]["inverse"] == "sensor" and s["graph_construction"] is None
    for bad in (dict(inverse="sloreta", mode="B"), dict(inverse="lcmv", mode="C"),
                dict(inverse="lcmv", mode="B", graph="no_such_graph")):
        with pytest.raises(ValueError):
            with_context(v, **bad)
    with pytest.raises(EmptySchedule):
        experiment_variants(experiment_configs()["exp5"], "lcmv", "B", variants=["no_such_variant"])


def test_feature_similarity_construction_sets_the_adjacency():
    v = expand_variants(load_experiment(experiment_configs()["exp5"]))[0]
    c = with_context(v, "lcmv", "B", graph="feature_similarity_knn_union_f20")
    assert c["graph"]["adjacency"] == "feature_similarity" and "adjacency" not in c["graph_construction"]
    assert set(graph_constructions()) >= {"dense", "knn_union_f20", "knn_mutual_f20"}


# ------------------------------------------------------------------------------------------- stores
def test_synthetic_release_stores_verify_and_lock(release_root):
    for inv in ("lcmv", "wmne"):
        for mode in ("B", "A"):
            r = verify_store(release_root, inv, mode)
            assert r["passed"], {k: c for k, c in r["checks"].items() if not c["passed"]}
            open_store(release_root, inv, mode)
    lock = json.loads((release_root / release.LOCK_FILE).read_text())
    assert set(lock["stores"]) == {"lcmv/B/dk68", "lcmv/A/dk68", "wmne/B/dk68", "wmne/A/dk68"}


def test_a_changed_or_unmatched_store_is_refused(release_root):
    from adgraph.stores import store_dir

    open_store(release_root, "lcmv", "B")                                  # locked
    d = store_dir(release_root, "lcmv", "B")
    man = json.loads((d / "manifest.json").read_text())
    man["graph_run_id"] = "something-else"
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(StoreError, match="validation.json|locked"):
        open_store(release_root, "lcmv", "B")
    w = store_dir(release_root, "wmne", "A")
    (w / "match.json").unlink()
    assert not verify_store(release_root, "wmne", "A")["passed"]


# ------------------------------------------------------------------------------------------- runs
REQUIRED = ("predictions_windows.csv", "predictions_participants.csv", "metrics.json", "fold_metrics.csv",
            "folds.csv", "training.csv", "fold_qc.csv", "manifest.json", "final.done.json")


def _short(monkeypatch, epochs=2):
    """GNNs train for 2 epochs in these tests: the configs keep the protocol's 60 (enforced by validate) and only
    the trainer is shortened here."""
    from adgraph.models import trainer

    orig = trainer.GNNModel.__init__

    def init(self, model, training, *a, **k):
        orig(self, model, {**training, "epochs": epochs}, *a, **k)
    monkeypatch.setattr(trainer.GNNModel, "__init__", init)


@pytest.mark.parametrize("exp,variant", [("exp1", "logreg__psd"), ("exp5", "plv__separate__mean")])
def test_end_to_end_run_outputs_and_resume(release_root, monkeypatch, exp, variant):
    _short(monkeypatch)
    cfg = experiment_configs()[exp]
    t = run_experiment(cfg, release_root, "lcmv", "B", variants=[variant])
    assert (t["state"] == "completed").all() and set(t["setting"]) == {"xu_dev_cv", "xu_holdout"}
    root = release.results_dir(release_root) / "lcmv" / "mode_B" / "dk68" / ("none" if exp == "exp1" else "dense")
    name = load_experiment(cfg)["experiment"]
    for setting in ("xu_dev_cv", "xu_holdout"):
        d = root / name / variant / setting
        assert all((d / f).exists() for f in REQUIRED), [f for f in REQUIRED if not (d / f).exists()]
        m = json.loads((d / "metrics.json").read_text())
        for k in ("roc_auc", "balanced_accuracy", "sensitivity", "specificity"):
            assert k in m["participant"]
        assert set(m["confusion"]) == {"tn", "fp", "fn", "tp"} and sum(m["confusion"].values()) == m["n_participants"]
        assert "ci_low" in m["participant_auc_delong"] and "roc_auc" in m["participant_bootstrap_ci95"]
        subj = pd.read_csv(d / "predictions_participants.csv")
        win = pd.read_csv(d / "predictions_windows.csv")
        assert np.allclose(subj.sort_values("key")["score"],
                           win.groupby("key")["score"].mean().sort_index().to_numpy())   # mean of window scores
        assert (subj["label"] == np.where(subj["y"] == 1, "AD", "HC")).all()
        if setting == "xu_dev_cv":
            assert m["n_participants"] == 52 and m["folds"]["n_folds"] == 5 and "mean_auc" in m["folds"]
            folds = pd.read_csv(d / "folds.csv")
            for f, g in folds.groupby("fold"):
                assert not set(g[g.side == "train"]["key"]) & set(g[g.side == "test"]["key"])
            assert len(pd.read_csv(d / "fold_qc.csv"))                       # fold QC recorded, nobody removed
        else:
            assert m["n_participants"] == 13 and (m["n_ad"], m["n_hc"]) == (7, 6)
        man = json.loads((d / "manifest.json").read_text())
        assert man["release"]["release"] == release.RELEASE and man["holdout_policy"] == release.HOLDOUT_POLICY
        tr = pd.read_csv(d / "training.csv")
        if exp == "exp5":
            assert (tr["epochs"] == 2).all() and tr["history_train_loss"].map(lambda s: len(json.loads(s))).eq(2).all()
    again = run_experiment(cfg, release_root, "lcmv", "B", variants=[variant])
    assert (again["state"] == "completed").all()
    st = status_table(cfg, release_root, "lcmv", "B")
    assert set(st.loc[st["variant"] == variant, "state"]) == {"completed"}


def test_experiments_refuse_unresolved_category_a(release_root, monkeypatch):
    from adgraph.qc.report import QCUnresolved
    from adgraph.stores import store_dir

    d = store_dir(release_root, "wmne", "B")
    with np.load(d / "graphs.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["psd_log"][0, 0, 0] = np.nan
    np.savez(d / "graphs.npz", **arrays)
    with pytest.raises(QCUnresolved):
        from adgraph.experiments.plan import open_context

        open_context(release_root, "wmne", "B")


# ------------------------------------------------------------------------------------------- reporting, selection
def test_report_pairs_lcmv_and_wmne(release_root, monkeypatch):
    from adgraph.experiments.report import collect, compare_inverses, development_only

    cfg = experiment_configs()["exp1"]
    for inv in ("lcmv", "wmne"):
        run_experiment(cfg, release_root, inv, "B", variants=["logreg__psd"])
    t = collect(release.results_dir(release_root))
    assert len(t) == 4 and set(development_only(t)["setting"]) == {"xu_dev_cv"}
    pairs = compare_inverses(t)
    assert len(pairs) == 2 and pairs["same_participants"].all() and pairs["delong_p"].notna().all()


def _table(aucs: dict, holdout: dict | None = None) -> pd.DataFrame:
    rows = []
    for (graph, exp, var, inv), auc in aucs.items():
        rows.append({"inverse": inv, "mode": "B", "atlas": "dk68", "graph": graph, "experiment": exp,
                     "variant": var, "setting": "xu_dev_cv", "auc": auc})
    for (graph, exp, var, inv), auc in (holdout or {}).items():
        rows.append({"inverse": inv, "mode": "B", "atlas": "dk68", "graph": graph, "experiment": exp,
                     "variant": var, "setting": "xu_holdout", "auc": auc})
    return pd.DataFrame(rows)


CELLS = [("exp4_band_mean", "plv__psd"), ("exp5_multibranch", "plv__separate__mean")]


def _grid(base=0.70, better=None, delta=0.0):
    from adgraph.experiments.selection import rules

    r = rules()["graph_construction"]
    out = {}
    for g in [r["default"]] + r["candidates"] + r["reported_only"]:
        for e, v in CELLS:
            for inv in ("lcmv", "wmne"):
                out[(g, e, v, inv)] = base + (delta if g == better else 0.0)
    return out


def test_selection_keeps_dense_unless_clearly_better_on_development_cv():
    from adgraph.experiments.selection import select

    assert select(_table(_grid()), atlas="dk68")["selected"] == "dense"            # ties: the default
    assert select(_table(_grid(better="knn_union_f20", delta=0.01)), atlas="dk68")["selected"] == "dense"
    d = select(_table(_grid(better="knn_union_f20", delta=0.03)), atlas="dk68")
    assert d["selected"] == "knn_union_f20" and d["source"].startswith("development CV")
    assert select(_table(_grid(better="feature_similarity_knn_union_f20", delta=0.2)), atlas="dk68")["selected"] \
        == "dense"
    g = _grid()                                                                    # holdout rows are never read
    hold = {k: (0.99 if k[0] == "knn_mutual_f10" else 0.5) for k in g}
    assert select(_table(g, hold), atlas="dk68")["selected"] == "dense"
    inc = _grid(better="knn_union_f20", delta=0.1)
    inc.pop(("knn_mutual_f30", "exp5_multibranch", "plv__separate__mean", "wmne"))
    d = select(_table(inc), atlas="dk68")
    assert d["selected"] == "dense" and d["reason"].startswith("comparison incomplete")


def test_selection_needs_consistency_across_cells():
    from adgraph.experiments.selection import select

    g = _grid()
    k = ("knn_union_f10", "exp4_band_mean", "plv__psd", "lcmv")
    g[k] = 0.95                                                                    # one cell much better, others equal
    assert select(_table(g), atlas="dk68")["selected"] == "dense"


# ------------------------------------------------------------------------------------------- plan, notebooks, pin
def test_run_plan_covers_the_final_architecture():
    plan = yaml.safe_load((REPO / "configs" / "final" / "run_plan.yaml").read_text())
    steps = plan["steps"]
    exps = [s["experiment"] for s in steps if "experiment" in s]
    assert exps == [f"exp{i}" for i in range(9)] == sorted(experiment_configs())
    names = [s["notebook"] for s in steps]
    assert names[:5] == ["00_setup_and_stores.ipynb", "01_participant_qc.ipynb", "02_parcellation.ipynb",
                         "04_roi_68_vs_18.ipynb", "03_graph_construction.ipynb"]       # atlas first, then graphs
    assert names[-1] == "99_results_report.ipynb"
    assert all((REPO / "notebooks" / n).exists() for n in names)
    assert plan["contexts"]["source"] == [["lcmv", "B"], ["wmne", "B"], ["lcmv", "A"], ["wmne", "A"]]
    assert plan["contexts"]["sensor"] == [["sensor", "B"], ["sensor", "A"]]
    assert {s["runtime"] for s in steps if s.get("experiment") in ("exp0", "exp1")} == {"CPU"}


def test_notebooks_are_up_to_date_and_thin(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_notebooks", REPO / "scripts" / "make_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "REPO", tmp_path)
    (tmp_path / "configs").symlink_to(REPO / "configs")
    mod.main()
    for f in sorted((REPO / "notebooks").glob("*")):
        assert (tmp_path / "notebooks" / f.name).read_text() == f.read_text(), f"{f.name} is stale: run make_notebooks"
    assert sorted(p.name for p in (tmp_path / "notebooks").glob("*")) == sorted(p.name for p in
                                                                                (REPO / "notebooks").glob("*"))
    for nb in (REPO / "notebooks").glob("*.ipynb"):
        src = "".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code")
        assert "!" not in [ln.strip()[:1] for ln in src.splitlines()]      # no shell magics: adgraph.colab.sh only
        assert "scripts/colab_setup.py" in src and "claude/unified-final-framework" in src
        assert "import torch.nn" not in src and "def fit" not in src        # no model code in notebooks


def test_one_release_pin_and_historical_pins_untouched(tmp_path):
    from adgraph.colab import has_completed_work, resolve_code_pin

    pin = tmp_path / release.PIN_FILE
    assert release.PIN_FILE == "FINAL_v1/CODE_COMMIT.txt"
    old = tmp_path / "D3_LCMV_XU" / "CODE_COMMIT.txt"
    old.parent.mkdir(parents=True)
    old.write_text("299aed72\n")
    commit, checkout = resolve_code_pin(pin, "abc123", has_work=has_completed_work(tmp_path, "FINAL"))
    assert (commit, checkout) == ("abc123", False) and pin.read_text().startswith("abc123")
    (tmp_path / release.ROOT / "results" / "x").mkdir(parents=True)
    (tmp_path / release.ROOT / "results" / "x" / "final.done.json").write_text("{}")
    assert has_completed_work(tmp_path, "FINAL")
    assert resolve_code_pin(pin, "def456", has_work=True) == ("abc123", True)   # never moves silently
    assert old.read_text() == "299aed72\n"
    setup = (REPO / "scripts" / "colab_setup.py").read_text()
    assert "--family" not in setup and "HISTORICAL_PINS" in setup


def test_a_real_store_is_never_run_without_its_qc(release_root):
    from adgraph.stores import open_store

    store, rec = open_store(release_root, "lcmv", "B")
    store.manifest = {**store.manifest, "synthetic": False}
    with pytest.raises(ValueError, match="QC"):
        run_experiment(experiment_configs()["exp1"], release_root, "lcmv", "B", variants=["logreg__psd"],
                       store=store, store_record=rec)


def test_notebooks_run_development_cv_until_the_frozen_final_analysis():
    nbs = {nb.name: "".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"]
                            if c["cell_type"] == "code")
           for nb in (REPO / "notebooks").glob("*.ipynb")}
    for name, src in nbs.items():
        if name != "99_results_report.ipynb":
            assert "xu_holdout" not in src and "final.py holdout" not in src and "final.py freeze" not in src, name
    assert "final.py freeze" in nbs["99_results_report.ipynb"] and "final.py holdout" in nbs["99_results_report.ipynb"]
    assert "--atlas selected" in nbs["03_graph_construction.ipynb"] and "ATLAS = 'selected'" in nbs["Exp5_multibranch.ipynb"]
    assert "build_stores.py match-atlas" in nbs["02_parcellation.ipynb"]
    assert "choose-atlas" not in nbs["04_roi_68_vs_18.ipynb"]                # 04 selects nothing
