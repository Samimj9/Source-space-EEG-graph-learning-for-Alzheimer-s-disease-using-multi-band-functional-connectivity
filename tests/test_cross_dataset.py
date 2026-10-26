"""Cross-dataset D3 <-> D2 (docs/cross_dataset/design.md): the D2 store builder, the compatibility report, the frozen
campaign, source-only fitting, label-blind target prediction, participant isolation, the unchanged D3 split,
checkpoints, the read-only release, participant-level uncertainty, and the synthetic end-to-end run in both
directions. Synthetic only (no EEG)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adgraph import release
from adgraph.checkpoint import IncompatibleCheckpoint
from adgraph.crossdataset import channels, compat, d2store, evaluate, namespace, plan, transfer
from adgraph.crossdataset.synthetic import write_all
from adgraph.evaluation.splits import subject_table, xu_partition

D3_ORDER = channels.load_table()["canonical_order"]


def _snapshot(dirs):
    out = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*")) if Path(d).exists() else []:
            if p.is_file():
                out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _quick(c, repeats=2, n_boot=200):
    return {**c, "source_cv": {**c["source_cv"], "D2": {**c["source_cv"]["D2"], "repeats": repeats}},
            "evaluation": {**c["evaluation"], "n_bootstrap": n_boot}}


QUICK_VARIANTS = None


def quick_variants(c):
    return sorted({v[0]["variant"] for v in plan.variants(c).values()} | {"logreg__psd"})


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """Synthetic D3 (release paths) and D2 (namespace) stores, one completed release development-CV result (Exp 1
    LR PSD, lcmv B lobes12), a frozen quick campaign, every family's first variant run in both directions with
    2 epochs, and its evaluation; fit / predict calls recorded."""
    from adgraph.experiments.config import with_context
    from adgraph.experiments.runner import run_variant
    from adgraph.models import classical, trainer

    tmp = tmp_path_factory.mktemp("cross")
    write_all(tmp, n_windows=4, psd_effect=0.8, conn_effect=0.8)
    c = _quick(plan.load_campaign())
    d3 = transfer.open_d3(tmp, "lcmv", "B", c)
    v = next(x for x in plan.variants(c)["exp1"] if x["variant"] == "logreg__psd")
    run_variant(d3, with_context(v, "lcmv", "B", "lobes12", None), "xu_dev_cv", release.results_dir(tmp))
    protected = _snapshot(namespace.protected(tmp))
    calls = {"fit": [], "predict": []}
    with pytest.MonkeyPatch.context() as mp:
        g_init, g_fit, g_pred = trainer.GNNModel.__init__, trainer.GNNModel.fit, trainer.GNNModel.predict_proba
        c_fit, c_pred = classical.ClassicalModel.fit, classical.ClassicalModel.predict_score

        def init(self, model, training, *a, **k):
            g_init(self, model, {**training, "epochs": 2}, *a, **k)

        def rec_fit(orig):
            def f(self, view):
                calls["fit"].append({"role": view.role, "keys": set(view.meta["key"]), "y": set(view.meta["y"])})
                return orig(self, view)
            return f

        def rec_pred(orig):
            def f(self, view, *a, **k):
                calls["predict"].append({"role": view.role, "keys": set(view.meta["key"]), "y": set(view.meta["y"])})
                return orig(self, view, *a, **k)
            return f
        mp.setattr(trainer.GNNModel, "__init__", init)
        mp.setattr(trainer.GNNModel, "fit", rec_fit(g_fit))
        mp.setattr(trainer.GNNModel, "predict_proba", rec_pred(g_pred))
        mp.setattr(classical.ClassicalModel, "fit", rec_fit(c_fit))
        mp.setattr(classical.ClassicalModel, "predict_score", rec_pred(c_pred))
        with pytest.raises(plan.NotFrozen):
            transfer.run(tmp, c, variants=quick_variants(c), synthetic=True)
        frozen = plan.freeze(tmp, c, transfer.stores_record(tmp, c), plan.decision_audit(tmp, c), synthetic=True)
        t = transfer.run(tmp, c, variants=quick_variants(c), synthetic=True)
        ev = evaluate.evaluate(tmp, c)
        t2 = transfer.run(tmp, c, variants=quick_variants(c), synthetic=True)
    return {"tmp": tmp, "c": c, "t": t, "t2": t2, "ev": ev, "calls": calls, "frozen": frozen,
            "protected": protected}


# ------------------------------------------------------------------------------------------- the D2 store builder
def _d2_inputs(tmp_path, n=3, seconds=70):
    import edfio

    rs = tmp_path / "EEGs" / "RestingState"
    rs.mkdir(parents=True)
    rng = np.random.default_rng(0)
    ids, groups = [2, 17, 3][:n], ["HC", "AD", "HC"][:n]
    for sid in ids:
        common = rng.standard_normal(1024 * seconds)
        data = (0.8 * common + 0.6 * rng.standard_normal((32, 1024 * seconds))) * 10.0
        edfio.Edf([edfio.EdfSignal(x, sampling_frequency=1024, label=f"Misc ExG{k + 1}", physical_dimension="uV",
                                   physical_range=(-20000, 20000)) for k, x in enumerate(data)]
                  ).write(rs / f"{sid}_842019_1.edf")
    (tmp_path / "EEGs" / "PictureNaming").mkdir()
    (tmp_path / "EEGs" / "PictureNaming" / "2_842019_2.edf").write_bytes(b"never read")
    sheet = tmp_path / "info.xlsx"
    pd.DataFrame({"Id": [f"{i}_842019" for i in ids], "Age": [60, 81, 63][:n], "Gender": ["F", "M", "F"][:n],
                  "Group": groups, "MMSE": [30, 18, 29][:n]}).to_excel(sheet, index=False)
    return {"dataset2": {"eeg_dir": str(rs), "participants": str(sheet)}, "cache_dir": str(tmp_path / "cache")}


def test_d2_store_build_validate_and_resume(tmp_path, monkeypatch):
    from adgraph.source.atlas import region_names
    from adgraph.sourcegraphs import wmne
    from adgraph.sourcegraphs.lcmv import sensor_info

    paths = _d2_inputs(tmp_path)
    table = channels.load_table()
    g = channels.template_geometry(table)
    cfg = d2store.resolve("wmne", table)
    W = np.random.default_rng(1).standard_normal((12, 19))                 # stand-in for the fsaverage operator
    ctxs = {g["hash"]: {"geometry": g, "info": sensor_info(cfg, g), "labels": None,
                        "roi_names": region_names("lobes12"), "channels": D3_ORDER, "wmne": {"W": W, "inv": None}}}
    monkeypatch.setattr(wmne, "verify", lambda *a, **k: 0.0)
    with pytest.raises(namespace.NotAuthorized):                            # never without the author's record
        d2store.run(paths, tmp_path, "wmne", g, tmp_path / "fs", limit=3, ctxs=ctxs)
    b, a = d2store.run(paths, tmp_path, "wmne", g, tmp_path / "fs", limit=3, ctxs=ctxs, authorized=True)
    assert namespace.root(tmp_path) in b.parents and b.name.startswith("graphsXD2W_v1_")
    assert not (b / "xu_split.json").exists()                               # no ds004504 partition in D2
    for d, mode in ((b, "B"), (a, "A")):
        r = d2store.validate(d, expected=None)
        assert r["passed"], {k: v for k, v in r["checks"].items() if not v["passed"]}
    st = d2store.open_d2(b)
    assert st.channels == D3_ORDER and st.roi_names == region_names("lobes12") and st.bands == [
        "delta", "theta", "alpha", "beta", "gamma"]
    assert st.arrays["plv"].shape[1:] == (5, 12, 12) and st.arrays["sensor_psd_de"].shape[1:] == (5, 19)
    assert set(st.windows["dataset"]) == {"D2"} and sorted(st.subjects["key"]) == ["D2-17", "D2-2", "D2-3"]
    assert json.loads(st.subjects["x_edf_picks"].iloc[0]) == [r["edf_label"] for r in table["channels"]]
    assert (st.windows.groupby("key").size() <= 291).all() and st.manifest["cross_dataset"]["dataset"] == "D2"
    assert len(d2store.open_d2(a).windows) == 10 * 3                        # mode A: first 10 accepted windows
    before = (b / "subjects" / "D2-2" / "arrays.npz").stat().st_mtime_ns
    d2store.run(paths, tmp_path, "wmne", g, tmp_path / "fs", limit=3, ctxs=ctxs, authorized=True)
    assert (b / "subjects" / "D2-2" / "arrays.npz").stat().st_mtime_ns == before     # resumed, not recomputed


def test_the_d2_configuration_differs_from_d3_only_by_the_documented_overrides():
    from adgraph.sourcegraphs.config import resolve
    from adgraph.sourcegraphs.match import _config_diff

    for inv in ("lcmv", "wmne"):
        diff = set(_config_diff(resolve(inv, "lobes12"), d2store.resolve(inv)))
        assert diff <= compat.EXPECTED_CONFIG_DIFFERENCES
        for k in ("bands", "filters", "windowing", "source", "psd", "features"):
            assert resolve(inv, "lobes12")[k] == d2store.resolve(inv)[k]
    assert resolve("lcmv", "dk68") and resolve("wmne", "dk68")              # the validated D3 hashes still resolve


# ------------------------------------------------------------------------------------------- compatibility
def test_compatibility_report_and_shape_gate(campaign):
    from adgraph.sourcegraphs.config import resolve

    tmp, c = campaign["tmp"], campaign["c"]
    d3, d2 = transfer.open_d3(tmp, "lcmv", "B", c), transfer.open_d2(tmp, "lcmv", "B", c)
    t = compat.compatibility(d3, d2, resolve("lcmv", "lobes12"), d2store.resolve("lcmv"), channels.load_table())
    assert compat.passed(t), t[t["status"].str.startswith("DIFFERENT")]
    assert set(t["status"]) >= {"identical", "approximate", "different"}
    assert t.set_index("property").loc["artifact cleaning", "status"] == "different"
    shift = compat.shift_diagnostics({"D3": d3, "D2": d2})
    assert {"D3_median", "D2_median", "standardized_difference_D2_minus_D3"} <= set(shift)
    bad = d2.subset(np.arange(d2.n_windows))
    bad.roi_names = bad.roi_names[::-1]
    with pytest.raises(transfer.InputMismatch, match="ROIs"):
        transfer.check_stores(d3, bad, sensor=False)
    t2 = compat.compatibility(d3, bad, resolve("lcmv", "lobes12"), d2store.resolve("lcmv"), channels.load_table())
    assert not compat.passed(t2)


def test_all_six_gnn_families_accept_both_harmonized_datasets(campaign):
    from adgraph.experiments.runner import with_store_context

    tmp, c = campaign["tmp"], campaign["c"]
    for inv in ("lcmv", "wmne"):
        d3, d2 = transfer.open_d3(tmp, inv, "B", c), transfer.open_d2(tmp, inv, "B", c)
        n = 0
        for exp in c["experiments"]["gnn"]:
            for v in plan.variants(c)[exp]:
                vc = with_store_context(plan.contextualize(v, inv, "B", c), d3)
                for src, tgt in ((d3, d2), (d2, d3)):
                    s, t = transfer._datasets(vc, src, tgt)
                    info = transfer.check_inputs(s, t, vc)
                    assert info["roi_names"] == d3.roi_names and len(info["roi_names"]) == 12
                n += 1
        assert n == 48
    done = campaign["t"]
    for exp in ("exp3", "exp4", "exp5", "exp6", "exp7", "exp8a"):
        assert (done[done["experiment"].str.startswith(exp + "_")]["state"] == "completed").all()


# ------------------------------------------------------------------------------------------- the campaign
def test_campaign_units_use_the_existing_definitions_and_exp8a(tmp_path):
    import yaml

    c = plan.load_campaign()
    vs = plan.variants(c)
    assert {k: len(v) for k, v in vs.items()} == {"exp3": 10, "exp4": 4, "exp5": 8, "exp6": 10, "exp7": 6,
                                                  "exp8a": 10, "exp0": 4, "exp1": 4}
    assert vs["exp8a"][0]["experiment"] == "exp8a_xu_dual_gcn_surface"
    assert all(v["model"].get("anatomical_grouping") != "lobe_hemisphere" for v in vs["exp8a"])
    us = plan.units(c)
    assert len(us) == 2 * 2 * (48 * 2 + 4 * 2) + 2 * 2 * 4        # directions x modes x (GNN+Exp1 x inverses) + Exp0
    bad = {**c, "experiments": {**c["experiments"], "gnn": c["experiments"]["gnn"] + ["exp8"]}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="exp8a"):                      # the degenerate original Exp 8 is refused
        plan.load_campaign(tmp_path / "c.yaml")
    for v in vs["exp3"] + vs["exp8a"]:
        assert v["training"]["epochs"] == 60 and v["protocol"]["name"] == "gnn_fixed60_v1"


def test_no_target_fitting_and_blind_prediction(campaign):
    tmp, c = campaign["tmp"], campaign["c"]
    d3, d2 = transfer.open_d3(tmp, "lcmv", "B", c), transfer.open_d2(tmp, "lcmv", "B", c)
    d2_keys, d3_keys = set(d2.windows["key"]), set(d3.windows["key"])
    fits = campaign["calls"]["fit"]
    assert fits and all(f["role"] == "train" for f in fits)
    for f in fits:                                                         # a fit sees ONE dataset only
        assert f["keys"] <= d2_keys or f["keys"] <= d3_keys
        assert -1 not in f["y"]
    blind_preds = [p for p in campaign["calls"]["predict"] if p["y"] == {-1}]
    assert blind_preds and all(p["role"] == "test" for p in blind_preds)
    for p in campaign["calls"]["predict"]:
        if p["y"] != {-1}:                                                 # labelled predictions: source-only CV
            assert p["keys"] <= d2_keys or p["keys"] <= d3_keys
    for d in (namespace.root(tmp) / "results").rglob("transfer/seed_0"):
        w = pd.read_csv(d / "target_windows_blind.csv")
        assert "y" not in w and "label" not in w
        assert "y" not in pd.read_csv(d / "target_participants_blind.csv")
    b = transfer.blind(d2)
    assert set(b.windows["y"]) == {-1} and set(b.windows["label"]) == {"masked"} and "age" not in b.subjects


def test_participant_isolation_and_the_unchanged_d3_split(campaign):
    tmp, c = campaign["tmp"], campaign["c"]
    d3 = transfer.open_d3(tmp, "lcmv", "B", c)
    dev, test = xu_partition(subject_table(d3.windows))
    saved = json.loads((d3.root / "xu_split.json").read_text())
    assert (dev, test) == (saved["development"], saved["test"]) and (len(dev), len(test)) == (52, 13)
    assert transfer.source_keys("D3_to_D2", d3) == dev
    for d in (namespace.root(tmp) / "results" / "D3_to_D2").rglob("training.json"):
        tr = json.loads(d.read_text())
        assert tr["source_participants"] == dev and not set(tr["source_participants"]) & set(test)
        assert tr["n_target_participants"] == 23
    for d in (namespace.root(tmp) / "results" / "D2_to_D3").rglob("training.json"):
        tr = json.loads(d.read_text())
        assert all(k.startswith("D2-") for k in tr["source_participants"]) and tr["n_target_participants"] == 65
    for d in (namespace.root(tmp) / "results" / "D2_to_D3").rglob("source_cv/units"):
        for f in d.glob("*.pred.csv"):
            p = pd.read_csv(f)
            assert set(p["dataset"]) == {"D2"}
        preds = pd.concat(pd.read_csv(f) for f in d.glob("fold_r000_*.pred.csv"))
        assert not preds["key"].duplicated().any() or preds.groupby("key")["fold"].nunique().max() == 1


def test_participant_aggregation_is_stable(campaign):
    tmp = campaign["tmp"]
    for d in list((namespace.root(tmp) / "results").rglob("transfer/seed_0"))[:6]:
        w = pd.read_csv(d / "target_windows_blind.csv")
        p = pd.read_csv(d / "target_participants_blind.csv").set_index("key")
        m = w.sample(frac=1, random_state=0).groupby("key")["score"].agg(["mean", "size"])
        assert np.allclose(p.loc[m.index, "score"], m["mean"]) and (p.loc[m.index, "n_windows"] == m["size"]).all()


def test_checkpoints_resume_and_identity(campaign):
    tmp, c = campaign["tmp"], campaign["c"]
    assert (campaign["t"]["state"] == "completed").all() and (campaign["t2"]["state"] == "reused").all()
    u = next(x for x in plan.units(c) if x["variant"] == "logreg__psd" and x["exp"] == "exp1"
             and x["direction"] == "D3_to_D2" and x["inverse"] == "lcmv" and x["mode"] == "B")
    d = transfer.unit_dir(tmp, u, c) / "transfer" / "seed_0"
    marker = json.loads((d / "transfer.done.json").read_text())
    assert marker["identity"]["kind"] == "cross_transfer" and marker["identity"]["seed"] == 0
    d3, d2 = transfer.open_d3(tmp, "lcmv", "B", c), transfer.open_d2(tmp, "lcmv", "B", c)
    frozen = plan.require_frozen(tmp, c)
    other = {**frozen, "campaign_hash": "different"}
    with pytest.raises(IncompatibleCheckpoint):
        transfer.run_unit(tmp, u, c, other, d3, d2)
    assert transfer.run_unit(tmp, u, c, frozen, d3, d2) == "reused"
    with pytest.raises(plan.NotFrozen, match="differs"):
        plan.require_frozen(tmp, {**c, "training": {**c["training"], "seeds": [7]}})
    with pytest.raises(plan.NotFrozen, match="different campaign"):
        plan.freeze(tmp, {**c, "modes": ["B"]}, frozen["stores"], {"passed": True}, synthetic=True)
    stale = {**frozen, "stores": {**frozen["stores"], "D2/lcmv/B": "a-rebuilt-store"}}
    with pytest.raises(RuntimeError, match="frozen campaign recorded"):          # a replaced store is refused
        transfer.check_frozen_stores(stale, "lcmv", "B", d3, d2)


def test_gates_authorization_freeze_and_commit(campaign, monkeypatch):
    from adgraph import provenance

    tmp, c = campaign["tmp"], campaign["c"]
    with pytest.raises(namespace.NotAuthorized, match="train"):
        transfer.run(tmp, c, variants=["logreg__psd"])                    # real data: needs the author's record
    monkeypatch.setattr(provenance, "git_commit", lambda *a, **k: "another-commit")
    with pytest.raises(plan.NotFrozen, match="frozen at commit"):
        plan.require_frozen(tmp, c)
    with pytest.raises(plan.NotFrozen, match="frozen at commit"):
        evaluate.evaluate(tmp, c)
    audit = plan.decision_audit(tmp, c)                                    # no recorded decisions in this tmp
    assert not audit["passed"] and audit["disclosures"]
    with pytest.raises(plan.NotFrozen, match="decision audit"):
        plan.freeze(tmp / "elsewhere", c, {}, audit)


def test_nothing_outside_the_namespace_is_written(campaign):
    tmp = campaign["tmp"]
    assert _snapshot(namespace.protected(tmp)) == campaign["protected"]  # release results, QC, lock, D3 stores
    with pytest.raises(namespace.NamespaceError):
        namespace.guard(tmp, release.root(tmp) / "x.json")
    with pytest.raises(namespace.NamespaceError):
        namespace.sub(tmp, "..", "FINAL_v1", "decisions", "atlas.json")
    top = {p.name for p in tmp.iterdir()}
    assert top <= {"FINAL_v1", "D3_LCMV_XU", "D3_WMNE_XU", namespace.ROOT}


# ------------------------------------------------------------------------------------------- evaluation
def test_participant_level_uncertainty():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(10), np.zeros(13)].astype(int)
    s = rng.random(23) + 0.3 * y
    draws = evaluate.stratified_bootstrap(y, s, 0.5, 300, 0)
    assert len(draws) == 300 and draws["roc_auc"].between(0, 1).all()
    part = pd.DataFrame({"y": y, "score": s})
    m, _ = evaluate.participant_metrics(part, 0.5, 300, 0)
    assert (m["n"], m["n_ad"], m["n_hc"]) == (23, 10, 13)
    assert m["auc_delong_low"] <= m["roc_auc"] <= m["auc_delong_high"]
    assert m["roc_auc_boot_low"] <= m["roc_auc"] <= m["roc_auc_boot_high"]
    assert np.allclose(evaluate.holm(np.array([0.01, 0.04, 0.03, np.nan])), [0.03, 0.06, 0.06, np.nan],
                       equal_nan=True)


def test_end_to_end_both_directions(campaign):
    ev, c, tmp = campaign["ev"], campaign["c"], campaign["tmp"]
    long = ev["metrics_long.csv"]
    assert set(long["direction"]) == {"D3_to_D2", "D2_to_D3"}
    a = long[long["stratum"] == "all"]
    assert set(a.loc[a["direction"] == "D3_to_D2", "n"]) == {23} and set(a.loc[a["direction"] == "D2_to_D3", "n"]) == {65}
    strata = long[long["direction"] == "D2_to_D3"].groupby("stratum")["n"].first().to_dict()
    assert strata == {"all": 65, "d3_development": 52, "d3_holdout": 13}
    assert (long.loc[long["direction"] == "D2_to_D3", "source_cv_status"] == "computed").all()
    ref = long[(long["direction"] == "D3_to_D2") & (long["variant"] == "logreg__psd") & (long["inverse"] == "lcmv")
               & (long["mode"] == "B") & long["experiment"].str.startswith("exp1")]
    assert ref["source_cv_status"].iloc[0] == "available" and np.isfinite(ref["degradation_roc_auc"].iloc[0])
    pred = ev["predictions_participants.csv"]
    for col in ("key", "subject_id", "y", "mean_score", "n_windows", "source_dataset", "target_dataset", "inverse",
                "experiment", "variant", "seed", "git_commit", "source_sha256"):
        assert col in pred
    assert len(ev["paired_comparisons.csv"]) and "p_holm" in ev["paired_comparisons.csv"]
    assert {"age_only_transfer", "age_only_source_cv"} <= set(ev["age_only_baseline.csv"]["analysis"])
    assert set(ev["demographics.csv"]["dataset"]) == {"D3", "D2"}
    from adgraph.crossdataset import report

    r = report.report(tmp, c)
    files = {p.name for p in Path(r["dir"]).iterdir()}
    assert {"table1_transfer.csv", "table2_d3_strata.csv", "table3_paired.csv", "table4_age.csv",
            "README.md"} <= files and any(f.startswith("fig_forest_") for f in files)
    assert "SYNTHETIC" in (Path(r["dir"]) / "README.md").read_text()


def test_cross_notebooks_use_their_own_branch_pin_and_gates():
    nbs = sorted((Path(__file__).resolve().parents[1] / "notebooks" / "cross_dataset").glob("*.ipynb"))
    assert [n.name for n in nbs] == ["00_CrossDataset_Audit_and_Build.ipynb", "01_CrossDataset_D3_to_D2.ipynb",
                                     "02_CrossDataset_D2_to_D3.ipynb", "03_CrossDataset_Comparison_Report.ipynb"]
    for nb in nbs:
        cells = ["".join(c["source"]) for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code"]
        src = "\n".join(cells)
        assert "claude/cross-dataset-d3-d2-19ch" in cells[0] and "'--namespace', 'cross'" in cells[0]
        assert "claude/unified-final-framework" not in src and "--amendment" not in src
        assert "final.py" not in src and "UPDATE_CODE = True" not in src
        assert "ACCEPT_ENVIRONMENT_DIFFERENCE = False" in src and "scripts/cross_dataset.py" in src
        assert "!" not in [ln.strip()[:1] for ln in src.splitlines()]
    gate = "\n".join("".join(c["source"]) for c in json.loads(nbs[0].read_text())["cells"])
    assert "AUTHORIZED_BY = ''" in gate and "FREEZE_CAMPAIGN = False" in gate
    for nb in nbs[1:3]:
        src = "".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"])
        assert "AUTHORIZED_BY = ''" in src and "evaluate" not in src        # no target labels before notebook 03
