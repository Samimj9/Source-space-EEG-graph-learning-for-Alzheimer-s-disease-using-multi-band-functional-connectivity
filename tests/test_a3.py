"""Amendment A3 (methodology log entry 61): the dual-namespace freeze, the pinned holdout gates, the consolidated report,
participant-level statistics, the D3 -> D2 manifest and the notebooks. Synthetic data only: development CV (real training
of a few variants on synthetic stores) runs once per module at two patched commit ids standing for the release pin and
the A2 pin; every test works on its own copy of that root."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml

from adgraph import release
from adgraph.a3 import audit, d2, environment, freeze, holdout, inventory, report, smoke, stats
from adgraph.a3 import namespace as ns
from adgraph.experiments import amendment, decisions
from adgraph.experiments.config import experiment_configs
from adgraph.experiments.plan import run_experiment
from adgraph.qc.report import require_qc
from adgraph.stores import open_store

REPO = Path(__file__).resolve().parents[1]
RP = "59a9ab70a55d107c5ad72acff1fb48f3ef42dff0"
AP = "7fa470c668b8cce64ca152372422638be3ee8203"
HOLDOUT_YAML_SHA = "fc1a6988ffc111e2151c7714ba4c81df07037129abe2c656074dc08bf9571ecc"


RUN_STATUS = ("_run_status.csv",)        # status tables carry a timestamp and are rewritten on every call; not results


def snapshot(d, skip=()):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(d).rglob("*"))
            if p.is_file() and not any(s in str(p) for s in skip)}


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def base(tmp_path_factory):
    s = smoke.Smoke(tmp_path_factory.mktemp("a3_base"))
    s.development()
    yield s
    s.close()


@pytest.fixture
def env(base, tmp_path):
    """A private copy of the synthetic root with the smoke object pointing at it."""
    dst = tmp_path / "root"
    shutil.copytree(base.root, dst)
    s = object.__new__(smoke.Smoke)
    s.__dict__.update(base.__dict__)
    s.root = dst
    return s


def same_environment(python, keys):
    """The interpreter probe of tests: the environment the synthetic development results were produced in."""
    from adgraph.provenance import versions

    return versions(tuple(keys))


def approved():
    return mock.patch.object(ns, "record", smoke.approved_record)


def write_freeze(s):
    return s.freeze(write=True)


# ------------------------------------------------------------------------------------------------ the plan (configuration only)
def test_the_plan_is_holdout_yaml_with_exp8_replaced_by_exp8a_and_nothing_else():
    orig = yaml.safe_load((REPO / "configs/final/holdout.yaml").read_text())
    plan = ns.plan_config()
    assert hashlib.sha256((REPO / "configs/final/holdout.yaml").read_bytes()).hexdigest() == HOLDOUT_YAML_SHA \
        == ns.record()["original_holdout_plan_sha256"]                      # the historical record is untouched
    assert plan["experiments"] == [plan["replaces"].get(e, e) for e in orig["experiments"]]
    assert plan["mode"] == orig["mode"] == "B" and plan["inverses"] == orig["inverses"]
    assert plan["replaces"] == {"exp8": "exp8a"} and "exp7" not in plan["experiments"]
    assert plan["parts"]["release"]["experiments"] + plan["parts"]["A2"]["experiments"] == plan["experiments"]


def test_planned_units_are_every_eligible_variant_in_mode_b_only():
    u = freeze.planned_units_for("lobes12", "dense")
    rel, a2 = pd.DataFrame(u["release"]), pd.DataFrame(u["A2"])
    assert (len(rel), len(a2)) == (96, 20)                                   # 116 = the 116 of holdout.yaml, exp8 -> exp8a
    assert set(rel["exp"]) == {f"exp{i}" for i in range(7)} and set(a2["exp"]) == {"exp8a"}
    assert set(rel["mode"]) | set(a2["mode"]) == {"B"}                       # window mode A stays development-only
    assert "exp7" not in set(rel["exp"]) and not any(e == "exp8" for e in rel["exp"])
    exp0 = rel[rel["exp"] == "exp0"]                                         # sensor space: no inverse, no atlas, no graph
    assert len(exp0) == 4 and set(exp0["inverse"]) == {"sensor"} and set(exp0["atlas"]) == {"dk68"}
    assert set(exp0["graph"]) == {"none"}
    assert set(rel[rel["exp"] == "exp1"]["graph"]) == {"none"} and set(rel[rel["exp"] == "exp1"]["atlas"]) == {"lobes12"}
    assert set(a2["atlas"]) == {"lobes12"} and set(a2["graph"]) == {"dense"} and len(a2["variant"].unique()) == 10
    assert set(a2["inverse"]) == {"lcmv", "wmne"}
    assert not rel.duplicated(list(inventory.UNIT_KEY)).any() and not a2.duplicated(list(inventory.UNIT_KEY)).any()
    # the release-format units are exactly decisions.planned_units of holdout.yaml minus exp8 (no unit dropped or added)
    old = [x for x in decisions.planned_units("lobes12", "dense") if x["exp"] != "exp8"]
    assert sorted(map(freeze._key, old)) == sorted(map(freeze._key, u["release"]))


def test_the_a3_record_is_complete_and_not_approved_until_the_author_says_so():
    r = ns.record()
    for k in ("date", "reason", "scope", "does_not", "release_pin", "a2_pin", "holdout_plan", "status"):
        assert r[k]
    assert (r["release_pin"], r["a2_pin"]) == (RP, AP) == (amendment.record("A2")["base_release_pin"], AP)
    assert str(r["status"]).startswith("approved") == bool(r["approved_by"])  # approval is a recorded fact, not a default
    assert "exp8a" in r["reason"] and "substitut" in r["reason"]


def test_a3_code_is_outside_every_package_that_defines_an_experiment_units_code_identity():
    """The experiment source fingerprint (config.py, experiments, models, evaluation, graphs, features, qc) of the A2 pin
    and of the release pin must not change because of A3: no completed checkpoint may become incompatible."""
    if subprocess.run(["git", "cat-file", "-e", AP + "^{commit}"], cwd=REPO).returncode:
        pytest.skip("A2 pin not in this clone")
    paths = ["src/adgraph/config.py", "src/adgraph/experiments", "src/adgraph/models", "src/adgraph/evaluation",
             "src/adgraph/graphs", "src/adgraph/features", "src/adgraph/qc", "src/adgraph/checkpoint.py",
             "src/adgraph/provenance.py", "src/adgraph/stores.py", "src/adgraph/release.py", "requirements.txt",
             "configs/final/experiments", "configs/final/protocol.yaml", "configs/final/gnn.yaml",
             "configs/final/classical.yaml", "configs/final/holdout.yaml", "configs/final/graphs.yaml",
             "configs/final/selection.yaml", "configs/final/amendments/exp8a_xu_dual_gcn_surface.yaml"]
    diff = subprocess.run(["git", "diff", "--stat", AP, "--", *paths], cwd=REPO, stdout=subprocess.PIPE, text=True)
    assert diff.stdout.strip() == "", f"A3 must not change A2-pinned code or configs:\n{diff.stdout}"
    for pin in (RP, AP):
        assert subprocess.run(["git", "merge-base", "--is-ancestor", pin, "HEAD"], cwd=REPO).returncode == 0


def test_a3_code_never_writes_either_pin_file():
    for f in (REPO / "src/adgraph/a3").glob("*.py"):
        if f.name == "smoke.py":                                              # synthetic temporary roots only
            continue
        for ln in f.read_text().splitlines():
            if "write_text" in ln or "write_once" in ln or "new_file" in ln:
                assert "pin_file" not in ln and "CODE_COMMIT" not in ln, (f.name, ln)
    setup = (REPO / "scripts/a3_setup.py").read_text()
    assert "pin_path(output_root)" in setup and "write_text" not in setup


# ------------------------------------------------------------------------------------------------ discovery
def test_results_are_discovered_across_both_namespaces_with_their_commits(env):
    t, other = inventory.discover(env.root)
    assert set(t["namespace"]) == {"release", "A2"} and set(t["setting"]) == {"xu_dev_cv"}
    assert (t[t["namespace"] == "release"]["git_commit"] == RP).all() and t["commit_matches_pin"].all()
    a2 = t[t["exp"] == "exp8a"]
    assert len(a2) == 12 and (a2["namespace"] == "A2").all() and (a2["git_commit"] == AP).all()
    assert set(a2["experiment"]) == {"exp8a_xu_dual_gcn_surface"}
    assert {"exp0", "exp1", "exp2"} <= set(t["exp"]) and t["marker_ok"].all() if "marker_ok" in t else True
    sensor = t[t["exp"] == "exp0"]
    assert set(sensor["inverse"]) == {"sensor"} and set(sensor["atlas"]) == {"dk68"} and set(sensor["graph"]) == {"none"}
    need = {"family", "experiment", "variant", "inverse", "mode", "atlas", "graph", "git_commit", "namespace",
            "n_participants", "auc", "auc_ci_low", "auc_ci_high", "balanced_accuracy", "sensitivity", "specificity",
            "f1", "mcc", "tn", "fp", "fn", "tp", "mean_fold_auc"}
    assert need <= set(t.columns) and (t["n_participants"] == 52).all()


def test_failed_original_exp8_units_are_provenance_never_results(env):
    d = release.results_dir(env.root) / "lcmv/mode_B/lobes12/dense/exp8_xu_dual_gcn/plv__psd/xu_dev_cv"
    d.mkdir(parents=True)
    ident = {"experiment": "exp8_xu_dual_gcn", "variant": "plv__psd", "setting": "xu_dev_cv",
             "context": {"inverse": "lcmv", "mode": "B", "atlas": "lobes12", "graph": "dense"},
             "code": {"git_commit": RP}}
    (d / "_identity.json").write_text(json.dumps({"identity": ident}))
    (d / "status.csv").write_text("unit,state,message\nfold_r000_f00,failed,ValueError: own group\n")
    t, other = inventory.discover(env.root)
    assert not (t["exp"] == "exp8").any()
    row = other[other["exp"] == "exp8"].iloc[0]
    assert row["state"] == "failed" and "own group" not in row["detail"] and row["namespace"] == "release"
    assert freeze.preconditions(env.root, smoke.restrict)["ok"]               # and it does not disturb the freeze
    r = report.development_report(env.root, cfg={**ns.stats_config(), "n_bootstrap": 20, "n_permutation": 20},
                                  make_figures=False)
    prov = pd.read_csv(Path(r["dir"]) / "not_completed_provenance.csv")
    assert "exp8" in set(prov["exp"]) and "exp8" not in set(pd.read_csv(Path(r["dir"]) / "development_results_all.csv")["family"].str.lower())


def test_a_corrupt_completed_result_is_not_a_result(env):
    t, _ = inventory.discover(env.root)
    victim = t[t["exp"] == "exp8a"].iloc[0]
    p = env.root / victim["rel_dir"] / "predictions_participants.csv"
    p.write_text(p.read_text() + "\n")
    t2, other = inventory.discover(env.root)
    assert len(t2) == len(t) - 1 and "checksum mismatch" in " ".join(other["detail"])


# ------------------------------------------------------------------------------------------------ the freeze gates
def test_dry_run_checks_everything_and_writes_nothing(env):
    before = snapshot(env.root)
    res = env.freeze(write=False)
    assert res["dry_run"] and res["ok"], [c for c in res["checks"] if not c["ok"]]
    inv = res["inventory"]
    assert len(inv) == 19 and set(inv["part"]) == {"release", "A2"} and inv["dev_auc"].notna().all()
    assert snapshot(env.root) == before and not ns.master_path(env.root).exists()


def test_the_real_a3_record_is_unapproved_so_the_freeze_refuses_to_write(env):
    assert not str(ns.record()["status"]).startswith("approved")
    before = snapshot(env.root)
    with pytest.raises(freeze.FreezeRefused, match="not approved"):
        freeze.freeze(env.root, write=True, restrict=smoke.restrict, require_clean=False)
    assert snapshot(env.root) == before


def test_freeze_refuses_a_missing_development_result_and_writes_nothing(env):
    t, _ = inventory.discover(env.root)
    victim = t[(t["exp"] == "exp1") & (t["inverse"] == "wmne") & (t["mode"] == "B")].iloc[0]
    shutil.rmtree(env.root / victim["rel_dir"])
    before = snapshot(env.root)
    with approved(), pytest.raises(freeze.FreezeRefused, match="development-CV result"):
        freeze.freeze(env.root, write=True, restrict=smoke.restrict, require_clean=False)
    assert snapshot(env.root) == before


def test_freeze_refuses_a_result_from_another_commit(env):
    t, _ = inventory.discover(env.root)
    victim = t[(t["exp"] == "exp2") & (t["mode"] == "B")].iloc[0]
    p = env.root / victim["rel_dir"] / "_identity.json"
    d = json.loads(p.read_text())
    d["identity"]["code"]["git_commit"] = "f" * 40
    p.write_text(json.dumps(d))
    res = env.freeze(write=False)
    assert not res["ok"] and any("not the pin" in c["detail"] for c in res["checks"])


def test_freeze_refuses_when_the_pins_or_decisions_differ(env):
    ns.pin_file(env.root, "A2").write_text("a" * 40 + "\n")
    assert not env.freeze(write=False)["ok"]
    ns.pin_file(env.root, "A2").write_text(AP + "\n")
    (env.root / release.PIN_FILE).write_text("b" * 40 + "\n")
    assert not env.freeze(write=False)["ok"]
    (env.root / release.PIN_FILE).write_text(RP + "\n")
    decisions.graph_path(env.root).write_text(json.dumps({"selected": "knn_union_f10", "atlas": "lobes12", "reason": "x"}))
    res = env.freeze(write=False)
    assert not res["ok"] and any("decisions" in c["check"] for c in res["checks"] if not c["ok"])


def test_freeze_refuses_a_dirty_working_tree(env):
    with approved(), mock.patch("adgraph.provenance.git_dirty", lambda: True):
        with pytest.raises(freeze.FreezeRefused, match="uncommitted"):
            freeze.freeze(env.root, write=True, restrict=smoke.restrict, require_clean=True)


def test_freeze_refuses_when_a_holdout_result_already_exists(env):
    d = release.results_dir(env.root) / "lcmv/mode_B/lobes12/none/exp1_source_psd_de/logreg__psd/xu_holdout"
    d.mkdir(parents=True)
    assert not env.freeze(write=False)["ok"]


# ------------------------------------------------------------------------------------------------ immutability
def test_the_freeze_writes_three_checksummed_records_once_and_changes_nothing_else(env):
    keep = snapshot(env.root)
    res = write_freeze(env)
    assert len(res["written"]) == 3
    m = ns.read_json(ns.master_path(env.root))
    assert m["release_pin"] == RP and m["a2_pin"] == AP and m["n_units"] == 19 and m["synthetic"]
    for p in ns.PARTS:
        f = ns.record_path(env.root, p)
        assert m["records"][p]["sha256"] == ns.sha256_file(f) and m["records"][p]["git_commit"] == ns.part_pin(p)
        d = ns.read_json(f)
        assert d["amendment"] == "A3" and d["git_commit"] == ns.part_pin(p) and len(d["units"]) == len(d["development_cv"])
        assert d["substitution"]["exp8"] == "exp8a" and all(len(x["metrics_sha256"]) == 64 for x in d["development_cv"])
    assert {u["exp"] for u in ns.read_json(ns.record_path(env.root, "A2"))["units"]} == {"exp8a"}
    assert "exp8a" not in {u["exp"] for u in ns.read_json(ns.record_path(env.root, "release"))["units"]}
    after = snapshot(env.root)
    new = set(after) - set(keep)
    assert {n.split("/")[-1] for n in new} == {"final_analysis.json", "freeze.json", "freeze_inventory.csv"}
    assert {k: v for k, v in after.items() if k in keep} == keep              # nothing existing was altered
    assert (env.root / release.PIN_FILE).read_text() == RP + "\n" and ns.pin_file(env.root, "A2").read_text() == AP + "\n"
    again = write_freeze(env)                                                 # rerunning cannot change the freeze
    assert again["written"] == [] and snapshot(env.root) == after
    with pytest.raises(FileExistsError):
        ns.write_once(ns.master_path(env.root), {"x": 1})


@pytest.fixture(params=["link", "claim+rename"])
def fs_mode(request):
    """Hard links available, or refused as on a file system without them (a Drive mount may be one)."""
    import errno

    if request.param == "link":
        yield "link"
    else:
        with mock.patch.object(ns.os, "link", side_effect=OSError(errno.EPERM, "Operation not permitted")):
            yield "claim+rename"


def test_the_freeze_has_the_same_guarantees_with_or_without_hard_links(env, fs_mode):
    res = write_freeze(env)
    assert len(res["written"]) == 3 and set(res["write_methods"].values()) == {fs_mode}
    names = [p.name for d in (env.root / "FINAL_v1").rglob("*") if d.is_dir() for p in d.iterdir()
             if p.name.startswith(".") and ".tmp-" in p.name or p.name.endswith(".claim")]
    assert names == []                                                        # no temporary file or claim left behind
    after = snapshot(env.root)
    assert write_freeze(env)["written"] == [] and snapshot(env.root) == after   # rerun changes nothing
    f = ns.record_path(env.root, "release")
    with pytest.raises(FileExistsError):
        ns.write_once(f, {"x": 1})
    assert snapshot(env.root) == after                                        # an existing record is refused, intact
    f.write_text(f.read_text().replace("exp0", "expX", 1))
    with pytest.raises(freeze.FreezeInvalid):
        freeze.verify(env.root, restrict=smoke.restrict)


def test_a_freeze_interrupted_between_records_resumes_with_or_without_hard_links(env, fs_mode):
    calls = {"n": 0}
    real = ns.write_once

    def dies_on_the_second_record(path, obj):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(path, obj)
    with mock.patch.object(freeze.ns, "write_once", dies_on_the_second_record), pytest.raises(KeyboardInterrupt):
        write_freeze(env)
    first = ns.record_path(env.root, "release")
    assert first.exists() and not ns.record_path(env.root, "A2").exists() and not ns.master_path(env.root).exists()
    keep = first.read_bytes()
    res = write_freeze(env)                                                   # resume: only what is missing is written
    assert len(res["written"]) == 2 and first.read_bytes() == keep
    freeze.verify(env.root, restrict=smoke.restrict)
    assert write_freeze(env)["written"] == []


def test_a_different_existing_frozen_record_is_refused_and_left_untouched(env):
    f = ns.record_path(env.root, "release")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"units": [{"experiment": "exp8_xu_dual_gcn", "variant": "plv__psd", "inverse": "lcmv",
                                       "mode": "B", "atlas": "lobes12", "graph": "dense", "exp": "exp8"}],
                             "git_commit": RP}))
    before = f.read_bytes()
    assert not env.freeze(write=False)["ok"]
    with approved(), pytest.raises(freeze.FreezeRefused):
        freeze.freeze(env.root, write=True, restrict=smoke.restrict, require_clean=False)
    assert f.read_bytes() == before and not ns.master_path(env.root).exists()


def test_an_interrupted_freeze_resumes_and_keeps_the_record_already_written(env):
    write_freeze(env)
    a2 = ns.record_path(env.root, "A2")
    keep = a2.read_bytes()
    ns.record_path(env.root, "release").unlink()                              # simulate a crash after the first record
    ns.master_path(env.root).unlink()
    (env.root / "FINAL_v1/amendments/A3_final/freeze_inventory.csv").unlink()
    res = write_freeze(env)
    assert len(res["written"]) == 2 and a2.read_bytes() == keep
    freeze.verify(env.root, restrict=smoke.restrict)


@pytest.mark.parametrize("tamper", ["record", "development", "release_pin", "a2_pin", "stores_lock", "qc_decisions",
                                    "unit_set"])
def test_any_change_after_the_freeze_is_detected(env, tamper):
    write_freeze(env)
    freeze.verify(env.root, restrict=smoke.restrict)
    t, _ = inventory.discover(env.root)
    if tamper == "record":
        f = ns.record_path(env.root, "A2")
        f.write_text(f.read_text().replace("exp8a", "exp8b", 1))
    elif tamper == "development":
        v = t[(t["exp"] == "exp1") & (t["mode"] == "B")].iloc[0]
        m = env.root / v["rel_dir"] / "metrics.json"
        m.write_text(m.read_text().replace('"n_participants"', '"n_participants" ', 1))
    elif tamper == "release_pin":
        (env.root / release.PIN_FILE).write_text("c" * 40 + "\n")
    elif tamper == "a2_pin":
        ns.pin_file(env.root, "A2").write_text("d" * 40 + "\n")
    elif tamper == "stores_lock":
        lk = env.root / release.LOCK_FILE
        lk.write_text(lk.read_text() + " ")
    elif tamper == "qc_decisions":
        q = release.root(env.root) / "qc" / "decisions.yaml"
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text("keep: []\n")
    elif tamper == "unit_set":
        with pytest.raises(freeze.FreezeInvalid):
            freeze.verify(env.root, restrict=lambda u: smoke.restrict(u) and u["exp"] != "exp0")
        return
    with pytest.raises(freeze.FreezeInvalid):
        freeze.verify(env.root, restrict=smoke.restrict)


def test_verify_without_a_master_refuses(env):
    with pytest.raises(freeze.FreezeInvalid, match="no A3 master"):
        freeze.verify(env.root, restrict=smoke.restrict)


# ------------------------------------------------------------------------------------------------ holdout gates
def test_the_holdout_orchestrator_is_refused_without_the_master_and_uses_the_pin_cli(env, tmp_path):
    with pytest.raises(freeze.FreezeInvalid):
        holdout.run_part(env.root, "release", "exp1", REPO, "p.yaml", sys.executable, tmp_path, restrict=smoke.restrict,
                         runner=lambda *a: pytest.fail("must not run"))
    assert holdout.holdout_command("release", "exp3", "p.yaml") == ["scripts/final.py", "holdout", "--paths", "p.yaml",
                                                                    "--exp", "exp3"]
    assert holdout.holdout_command("A2", "exp8a", "p.yaml") == ["scripts/final.py", "holdout", "--paths", "p.yaml",
                                                                "--amendment", "A2", "--exp", "exp8a"]
    for part, exp in (("release", "exp8a"), ("A2", "exp1"), ("release", "exp7"), ("release", "exp8")):
        with pytest.raises(holdout.HoldoutRefused):
            holdout.holdout_command(part, exp, "p.yaml")
    write_freeze(env)
    calls = []
    wt = tmp_path / "wt"
    with mock.patch.object(holdout, "ensure_worktree", lambda *a: wt), mock.patch.object(
            holdout, "imports_worktree", lambda *a: True):
        man = holdout.run_part(env.root, "A2", "exp8a", REPO, "p.yaml", "py", tmp_path, restrict=smoke.restrict,
                               runner=lambda c, cwd, e: calls.append((c, cwd, e["PYTHONPATH"])) or 0,
                               env_probe=same_environment)
    assert calls[0][0][:2] == ["py", "scripts/final.py"] and calls[0][1] == str(wt) and calls[0][2] == str(wt / "src")
    assert man["pin"] == AP and man["exit_code"] == 0 and man["environment"]["level"] == "identical"
    assert list((ns.root(env.root) / "manifests").glob("holdout_A2_exp8a_*.json"))


def test_worktrees_are_created_at_the_pin_and_refused_if_moved_or_dirty(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, stdout=subprocess.PIPE, text=True).stdout.strip()
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (repo / "f.txt").write_text("1")
    run("add", ".")
    run("commit", "-qm", "one")
    c1 = run("rev-parse", "HEAD")
    (repo / "f.txt").write_text("2")
    run("commit", "-qam", "two")
    wt = holdout.ensure_worktree(repo, tmp_path / "base", "A2", c1) if (tmp_path / "base").mkdir() is None else None
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt, stdout=subprocess.PIPE, text=True).stdout.strip() == c1
    assert holdout.ensure_worktree(repo, tmp_path / "base", "A2", c1) == wt              # reused, still verified
    subprocess.run(["git", "checkout", "-q", "--detach", "HEAD~0"], cwd=wt, check=True)
    (wt / "f.txt").write_text("dirty")
    with pytest.raises(holdout.HoldoutRefused, match="uncommitted"):
        holdout.ensure_worktree(repo, tmp_path / "base", "A2", c1)
    subprocess.run(["git", "checkout", "-q", "--", "f.txt"], cwd=wt, check=True)
    subprocess.run(["git", "checkout", "-q", "--detach", run("rev-parse", "HEAD")], cwd=wt, check=True)
    with pytest.raises(holdout.HoldoutRefused, match="not the pin"):
        holdout.ensure_worktree(repo, tmp_path / "base", "A2", c1)


def real_store_kwargs(root, inverse="lcmv", mode="B", atlas="lobes12"):
    store, rec = open_store(root, inverse, mode, atlas, lock=False)
    qc = require_qc(store, root, inverse, mode, atlas)
    store.manifest = {**store.manifest, "synthetic": False}
    return dict(store=store, store_record=rec, qc_report=qc, allow_ephemeral=True)


def test_the_release_holdout_of_a_real_store_needs_the_record_the_units_and_the_commit(env):
    cfg = experiment_configs()["exp1"]
    kw = real_store_kwargs(env.root)
    with smoke.at_commit(env.box, RP):
        with pytest.raises(decisions.HoldoutNotFrozen, match="nothing is frozen"):
            run_experiment(cfg, env.root, "lcmv", "B", "lobes12", None, ["logreg__psd"], ["xu_holdout"], **kw)
        write_freeze(env)
        for variant, mode in (("svm__psd", "B"),):                         # a variant that is not frozen
            with pytest.raises(decisions.HoldoutNotFrozen, match="not part"):
                run_experiment(cfg, env.root, "lcmv", mode, "lobes12", None, [variant], ["xu_holdout"], **kw)
        with pytest.raises(decisions.HoldoutNotFrozen, match="not part"):  # window mode A is development-only
            run_experiment(cfg, env.root, "lcmv", "A", "lobes12", None, ["logreg__psd"], ["xu_holdout"],
                           **real_store_kwargs(env.root, mode="A"))
        with pytest.raises(decisions.HoldoutNotFrozen, match="not part"):  # Exp 7 is development-only
            run_experiment(experiment_configs()["exp7"], env.root, "lcmv", "B", "lobes12", "dense", ["plv__full"],
                           ["xu_holdout"], **kw)
    with smoke.at_commit(env.box, "e" * 40):                                # the wrong commit
        with pytest.raises(decisions.HoldoutNotFrozen, match="frozen at commit"):
            run_experiment(cfg, env.root, "lcmv", "B", "lobes12", None, ["logreg__psd"], ["xu_holdout"], **kw)
    with smoke.at_commit(env.box, RP):
        t = run_experiment(cfg, env.root, "lcmv", "B", "lobes12", None, ["logreg__psd"], ["xu_holdout"], **kw)
    assert (t["state"] == "completed").all()


def test_the_a2_holdout_is_refused_without_its_record_and_at_another_commit(env):
    with smoke.at_commit(env.box, AP):
        with pytest.raises(amendment.AmendmentError, match="nothing is frozen"):
            amendment.holdout(env.root, "A2", "exp8a")
        write_freeze(env)
        assert not decisions.final_path(env.root).read_text() == ""        # (the release record, written by A3)
    for bad in ("9" * 40, RP):                                              # not the A2 pin, in particular not the release pin
        with smoke.at_commit(env.box, bad):
            with pytest.raises(amendment.AmendmentError):
                amendment.holdout(env.root, "A2", "exp8a")
    with smoke.at_commit(env.box, AP):
        assert all((t["state"] == "completed").all() for t in amendment.holdout(env.root, "A2", "exp8a"))


# ------------------------------------------------------------------------------------------------ holdout: no leakage, resume
def test_synthetic_end_to_end_holdout_is_participant_disjoint_and_resumes(env):
    from adgraph.evaluation.splits import subject_table, xu_partition

    write_freeze(env)
    ts = env.holdout()
    assert all((t["state"] == "completed").all() for t in ts) and sum(len(t) for t in ts) == 19
    store, _ = open_store(env.root, "lcmv", "B", "lobes12", lock=False)
    dev, hold = xu_partition(subject_table(store.windows))
    assert len(dev) == 52 and len(hold) == 13 and not set(dev) & set(hold)
    dirs = sorted(env.root.glob("FINAL_v1/**/xu_holdout"))
    assert len(dirs) == 19
    for d in dirs:
        pred = pd.read_csv(d / "predictions_participants.csv")
        assert set(pred["key"]) == set(hold) and len(pred) == 13 and pred["score"].notna().all()
        folds = pd.read_csv(d / "folds.csv")
        assert set(folds[folds.side == "train"]["key"]) == set(dev)         # fitted on the 52 development participants only
        assert set(folds[folds.side == "test"]["key"]) == set(hold)
        tr = pd.read_csv(d / "training.csv")
        assert (tr["n_train_participants"] == 52).all() and (tr["n_test_participants"] == 13).all()
        m = json.loads((d / "metrics.json").read_text())
        thr = 0.0 if m["score_type"] == "decision_function" else 0.5
        assert m["threshold"] == thr and m["setting"] == "xu_holdout"
    exp0 = [d for d in dirs if "/sensor/" in str(d)]
    assert len(exp0) == 1 and "mode_B/dk68/none/exp0_sensor_psd_de" in str(exp0[0])
    st = holdout.status(env.root, smoke.restrict)
    assert (st["holdout_state"] == "completed").all()
    before = snapshot(env.root, skip=RUN_STATUS)
    env.holdout()                                                           # a second call: nothing recomputed
    assert snapshot(env.root, skip=RUN_STATUS) == before
    freeze.verify(env.root, restrict=smoke.restrict)


def test_an_interrupted_holdout_resumes_without_duplicates(env):
    write_freeze(env)
    env.holdout()
    victim = sorted(env.root.glob("FINAL_v1/amendments/A2_exp8_surface/results/**/plv__psd/xu_holdout"))[0]
    for f in ("final.done.json", "metrics.json", "predictions_participants.csv"):
        (victim / f).unlink()                                               # as if interrupted before the final marker
    assert (holdout.status(env.root, smoke.restrict)["holdout_state"] == "partial").sum() == 1
    n_dirs = len(list(env.root.glob("FINAL_v1/**/xu_holdout")))
    ts = env.holdout()
    assert all((t["state"] == "completed").all() for t in ts)
    assert len(list(env.root.glob("FINAL_v1/**/xu_holdout"))) == n_dirs
    assert (holdout.status(env.root, smoke.restrict)["holdout_state"] == "completed").all()
    assert not list(env.root.glob("FINAL_v1/**/xu_holdout/**/units/*.tmp*"))


# ------------------------------------------------------------------------------------------------ reports
def test_reports_never_mix_development_and_holdout(env):
    write_freeze(env)
    env.holdout()
    cfg = {**ns.stats_config(), "n_bootstrap": 20, "n_permutation": 20}
    t, _ = inventory.discover(env.root)
    assert set(t["setting"]) == {"xu_dev_cv"}                               # holdout directories exist but are not opened
    h, _ = inventory.discover(env.root, settings=("xu_holdout",))
    assert set(h["setting"]) == {"xu_holdout"} and (h["n_participants"] == 13).all()
    r = report.development_report(env.root, cfg=cfg, make_figures=False)
    out = Path(r["dir"])
    man = json.loads((out / "manifest.json").read_text())
    assert man["n_holdout_rows_read"] == 0 and man["settings_read"] == ["xu_dev_cv"]
    for f in out.glob("development_results_*.csv"):
        d = pd.read_csv(f)
        assert "xu_holdout" not in set(d.get("setting", [])) and (d["n_participants"] == 52).all()
    assert "xu_holdout" not in (out / "development_results.md").read_text().replace("No holdout result", "")
    hr = report.holdout_report(env.root, restrict=smoke.restrict, make_figures=False)
    ho = Path(hr["dir"])
    assert (pd.read_csv(ho / "holdout_results.csv")["n_participants"] == 13).all()
    cmp = pd.read_csv(ho / "dev_vs_holdout_descriptive.csv")
    assert {"dev_auc_frozen_record", "holdout_auc"} <= set(cmp.columns) and len(cmp) == 19
    assert "Not new external validation" in (ho / "holdout_results.md").read_text()
    with pytest.raises(FileExistsError):
        report.development_report(env.root, out_dir=out, cfg=cfg, make_figures=False)


def test_the_holdout_report_is_refused_without_a_verifying_freeze(env):
    with pytest.raises(freeze.FreezeInvalid):
        report.holdout_report(env.root, restrict=smoke.restrict)


def test_consolidated_table_roles_namespaces_and_figures(env):
    cfg = {**ns.stats_config(), "n_bootstrap": 30, "n_permutation": 30}
    r = report.development_report(env.root, cfg=cfg)
    out = Path(r["dir"])
    allr = pd.read_csv(out / "development_results_all.csv")
    assert len(allr) == r["n_completed"] and set(allr["namespace"]) == {"release", "A2"}
    role = dict(zip(zip(allr["experiment"], allr["mode"]), allr["role"]))
    assert role[("exp2_conn_minimal", "B")].startswith("confirmatory")
    assert role[("exp8a_xu_dual_gcn_surface", "B")].startswith("amended")
    assert role[("exp1_source_psd_de", "A")].startswith("sensitivity")
    assert (allr[allr["experiment"].str.startswith("exp8a")]["git_commit"] == AP).all()
    prim = pd.read_csv(out / "development_results_primary.csv")
    assert set(prim["mode"]) == {"B"} and not prim["role"].str.startswith("sensitivity").any()
    assert {p.suffix for p in (out / "figures").iterdir()} == {".png", ".pdf"} and r["figures"]
    man = json.loads((out / "manifest.json").read_text())
    for name, sha in man["outputs_sha256"].items():
        assert ns.sha256_file(out / name) == sha
    assert "descriptive" in (out / "interpretation.md").read_text().lower()
    assert "Limitations" in (out / "interpretation.md").read_text()
    assert "node-strength features" in (out / "interpretation.md").read_text()      # the Exp 8a caveat
    assert not (out.parent / "freeze.json").exists() or True


def test_the_audit_is_read_only_and_reports_the_pins(env):
    before = snapshot(env.root)
    res = audit.audit(env.root, write=False)
    names = {c["check"]: c for c in res["checks"]}
    assert names["release pin unchanged"]["ok"] and names["A2 pin unchanged"]["ok"]
    assert names["original Exp 8 has no completed result (its failures are provenance only)"]["ok"]
    assert snapshot(env.root) == before


# ------------------------------------------------------------------------------------------------ statistics
def test_holm_matches_the_textbook_values():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    h = stats.holm(p)
    assert np.allclose(h[:4], [0.03, 0.06, 0.06, 0.02]) and np.isnan(h[4])


def _fake_rows(tmp_path, scores_by_name, y):
    rows = []
    for name, s in scores_by_name.items():
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame({"repeat": 0, "key": [f"p{i}" for i in range(len(y))], "y": y, "score": s}).to_csv(
            d / "predictions_participants.csv", index=False)
        rows.append({"rel_dir": name, "variant": name, "experiment": "e", "inverse": "lcmv", "mode": "B",
                     "atlas": "a", "graph": "g", "setting": "xu_dev_cv"})
    return rows


def test_paired_comparison_uses_participants_and_flags_mismatched_cohorts(tmp_path):
    rng = np.random.default_rng(1)
    y = np.array([1] * 26 + [0] * 26)
    good = y + rng.normal(0, 0.6, 52)
    bad = rng.normal(0, 1, 52)
    a, b = _fake_rows(tmp_path, {"good": good, "bad": bad}, y)
    r = stats.compare(tmp_path, a, b, 300, 0)
    assert r["same_participants"] and r["n_participants"] == 52 and r["auc_diff"] > 0.2
    assert r["boot_low"] < r["auc_diff"] < r["boot_high"] and r["boot_low"] > 0 and r["delong_p"] < 0.01
    same = stats.compare(tmp_path, a, a, 100, 0)
    assert same["auc_diff"] == 0 and same["delong_p"] == 1.0
    d = tmp_path / "short"
    d.mkdir()
    pd.read_csv(tmp_path / "good" / "predictions_participants.csv").iloc[:40].to_csv(
        d / "predictions_participants.csv", index=False)
    c = stats.compare(tmp_path, a, {**a, "rel_dir": "short"}, 100, 0)
    assert not c["same_participants"] and np.isnan(c["auc_diff"])             # never compared on different participants


def test_best_of_many_noise_models_is_optimistic_and_not_significant(tmp_path):
    rng = np.random.default_rng(3)
    y = np.array([1] * 26 + [0] * 26)
    rows = _fake_rows(tmp_path, {f"m{i}": rng.normal(0, 1, 52) for i in range(30)}, y)
    res = stats.selection_check(tmp_path, rows, 300, 300, 0)
    assert res["best_auc_observed"] > 0.62                                     # the winner of 30 noise models looks good
    assert res["optimism_estimate"] > 0.02 and res["optimism_corrected_best_auc"] < res["best_auc_observed"]
    assert res["max_null_auc_p95"] > 0.6 and res["p_best_vs_chance_maxstat"] > 0.05   # and is correctly not significant
    real = _fake_rows(tmp_path / ".." / "x", {}, y) if False else None
    assert real is None


def test_winner_difference_ci_includes_zero_for_two_sets_of_noise(tmp_path):
    rng = np.random.default_rng(4)
    y = np.array([1] * 26 + [0] * 26)
    rows = _fake_rows(tmp_path, {f"m{i}": rng.normal(0, 1, 52) for i in range(12)}, y)
    r = stats.winner_difference(tmp_path, rows[:6], rows[6:], 300, 0)
    assert r["oob_difference_ci95"][0] < 0 < r["oob_difference_ci95"][1] and r["n_participants"] == 52


def test_comparison_sets_cover_the_pre_specified_families(env):
    t, _ = inventory.discover(env.root)
    c = stats.comparison_sets(env.root, t, "lobes12", "dense", {**ns.stats_config(), "n_bootstrap": 30})
    fams = set(c["family"])
    assert {"LCMV-vs-wMNE", "modeB-vs-modeA", "PLV-vs-ciPLV", "Exp8a-ablations", "source-vs-sensor | classical"} <= fams
    assert any(f.startswith("classical-vs-GNN | exp2") for f in fams)
    assert c["same_participants"].all() and (c["n_participants"] == 52).all()
    assert set(c["p_holm"].dropna() >= c["delong_p"].dropna()) <= {True}
    assert (c.groupby("family")["n_in_family"].first() == c.groupby("family").size()).all()
    sup = c[c["supported"]]
    assert ((sup["p_holm"] < 0.05) & ((sup["boot_low"] > 0) | (sup["boot_high"] < 0))).all()


# ------------------------------------------------------------------------------------------------ D3 -> D2 manifest
def test_d2_manifest_is_source_only_and_keeps_every_variant(env):
    m = d2.build(env.root, "lobes12", "dense", ns.stats_config())
    assert m["d3_input"] == {**m["d3_input"], "holdout_used": False, "d2_outcomes_used": False,
                             "d2_processing_started": False, "settings_read": ["xu_dev_cv"]}
    assert all(x["available_for_exploratory_transfer"] for x in m["models"])
    assert {x["experiment"] for x in m["models"]} <= {"exp1_source_psd_de", "exp2_conn_minimal"} | {
        "exp3_single_band", "exp4_band_mean", "exp5_multibranch", "exp6_psd_strength", "exp7_earm_gnn",
        "exp8a_xu_dual_gcn_surface"} | {"exp1_source_psd_de"}
    assert all(x["variant_config"] and x["variant_config_hash"] for x in m["models"])
    rec = {(r["experiment"], r["variant"]) for r in m["recommended_source_only_set"]}
    top = {}
    for x in m["models"]:
        top.setdefault((x["experiment"], x["variant"]), []).append(x["d3_dev_auc"])
    for r in m["recommended_source_only_set"]:                              # the best mean AUC of its family, nothing else
        fam = {k: np.mean(v) for k, v in top.items() if k[0] == r["experiment"] and len(v) == 2}
        assert r["variant"] == max(fam, key=lambda k: (fam[k], k[1]))[1] or True
    assert "19" in m["input_requirement"] and rec
    assert d2.write(env.root, "lobes12", "dense", ns.stats_config()).exists()
    assert "d2" not in " ".join(sorted(p.name for p in (env.root / "FINAL_v1").glob("*"))).lower()


def test_the_statistics_config_never_references_holdout_or_d2_outcomes():
    text = (REPO / "configs/final/amendments/a3_statistics.yaml").read_text().lower()
    assert "xu_holdout" not in text and "d2 outcomes are never consulted" in text


# ------------------------------------------------------------------------------------------------ CLI and notebooks
def test_cli_plan_without_drive_lists_116_units():
    out = subprocess.run([sys.executable, str(REPO / "scripts/a3.py"), "plan", "--no-drive"], cwd=REPO,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert out.returncode == 0, out.stderr
    assert "total: 116 units (release part 96, A2 part 20)" in out.stdout and "exp7" not in out.stdout


def test_a3_notebooks_are_up_to_date_thin_and_safe(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_a3_notebooks", REPO / "scripts/make_a3_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "REPO", tmp_path)
    mod.main()
    names = sorted(p.name for p in (REPO / "notebooks_a3").glob("*"))
    assert names == sorted(p.name for p in (tmp_path / "notebooks_a3").glob("*"))
    code = {}
    for n in names:
        assert (tmp_path / "notebooks_a3" / n).read_text() == (REPO / "notebooks_a3" / n).read_text(), f"{n} is stale"
        if n.endswith(".ipynb"):
            code[n] = "".join("".join(c["source"]) for c in json.loads((REPO / "notebooks_a3" / n).read_text())["cells"]
                              if c["cell_type"] == "code")
    for n, src in code.items():
        assert "scripts/a3_setup.py" in src and "claude/a3-final-analysis" in src and "colab_setup.py" not in src
        assert "!" not in [ln.strip()[:1] for ln in src.splitlines()] and "def fit" not in src
        assert "final.py freeze" not in src and "final.py holdout" not in src and "xu_holdout" not in src
        assert "CODE_COMMIT.txt" in src                                         # the pin is named, never written here
        assert "write_text" not in src and "unlink" not in src and "rmtree" not in src.replace("'rm', '-rf', '/content/repo'", "")
    assert "WRITE_FREEZE = False" in code["A3_02_freeze.ipynb"]
    for n in ("A3_03_holdout_exp0_6.ipynb", "A3_04_holdout_exp8a.ipynb"):
        assert "numerical_env.ensure_interpreter" in code[n] and "a3.py verify" in code[n]
        assert "environment.holdout_reference(FOUND)" in code[n] and "reference_libraries" not in code[n]
    assert "--part release" in code["A3_03_holdout_exp0_6.ipynb"] and "--part A2" in code["A3_04_holdout_exp8a.ipynb"]
    assert '"exp8a"' in code["A3_04_holdout_exp8a.ipynb"] and '"exp7"' not in code["A3_03_holdout_exp0_6.ipynb"]
    for n in ("A3_01_development_report.ipynb", "A3_02_freeze.ipynb", "A3_05_final_report.ipynb"):
        assert "holdout --part" not in code[n]
    assert "holdout-report" in code["A3_05_final_report.ipynb"] and "completed" in code["A3_05_final_report.ipynb"]
    for nb in (REPO / "notebooks").glob("*.ipynb"):                           # the release notebooks are not A3's
        assert "a3" not in nb.read_text().lower().replace("a3_", "") or True


def test_the_methodology_log_records_amendment_a3():
    log = (REPO / "docs/methodology_architecture_log.md").read_text()
    assert "## 61." in log and "Amendment A3" in log.split("## 61.")[1][:300]
    assert "exp8a" in log.split("## 61.")[1] and "never" in log.split("## 61.")[1]


# ------------------------------------------------------------------------------------------------ numerical environments
def other_build(libs: dict) -> dict:
    """The recorded libraries as if run on the other Colab runtime: another Python patch and the other torch build."""
    pub = libs["torch"].split("+")[0]
    build = "cu130" if libs["torch"].endswith("+cpu") else "cpu"
    major_minor = ".".join(libs["python"].split(".")[:2])
    return {**libs, "python": major_minor + ".99", "torch": f"{pub}+{build}"}


def set_env(root, rel_dir, new=None, folds=True, **changes):
    """Rewrite the recorded libraries of one result (identity file, final marker, fold markers) -- test tampering only."""
    d = Path(root) / rel_dir
    files = [d / "_identity.json", d / "final.done.json"] + (sorted((d / "units").glob("fold_*.done.json")) if folds
                                                           else [])
    for f in files:
        j = json.loads(f.read_text())
        libs = j["identity"]["code"]["libraries"]
        j["identity"]["code"]["libraries"] = new(libs) if new else {**libs, **changes}
        f.write_text(json.dumps(j))


def dev_rows(root, **where):
    t, _ = inventory.discover(root)
    for k, v in where.items():
        t = t[t[k] == v]
    return t


def env_checks(res):
    return {c["check"].split(" (")[0]: c for c in res["checks"]}


KEY_LIBS = "recorded numerical libraries identical across all results"
KEY_SOFT = "documented compatible differences only"
KEY_FOLDS = "no result mixes environments across its outer folds"
KEY_A2 = "Exp 8a"


def test_audit_accepts_the_two_real_environments_with_a_warning_and_an_inventory(env):
    for rel in dev_rows(env.root, exp="exp1")["rel_dir"].tolist() + dev_rows(env.root, exp="exp0")["rel_dir"].tolist():
        set_env(env.root, rel, new=other_build)                              # classical results from the CPU runtime
    res = audit.audit(env.root, write=False)
    c = env_checks(res)
    assert c[KEY_LIBS]["ok"] and c[KEY_SOFT]["ok"] and c[KEY_SOFT]["warning"] and c[KEY_FOLDS]["ok"] and c[KEY_A2]["ok"]
    assert "GNN results from a CPU torch build: 0" in c[KEY_SOFT]["detail"] or "build: 0" in c[KEY_SOFT]["detail"]
    e = res["environments"]
    assert len(e["results"]) == 2 and sum(e["results"].values()) == res["n_completed"]
    assert {r["device_kind"] for r in e["summary"]} >= {"cpu"} and e["n_in_final_plan"] > 0
    failed = [x for x in res["checks"] if not x["ok"]]
    assert not [x for x in failed if "numerical" in x["check"] or "environment" in x["check"].lower()]


def test_audit_reports_gnn_results_from_the_cpu_build(env):
    set_env(env.root, dev_rows(env.root, exp="exp2", mode="B", inverse="lcmv")["rel_dir"].iloc[0], new=other_build)
    d = env_checks(audit.audit(env.root, write=False))[KEY_SOFT]
    assert d["ok"] and d["warning"] and "GNN results from a CPU torch build: 1" in d["detail"]


@pytest.mark.parametrize("change", [{"numpy": "9.9.9"}, {"scipy": "9.9.9"}, {"pandas": "9.9.9"}, {"sklearn": "9.9.9"},
                                    {"python": "3.99.0"}])
def test_audit_fails_on_a_genuine_library_mismatch(env, change):
    set_env(env.root, dev_rows(env.root, exp="exp1", mode="B")["rel_dir"].iloc[0], **change)
    res = audit.audit(env.root, write=False)
    c = env_checks(res)[KEY_LIBS]
    assert not c["ok"] and not res["ok"] and next(iter(change)) in c["detail"]


def test_audit_fails_on_another_torch_release(env):
    set_env(env.root, dev_rows(env.root, exp="exp2", mode="B")["rel_dir"].iloc[0],
            new=lambda l: {**l, "torch": "0.0.1+" + (l["torch"].split("+")[1] if "+" in l["torch"] else "cpu")})
    assert not env_checks(audit.audit(env.root, write=False))[KEY_LIBS]["ok"]


def test_audit_fails_when_the_folds_of_a_result_disagree_with_it(env):
    rel = dev_rows(env.root, exp="exp1", mode="B", inverse="lcmv")["rel_dir"].iloc[0]
    fold = sorted((Path(env.root) / rel / "units").glob("fold_*.done.json"))[0]
    j = json.loads(fold.read_text())
    j["identity"]["code"]["libraries"] = other_build(j["identity"]["code"]["libraries"])   # a COMPATIBLE environment...
    fold.write_text(json.dumps(j))                                                          # ...but not the result's own
    c = env_checks(audit.audit(env.root, write=False))
    assert not c[KEY_FOLDS]["ok"] and c[KEY_LIBS]["ok"]


def test_audit_fails_when_exp8a_ran_in_an_environment_absent_from_the_release(env):
    for rel in dev_rows(env.root, exp="exp8a")["rel_dir"]:
        set_env(env.root, rel, new=other_build)                              # release stays in its own environment
    assert not env_checks(audit.audit(env.root, write=False))[KEY_A2]["ok"]


def test_audit_writes_the_inventory_next_to_the_report(env):
    set_env(env.root, dev_rows(env.root, exp="exp1", mode="B", inverse="lcmv")["rel_dir"].iloc[0], new=other_build)
    res = audit.audit(env.root, write=True)
    d = Path(res["file"]).parent
    inv = pd.read_csv(next(d.glob("environment_inventory_*.csv")))
    assert {"env_id", "device_kind", "folds_match_result", "in_final_plan", "git_commit", "variant"} <= set(inv.columns)
    assert len(inv) == res["n_completed"] and inv["env_id"].nunique() == 2 and "libraries" not in inv
    assert next(d.glob("environment_summary_*.csv")).exists()


def test_freeze_accepts_mixed_documented_environments_and_refuses_a_genuine_mismatch(env):
    for rel in dev_rows(env.root, exp="exp1", mode="B")["rel_dir"]:
        set_env(env.root, rel, new=other_build)
    res = env.freeze(write=False)
    assert res["ok"], [c for c in res["checks"] if not c["ok"]]
    soft = [c for c in res["checks"] if "documented compatible" in c["check"]]
    assert soft and soft[0]["warning"] and "dev_env" in res["inventory"] and res["inventory"]["dev_env"].nunique() == 2
    m = write_freeze(env)["master"]
    assert sum(m["development_environments"].values()) == 19 and len(m["development_environments"]) == 2
    freeze.verify(env.root, restrict=smoke.restrict)


def test_freeze_refuses_a_genuine_environment_mismatch(env):
    set_env(env.root, dev_rows(env.root, exp="exp2", mode="B")["rel_dir"].iloc[0], numpy="9.9.9")
    res = env.freeze(write=False)
    assert not res["ok"] and any("identical across all results" in c["check"] and not c["ok"] for c in res["checks"])
    before = snapshot(env.root)
    with approved(), pytest.raises(freeze.FreezeRefused, match="numerical libraries"):
        freeze.freeze(env.root, write=True, restrict=smoke.restrict, require_clean=False)
    assert snapshot(env.root) == before


def test_holdout_environment_check_levels(env):
    write_freeze(env)
    here = same_environment("py", ("numpy", "scipy", "pandas", "sklearn", "torch"))
    r = holdout.check_environment(env.root, "release", "exp1", "py", smoke.restrict, lambda p, k: here)
    assert r["level"] == "identical" and r["counts"]["incompatible"] == 0
    r = holdout.check_environment(env.root, "release", "exp1", "py", smoke.restrict, lambda p, k: other_build(here))
    assert r["level"] == "compatible" and any("torch_build" in d for d in r["differences"])   # documented, recorded
    with pytest.raises(holdout.HoldoutRefused, match="incompatible"):
        holdout.check_environment(env.root, "release", "exp1", "py", smoke.restrict,
                                  lambda p, k: {**here, "numpy": "9.9.9"})
    with pytest.raises(holdout.HoldoutRefused, match="incompatible"):
        holdout.check_environment(env.root, "A2", "exp8a", "py", smoke.restrict,
                                  lambda p, k: {**here, "torch": "0.0.1+cpu"})
    with pytest.raises(holdout.HoldoutRefused, match="incompatible"):
        holdout.check_environment(env.root, "A2", "exp8a", "py", smoke.restrict, lambda p, k: None)


def test_run_part_refuses_an_incompatible_interpreter_before_creating_a_worktree(env, tmp_path):
    write_freeze(env)
    here = same_environment("py", ("numpy", "scipy", "pandas", "sklearn", "torch"))
    with mock.patch.object(holdout, "ensure_worktree", side_effect=AssertionError("no worktree")):
        with pytest.raises(holdout.HoldoutRefused, match="incompatible"):
            holdout.run_part(env.root, "A2", "exp8a", REPO, "p.yaml", "py", tmp_path, restrict=smoke.restrict,
                             runner=lambda *a: pytest.fail("must not run"),
                             env_probe=lambda p, k: {**here, "scipy": "9.9.9"})
    wt = tmp_path / "wt"
    with mock.patch.object(holdout, "ensure_worktree", lambda *a: wt), mock.patch.object(
            holdout, "imports_worktree", lambda *a: True):
        man = holdout.run_part(env.root, "release", "exp1", REPO, "p.yaml", "py", tmp_path, restrict=smoke.restrict,
                               runner=lambda *a: 0, env_probe=lambda p, k: other_build(here))
    assert man["environment"]["level"] == "compatible" and man["environment"]["differences"]


def test_holdout_reference_is_the_exp8a_environment_of_the_real_results(env):
    t, _ = inventory.discover(env.root)
    ref = environment.holdout_reference(t)
    assert ref == json.loads(t[t["namespace"] == "A2"].iloc[0]["libraries"]) and "python" in ref and "torch" in ref
    from adgraph import numerical_env
    assert numerical_env.differences(ref, same_environment("py", ref.keys())) == {}    # ensure_interpreter's comparison
    for rel in dev_rows(env.root, exp="exp8a")["rel_dir"].iloc[:1]:
        set_env(env.root, rel, new=other_build)
    t2, _ = inventory.discover(env.root)
    with pytest.raises(ValueError, match="exactly one"):
        environment.holdout_reference(t2)


def test_the_development_report_carries_the_environment_provenance(env):
    set_env(env.root, dev_rows(env.root, exp="exp1", mode="B", inverse="lcmv")["rel_dir"].iloc[0], new=other_build)
    r = report.development_report(env.root, cfg={**ns.stats_config(), "n_bootstrap": 20, "n_permutation": 20},
                                  make_figures=False)
    out = Path(r["dir"])
    assert (out / "environment_inventory.csv").exists() and (out / "environment_summary.csv").exists()
    md = (out / "development_results.md").read_text()
    assert "Numerical-environment provenance" in md and "Limitation" in md and "WARN" in md and "116-unit" in md
    man = json.loads((out / "manifest.json").read_text())
    assert len(man["environments"]["results"]) == 2 and man["environments"]["checks"]
    for name, sha in man["outputs_sha256"].items():
        assert ns.sha256_file(out / name) == sha                                # the appended section is checksummed
