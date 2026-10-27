"""A3 numerical-environment policy (``a3.environment``): identical / documented-compatible / incompatible, per-unit and
per-fold checks, the holdout reference and the interpreter check. Pure logic, no training. The two environments of the
real release results are used verbatim."""

import json

import pandas as pd
import pytest

from adgraph.a3 import environment as env

COMMON = {"numpy": "2.1.3", "scipy": "1.16.3", "pandas": "2.2.3", "sklearn": "1.6.1"}
GPU = {"python": "3.13.15", **COMMON, "torch": "2.11.0+cu130"}       # release (most) and Exp 8a
CPU = {"python": "3.13.16", **COMMON, "torch": "2.11.0+cpu"}          # the second release environment


def found(rows):
    """A table shaped like inventory.discover for the checks: (namespace, family, exp, libs)."""
    return pd.DataFrame([{"namespace": ns, "family": exp.capitalize(), "exp": exp, "experiment": exp, "variant": f"v{i}",
                          "inverse": "lcmv", "mode": "B", "atlas": "lobes12", "graph": "dense", "setting": "xu_dev_cv",
                          "git_commit": "c", "libraries": json.dumps(libs), "rel_dir": f"d{i}"}
                         for i, (ns, exp, libs) in enumerate(rows)])


def inv(rows):
    return env.inventory("/nonexistent", found(rows), None, check_folds=False)


def by_name(checks):
    return {c["check"].split(" (")[0]: c for c in checks}


def test_the_real_pair_is_compatible_and_says_why():
    c = env.compatibility(GPU, CPU)
    assert c["level"] == "compatible" and c["core_differences"] == {}
    assert set(c["differences"]) == {"python_patch", "torch_build"}
    assert env.compatibility(GPU, dict(GPU))["level"] == "identical"
    assert env.parse(GPU)["device_kind"] == "cuda" and env.parse(CPU)["device_kind"] == "cpu"
    assert env.env_id(CPU) == "py3.13.16|torch2.11.0+cpu|numpy2.1.3|scipy1.16.3|pandas2.2.3|sklearn1.6.1"
    assert env.env_id(CPU) != env.env_id({**CPU, "numpy": "2.2.0"})


@pytest.mark.parametrize("change", [{"numpy": "2.2.0"}, {"scipy": "1.17.0"}, {"pandas": "2.3.0"}, {"sklearn": "1.7.0"},
                                    {"torch": "2.12.0+cpu"}, {"python": "3.14.0"}, {"mne": "1.13.2"}])
def test_any_other_difference_is_incompatible(change):
    other = {**CPU, **change}
    c = env.compatibility(GPU, other)
    assert c["level"] == "incompatible" and c["core_differences"]


def test_the_two_real_environments_pass_with_a_warning_when_only_classical_models_used_the_cpu_build():
    rows = [("release", "exp1", CPU)] * 8 + [("release", "exp0", CPU)] * 4 + [("release", "exp5", GPU)] * 5 \
        + [("A2", "exp8a", GPU)] * 3
    checks = env.checks(inv(rows))
    named = by_name(checks)
    assert all(c["ok"] for c in checks)                                    # no failure
    w = named["documented compatible differences only"]
    assert w["warning"] and "GNN results from a CPU torch build: 0" in w["detail"]
    assert named["recorded numerical libraries identical across all results"]["detail"].startswith("2 environment(s)")


def test_gnn_results_from_the_cpu_build_are_reported_not_hidden():
    rows = [("release", "exp2", CPU)] * 2 + [("release", "exp1", GPU)] + [("A2", "exp8a", GPU)]
    w = by_name(env.checks(inv(rows)))["documented compatible differences only"]
    assert w["ok"] and w["warning"] and "GNN results from a CPU torch build: 2" in w["detail"] and "Exp2" in w["detail"]


def test_a_single_environment_passes_cleanly():
    checks = env.checks(inv([("release", "exp1", GPU), ("A2", "exp8a", GPU)]))
    assert all(c["ok"] and not c["warning"] for c in checks)


def test_a_genuine_library_mismatch_fails():
    checks = env.checks(inv([("release", "exp1", GPU), ("release", "exp2", {**GPU, "numpy": "2.2.0"}),
                             ("A2", "exp8a", GPU)]))
    c = by_name(checks)["recorded numerical libraries identical across all results"]
    assert not c["ok"] and "numpy" in c["detail"]


def test_exp8a_must_use_an_environment_that_also_occurs_in_the_release():
    third = {**GPU, "python": "3.13.17"}                                   # compatible, but not any release environment
    c = by_name(env.checks(inv([("release", "exp1", GPU), ("release", "exp2", CPU), ("A2", "exp8a", third)])))
    assert not c["Exp 8a"]["ok"]
    ok = by_name(env.checks(inv([("release", "exp1", GPU), ("release", "exp2", CPU), ("A2", "exp8a", GPU)])))
    assert ok["Exp 8a"]["ok"]


def test_a_result_whose_folds_disagree_with_it_fails(tmp_path):
    d = tmp_path / "r"
    (d / "units").mkdir(parents=True)
    for i, libs in enumerate((GPU, GPU, CPU)):
        (d / "units" / f"fold_r000_f0{i}.done.json").write_text(json.dumps({"identity": {"code": {"libraries": libs}}}))
    assert len(env.fold_environments(d)) == 2
    f = found([("release", "exp1", GPU)])
    f["rel_dir"] = "r"
    i = env.inventory(tmp_path, f, None)
    assert not i.iloc[0]["folds_match_result"]
    assert not by_name(env.checks(i))["no result mixes environments across its outer folds"]["ok"]
    for k in range(3):                                                     # all folds as recorded: consistent
        (d / "units" / f"fold_r000_f0{k}.done.json").write_text(json.dumps({"identity": {"code": {"libraries": GPU}}}))
    assert env.inventory(tmp_path, f, None).iloc[0]["folds_match_result"]


def test_320_results_versus_the_116_unit_plan_are_distinguished():
    rows = [("release", "exp1", GPU)] * 3 + [("A2", "exp8a", GPU)]
    i = env.inventory("/x", found(rows), {("release", "exp1", "v0", "lcmv", "B", "lobes12", "dense")}, check_folds=False)
    assert i["in_final_plan"].sum() == 1 and len(i) == 4
    s = env.summary(i)
    assert s["n_results"].sum() == 4 and s["n_in_final_plan"].sum() == 1


def test_the_holdout_reference_is_the_single_exp8a_environment():
    f = found([("release", "exp1", CPU), ("release", "exp2", GPU), ("A2", "exp8a", GPU), ("A2", "exp8a", GPU)])
    assert env.holdout_reference(f) == GPU
    with pytest.raises(ValueError, match="exactly one"):
        env.holdout_reference(found([("A2", "exp8a", GPU), ("A2", "exp8a", CPU)]))
    with pytest.raises(ValueError, match="exactly one"):
        env.holdout_reference(found([("release", "exp1", GPU)]))


def test_the_interpreter_check_accepts_documented_differences_and_refuses_the_rest():
    units = [json.dumps(GPU), json.dumps(CPU)]
    same = env.interpreter_check(GPU, [json.dumps(GPU)])
    assert same["level"] == "identical"
    mixed = env.interpreter_check(GPU, units)                              # CPU-build units refit in the GPU environment
    assert mixed["level"] == "compatible" and mixed["counts"] == {"identical": 1, "compatible": 1, "incompatible": 0}
    assert any("torch_build" in d for d in mixed["differences"])
    bad = env.interpreter_check({**GPU, "numpy": "2.2.0"}, units)
    assert bad["level"] == "incompatible" and bad["counts"]["incompatible"] == 2
    assert env.interpreter_check(None, units)["level"] == "incompatible"
