"""The lobes12 (rule-recommended) and r18 (exploratory, post-result approval) arms (methodology log entry 59b):
authorization of a reconstruction, the approval record and its validation, the unchanged rule, atlas labels, the
three-atlas comparison with missing results, matching / QC gates, code pinning and holdout isolation. Synthetic
stores and records only; nothing is reconstructed."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from adgraph import release
from adgraph.experiments import decisions
from adgraph.experiments.config import experiment_configs
from adgraph.experiments.plan import run_experiment
from adgraph.stores import open_store, store_dir, verify_store

REPO = Path(__file__).resolve().parents[1]
EXACT = "fsaverage BEM fsaverage-5120-5120-5120-bem-sol.fif (as the stores)"
NOTEBOOK02 = {  # the exact-BEM result reported by the author (authoritative)
    "r18": {"wmne": (0.511, 0.889), "lcmv": (0.536, 0.778)},
    "lobes12": {"wmne": (0.377, 1.000), "lcmv": (0.415, 0.833)},
}


def _resolution(processed, forward_model=EXACT, stamp="20261019_120000", checks_ok=True):
    """notebook 02's saved outputs: the rule's verdicts for the reported numbers, and passed anatomical checks."""
    from adgraph.source.resolution import recommend_atlas

    rule = yaml.safe_load((REPO / "configs" / "final" / "selection.yaml").read_text())["atlas_resolution"]
    rows = [{"atlas": "dk68", "inverse": i, "redundant_signals": 50, "crosstalk_fraction_median": 0.85,
             "rois_dominated_by_own_region": 0.55, "n_regions": 68} for i in ("lcmv", "wmne")]
    rows += [{"atlas": "dk34_gyral", "inverse": i, "redundant_signals": 16, "crosstalk_fraction_median": 0.72,
              "rois_dominated_by_own_region": 0.74, "n_regions": 34} for i in ("lcmv", "wmne")]
    for a, by in NOTEBOOK02.items():
        rows += [{"atlas": a, "inverse": i, "redundant_signals": 0, "crosstalk_fraction_median": c,
                  "rois_dominated_by_own_region": d, "n_regions": int(a.strip("rlobes"))} for i, (c, d) in by.items()]
    rec = recommend_atlas(pd.DataFrame(rows), rule)
    d = release.root(processed) / "parcellation" / f"resolution_{stamp}"
    d.mkdir(parents=True)
    (d / "recommendation.json").write_text(json.dumps({**rec, "forward_model": forward_model}))
    for a in ("lobes12", "r18"):
        c = release.root(processed) / "parcellation" / f"atlas_{a}_{stamp}"
        c.mkdir(parents=True)
        (c / "checks.json").write_text(json.dumps({"partition": {"passed": checks_ok, "detail": ""}}))
    return d / "recommendation.json"


# ------------------------------------------------------------------------------------------- the rule, untouched
def test_the_rule_and_its_verdict_are_unchanged(tmp_path):
    rule = yaml.safe_load((REPO / "configs" / "final" / "selection.yaml").read_text())["atlas_resolution"]
    assert (rule["max_median_crosstalk"], rule["min_own_region_dominance"], rule["require_no_forced_redundancy"]) \
        == (0.60, 0.80, True)
    rec_file = _resolution(tmp_path)
    rec = json.loads(rec_file.read_text())
    assert rec["recommended"] == "lobes12"
    r18 = next(v for v in rec["verdicts"] if v["atlas"] == "r18")
    assert not r18["eligible"] and "own-region dominance 0.78 < 0.8" in r18["reasons"][0]
    before = rec_file.read_bytes()
    decisions.record_exploratory_approval(tmp_path, "r18", "A1")
    assert rec_file.read_bytes() == before                                     # the approval alters no verdict
    assert not decisions.rule_verdict(tmp_path, "r18")["eligible"]


# ------------------------------------------------------------------------------------------- authorization
def test_lobes12_is_authorized_by_the_rule_and_r18_only_by_the_recorded_approval(tmp_path):
    _resolution(tmp_path)
    a = decisions.atlas_authorization(tmp_path, "lobes12")
    assert a["basis"] == "rule" and "rule-eligible" in a["label"]
    with pytest.raises(decisions.AtlasNotAuthorized, match="no exploratory approval"):
        decisions.atlas_authorization(tmp_path, "r18")                         # listed in configs: not approval
    rec = decisions.record_exploratory_approval(tmp_path, "r18", "A1")
    assert rec["rule_eligible"] is False and rec["amendment_record"]["date"] == "2026-10-19"
    a = decisions.atlas_authorization(tmp_path, "r18")
    assert a["basis"] == "exploratory" and "not rule-eligible" in a["label"] and "A1" in a["label"]
    assert decisions.record_exploratory_approval(tmp_path, "r18", "A1")["recorded_at"] == rec["recorded_at"]  # once
    with pytest.raises(decisions.AtlasNotAuthorized):
        decisions.atlas_authorization(tmp_path, "dk68")                        # the reference is never rebuilt
    assert not (decisions.decisions_dir(tmp_path) / "atlas.json").exists()     # nothing is selected


@pytest.mark.parametrize("case", ["unknown_amendment", "wrong_atlas", "rule_eligible", "tampered", "incomplete",
                                  "approximate_bem", "anatomy_failed"])
def test_invalid_approvals_never_authorize(tmp_path, case):
    if case == "approximate_bem":
        _resolution(tmp_path, forward_model="approximate 3-layer fsaverage BEM fsaverage_approx3-bem-sol.fif")
        with pytest.raises(decisions.DecisionMissing, match="exact-BEM"):
            decisions.record_exploratory_approval(tmp_path, "r18", "A1")
        with pytest.raises(decisions.DecisionMissing):
            decisions.atlas_authorization(tmp_path, "lobes12")
        return
    _resolution(tmp_path, checks_ok=case != "anatomy_failed")
    if case == "unknown_amendment":
        with pytest.raises(decisions.AtlasNotAuthorized, match="not in"):
            decisions.record_exploratory_approval(tmp_path, "r18", "A9")
    elif case == "wrong_atlas":
        with pytest.raises(decisions.AtlasNotAuthorized):
            decisions.record_exploratory_approval(tmp_path, "dk34_gyral", "A1")
    elif case == "rule_eligible":
        root = tmp_path / "cfg"
        root.mkdir()
        a1 = decisions.amendments()["A1"]
        (root / "amendments.yaml").write_text(yaml.safe_dump({"A2": {**a1, "atlas": "lobes12"}}))
        with pytest.raises(decisions.AtlasNotAuthorized, match="needs no exploratory approval"):
            decisions.record_exploratory_approval(tmp_path, "lobes12", "A2", root=root)
    elif case == "tampered":
        decisions.record_exploratory_approval(tmp_path, "r18", "A1")
        p = decisions.exploratory_path(tmp_path, "r18")
        d = json.loads(p.read_text())
        d["amendment_record"]["scope"] = ["make r18 the analysis atlas"]
        p.write_text(json.dumps(d))
        with pytest.raises(decisions.AtlasNotAuthorized, match="does not match"):
            decisions.atlas_authorization(tmp_path, "r18")
    elif case == "incomplete":
        root = tmp_path / "cfg"
        root.mkdir()
        a1 = {k: v for k, v in decisions.amendments()["A1"].items() if k != "reason"}
        (root / "amendments.yaml").write_text(yaml.safe_dump({"A1": a1}))
        with pytest.raises(decisions.AtlasNotAuthorized, match="reason"):
            decisions.record_exploratory_approval(tmp_path, "r18", "A1", root=root)
    else:
        with pytest.raises(decisions.AtlasNotAuthorized, match="anatomical"):
            decisions.atlas_authorization(tmp_path, "lobes12")


def test_the_store_builder_refuses_an_unauthorized_atlas_before_any_processing(tmp_path):
    _resolution(tmp_path)
    paths = tmp_path / "paths.yaml"
    paths.write_text(yaml.safe_dump({"drive_root": str(tmp_path), "processed_dir": str(tmp_path),
                                     "cache_dir": str(tmp_path / "cache")}))
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    for atlas, ok in (("r18", False), ("dk68", False)):
        p = subprocess.run([sys.executable, str(REPO / "scripts" / "build_stores.py"), "build", "--inverse", "lcmv",
                            "--atlas", atlas, "--paths", str(paths)], cwd=REPO, env=env, capture_output=True,
                           text=True)
        assert p.returncode != 0 and ("not authorized" in p.stderr + p.stdout or "68-ROI" in p.stderr + p.stdout)
    assert not list(tmp_path.glob("D3_LCMV_XU/graphsLX-r18_*"))                 # nothing was built


# ------------------------------------------------------------------------------------------- three atlases
@pytest.fixture
def three_atlases(tmp_path):
    from adgraph.synthetic import write_release_stores

    for a in ("dk68", "lobes12", "r18"):
        write_release_stores(tmp_path, n_windows=3, psd_effect=0.8, conn_effect=0.8, band_direction=0.5, atlas=a)
    return tmp_path


def test_new_stores_need_their_atlas_match_and_get_their_own_qc(three_atlases):
    from adgraph.qc.report import require_qc

    for a in ("lobes12", "r18"):
        for inv in ("lcmv", "wmne"):
            for mode in ("B", "A"):
                assert verify_store(three_atlases, inv, mode, a)["passed"]
        store, _ = open_store(three_atlases, "wmne", "B", a)
        rep = require_qc(store, three_atlases, "wmne", "B", a)
        assert rep["summary"]["n_participants"] == 65 and rep["dir"].endswith(f"wmne_B_{a}")
    (store_dir(three_atlases, "lcmv", "A", "lobes12") / "atlas_match.json").unlink()
    assert not verify_store(three_atlases, "lcmv", "A", "lobes12")["passed"]


def test_three_atlas_comparison_labels_pairs_and_missing_results(three_atlases):
    from adgraph.experiments import roi_comparison as rc
    from adgraph.experiments.report import collect

    cfg = {**rc.config(), "inverses": ["lcmv"], "models": {"exp1_source_psd_de": ["logreg__psd"]}, "n_bootstrap": 100}
    assert cfg["atlases"] == ["dk68", "lobes12", "r18"] and cfg["reference"] == "dk68"
    assert "rule-recommended" in cfg["roles"]["lobes12"] and "exploratory" in cfg["roles"]["r18"]
    assert cfg["comparisons"] == [["lobes12", "dk68"], ["r18", "dk68"], ["r18", "lobes12"]]
    for r in rc.runs(cfg):
        if r["atlas"] != "r18":                                                  # r18 not run yet: missing
            run_experiment(experiment_configs()[r["exp"]], three_atlases, r["inverse"], "B", r["atlas"], None,
                           [r["variant"]], ["xu_dev_cv"])
    table = collect(release.results_dir(three_atlases))
    atl, missing = rc.per_atlas(table, cfg)
    cls, _ = rc.classification(table, cfg)
    summ = rc.summary(cls, missing, cfg)
    assert missing == ["r18/lcmv/exp1_source_psd_de/logreg__psd"] and set(atl["atlas"]) == {"dk68", "lobes12"}
    assert summ["comparisons"]["lobes12 - dk68"]["complete"] and not summ["complete"]
    assert not summ["comparisons"]["r18 - dk68"]["complete"] and not summ["comparisons"]["r18 - lobes12"]["complete"]
    run_experiment(experiment_configs()["exp1"], three_atlases, "lcmv", "B", "r18", None, ["logreg__psd"],
                   ["xu_dev_cv"])
    table = collect(release.results_dir(three_atlases))
    atl, missing = rc.per_atlas(table, cfg)
    cls, _ = rc.classification(table, cfg)
    summ = rc.summary(cls, missing, cfg)
    assert summ["complete"] and list(cls["comparison"]) == ["lobes12 - dk68", "r18 - dk68", "r18 - lobes12"]
    assert cls["same_participants"].all() and (cls["n_participants"] == 52).all()
    for _, r in cls.iterrows():
        assert r["auc_diff"] == pytest.approx(r["auc_a"] - r["auc_b"], abs=1e-9)
    md = rc.report_markdown(atl, cls, summ, {}, cfg, {"lobes12": "rule-eligible", "r18": "exploratory (A1)"})
    assert "rule-recommended" in md and "exploratory" in md and "not corrected for multiplicity" in md
    assert md.index("Source localization") < md.index("Classification per atlas") < md.index("Paired differences")
    assert not list(release.results_dir(three_atlases).glob("**/xu_holdout"))     # development CV only


def test_roi_compare_cli_selects_nothing_and_never_touches_the_holdout(three_atlases):
    from adgraph.experiments import roi_comparison as rc

    paths = three_atlases / "paths.yaml"
    paths.write_text(yaml.safe_dump({"drive_root": str(three_atlases), "processed_dir": str(three_atlases),
                                     "cache_dir": str(three_atlases / "cache")}))
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "final.py"), "roi-compare", "--paths", str(paths)],
                       cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode == 1 and "incomplete" in p.stdout                           # nothing run: incomplete, says so
    out = sorted((release.root(three_atlases) / "roi_comparison").glob("compare_*"))[-1]
    s = json.loads((out / "summary.json").read_text())
    assert s["atlas_selected"] is None and s["holdout"] == "not evaluated" and not s["complete"]
    assert len(s["missing"]) == len(rc.runs())
    d = decisions.decisions_dir(three_atlases)
    assert not any((d / f).exists() for f in ("atlas.json", "graph_construction.json", "final_analysis.json"))
    nb = json.loads((REPO / "notebooks" / "04_roi_68_vs_18.ipynb").read_text())
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for banned in ("choose-atlas", "xu_holdout", "final.py holdout", "final.py freeze", "final.py select",
                   "03_graph"):
        assert banned not in code


def test_notebook_02_gates_r18_on_the_explicit_approval():
    nb = json.loads((REPO / "notebooks" / "02_parcellation.ipynb").read_text())
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "EXPLORATORY_R18_APPROVAL = None" in code                              # never pre-approved
    assert "approve-exploratory --atlas r18" in code and "final.py authorize --atlas" in code
    assert code.index("final.py authorize") < code.index("build_stores.py build")
    for step in ("validate", "match --atlas", "match-atlas", "size", "final.py qc", "store-resolution", "reliability"):
        assert step in code


# ------------------------------------------------------------------------------------------- code pin
def test_completed_qc_survives_a_new_code_pin(three_atlases, monkeypatch):
    from adgraph import provenance
    from adgraph.colab import resolve_code_pin
    from adgraph.qc.report import require_qc

    store, _ = open_store(three_atlases, "lcmv", "B", "lobes12")
    first = require_qc(store, three_atlases, "lcmv", "B", "lobes12")["summary"]["computed_at"]
    monkeypatch.setattr(provenance, "git_commit", lambda *a, **k: "a-new-commit")
    assert require_qc(store, three_atlases, "lcmv", "B", "lobes12")["summary"]["computed_at"] == first   # reused
    pin = three_atlases / release.PIN_FILE
    pin.write_text("540135c\n")
    assert resolve_code_pin(pin, "newhead", has_work=True) == ("540135c", True)        # never moves silently
    assert resolve_code_pin(pin, "newhead", update=True, has_work=True) == ("newhead", False)
    assert "# previous pin: 540135c" in pin.read_text()                               # history kept
