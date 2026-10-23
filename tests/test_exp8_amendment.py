"""Amendment A2 (methodology log entry 60): Exp 8 with a cortical-surface anatomical layer, run in its own namespace
with its own code pin. The degenerate original layer on lobes12, the surface adjacency, the amended config and its
10 variants (layers, training, predictions), participant isolation, the namespace / provenance / resume, the
read-only use of the release's stores and QC, the unchanged QC code, the Colab pin and the notebook. Synthetic only."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from adgraph import release
from adgraph.experiments import amendment
from adgraph.experiments.config import expand_variants, experiment_configs, load_experiment
from adgraph.models.xu_dual import XuDualGCN, anatomical_adjacency
from adgraph.source import atlas

REPO = Path(__file__).resolve().parents[1]
BASE_PIN = "59a9ab70a55d107c5ad72acff1fb48f3ef42dff0"
LOBES = atlas.region_names("lobes12")


def _variants(path):
    return {v["variant"]: v for v in expand_variants(load_experiment(path))}


AMENDED = _variants(amendment.experiment_configs("A2")["exp8a"])
ORIGINAL = _variants(experiment_configs()["exp8"])


# ------------------------------------------------------------------------------------------- the anatomical layer
def test_the_original_layer_is_empty_on_lobes12_and_the_surface_layer_is_not():
    assert LOBES == sorted(LOBES) and len(LOBES) == 12 and LOBES[:2] == ["cingulate-lh", "cingulate-rh"]
    with pytest.raises(ValueError, match="own group"):
        anatomical_adjacency(LOBES, "lobe_hemisphere")                      # the failure seen in Colab
    a = anatomical_adjacency(LOBES, "surface_contiguity")
    assert a.shape == (12, 12) and a.dtype == np.float32 and a.any()
    assert np.array_equal(a, a.T) and not np.diagonal(a).any() and set(np.unique(a)) == {0.0, 1.0}
    assert np.array_equal(a, atlas.contiguity(LOBES))                       # the atlas's own mesh contiguity
    hemi = np.array([n.rsplit("-", 1)[1] for n in LOBES])
    assert not a[hemi[:, None] != hemi[None, :]].any()                      # never across hemispheres
    assert int(a.sum() // 2) == 22 and (a.sum(1) >= 3).all()                 # every region has neighbours
    b = anatomical_adjacency(LOBES, "lobe_bilateral")                        # the alternative prior: homologues
    assert int(b.sum() // 2) == 6 and all(b[i, j] == (LOBES[i][:-3] == LOBES[j][:-3]) for i in range(12)
                                          for j in range(12) if i != j)
    assert not (a * b).any()                                                 # a different prior, not a subset


def test_dk68_behaviour_of_the_original_groupings_is_unchanged():
    dk = atlas.region_names("dk68")
    assert int(anatomical_adjacency(dk).sum() // 2) == 226
    assert int(anatomical_adjacency(dk, "lobe_bilateral").sum() // 2) > 226


# ------------------------------------------------------------------------------------------- the amended config
def test_the_amended_config_differs_from_exp8_only_as_documented():
    rename = {"plv__psd_de_strength__abl_bilateral_lobes": "plv__psd_de_strength__alt_bilateral_lobes"}
    assert sorted(rename.get(k, k) for k in ORIGINAL) == sorted(AMENDED)
    for k, o in ORIGINAL.items():
        a = AMENDED[rename.get(k, k)]
        assert a["experiment"] == "exp8a_xu_dual_gcn_surface" and a["amendment"] == "A2"
        o2 = {x: y for x, y in o.items() if x not in ("experiment", "name", "variant")}
        a2 = {x: y for x, y in a.items() if x not in ("experiment", "name", "variant", "amendment")}
        want = o2["model"]["anatomical_grouping"]
        o2["model"] = {**o2["model"], "anatomical_grouping": "surface_contiguity" if want == "lobe_hemisphere" else want}
        assert a2 == o2, k                                                    # training, features, graph: identical


EXPECTED = {  # variant: (layer1, layer2, anatomical prior, band-specific weights)
    "plv__psd_de_strength": ("functional", "anatomical", "surface_contiguity", True),
    "ciplv__psd_de_strength": ("functional", "anatomical", "surface_contiguity", True),
    "plv__psd": ("functional", "anatomical", "surface_contiguity", True),
    "ciplv__psd": ("functional", "anatomical", "surface_contiguity", True),
    "plv__de": ("functional", "anatomical", "surface_contiguity", True),
    "ciplv__de": ("functional", "anatomical", "surface_contiguity", True),
    "plv__psd_de_strength__abl_functional_both": ("functional", "functional", None, True),
    "plv__psd_de_strength__abl_anatomical_both": ("anatomical", "anatomical", "surface_contiguity", True),
    "plv__psd_de_strength__alt_bilateral_lobes": ("functional", "anatomical", "lobe_bilateral", True),
    "plv__psd_de_strength__abl_shared_weights": ("functional", "anatomical", "surface_contiguity", False),
}


def _net(v, in_dim=4):
    keys = ("hidden", "node_out", "fc1", "fc2", "dropout", "band_specific", "layer1", "layer2", "anatomical_grouping")
    return XuDualGCN(in_dim=in_dim, n_bands=5, n_rois=12, roi_names=LOBES, **{k: v["model"][k] for k in keys})


@pytest.mark.parametrize("variant", sorted(EXPECTED))
def test_each_variant_uses_its_intended_layers(variant):
    l1, l2, prior, specific = EXPECTED[variant]
    v = AMENDED[variant]
    net = _net(v).eval()                                                    # no dropout noise
    assert (net.layer1, net.layer2) == (l1, l2)
    assert net.gcn1.weight.shape[0] == (5 if specific else 1)
    if prior:
        assert np.array_equal(net.a_anat.numpy(), anatomical_adjacency(LOBES, prior))
    torch.manual_seed(0)
    x, adj = torch.randn(3, 5, 12, 4), torch.rand(3, 5, 12, 12)
    adj = (adj + adj.transpose(-1, -2)) / 2
    with torch.no_grad():
        out = net.fused_embedding(x, adj)
        other = net.fused_embedding(x, adj * 0.1)                           # changes iff a functional layer exists
    assert torch.isfinite(out).all()
    assert torch.allclose(out, other) == ("functional" not in (l1, l2))
    assert v["training"]["epochs"] == 60 and not (v["training"].get("early_stopping") or {}).get("enabled")


# ------------------------------------------------------------------------------------------- the namespace
@pytest.fixture
def amended_root(tmp_path):
    """Synthetic dk68 + lobes12 stores at the release paths, their QC reports (as notebooks 01 / 02), the release
    pin and a completed release unit identity (for the environment check)."""
    from adgraph.provenance import versions
    from adgraph.qc.report import require_qc
    from adgraph.stores import open_store
    from adgraph.synthetic import write_release_stores

    for a in ("dk68", "lobes12"):
        write_release_stores(tmp_path, n_windows=3, psd_effect=0.8, conn_effect=0.8, band_direction=0.5, atlas=a)
    for inv in ("lcmv", "wmne"):
        for mode in ("B", "A"):
            store, _ = open_store(tmp_path, inv, mode, "lobes12")
            require_qc(store, tmp_path, inv, mode, "lobes12")
    (tmp_path / release.PIN_FILE).write_text(BASE_PIN + "\n")
    ref = release.results_dir(tmp_path) / "lcmv/mode_B/lobes12/dense/exp7_earm_gnn/v/xu_dev_cv/_identity.json"
    ref.parent.mkdir(parents=True)
    ref.write_text(json.dumps({"identity": {"code": {"git_commit": BASE_PIN, "libraries": versions()}}}))
    return tmp_path


def _snapshot(d):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(d).rglob("*"))
            if p.is_file()}


def _short(monkeypatch, epochs=2):
    from adgraph.models import trainer

    orig = trainer.GNNModel.__init__

    def init(self, model, training, *a, **k):
        orig(self, model, {**training, "epochs": epochs}, *a, **k)
    monkeypatch.setattr(trainer.GNNModel, "__init__", init)


def test_all_ten_variants_train_predict_and_resume_in_the_amendment_namespace(amended_root, monkeypatch):
    _short(monkeypatch)
    main = release.results_dir(amended_root)
    partial = main / "lcmv/mode_B/lobes12/dense/exp8_xu_dual_gcn/plv__psd/xu_dev_cv"   # the original, failed exp8
    partial.mkdir(parents=True)
    (partial / "status.csv").write_text("unit,state\nfold_r000_f00,running\n")
    qc_before = _snapshot(release.root(amended_root) / "qc")
    main_before, lock_before = _snapshot(main), (amended_root / release.LOCK_FILE).read_bytes()
    man = amendment.initialize(amended_root, "A2")
    assert man["base_release_pin"] == BASE_PIN and man["environment_ok"]
    t = amendment.run(amended_root, "A2", "exp8a", "lcmv", "B", "lobes12", "dense")
    assert len(t) == 10 and (t["state"] == "completed").all() and set(t["setting"]) == {"xu_dev_cv"}
    root = amendment.results_dir(amended_root, "A2") / "lcmv/mode_B/lobes12/dense/exp8a_xu_dual_gcn_surface"
    assert sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")) == sorted(EXPECTED)
    from adgraph.evaluation.splits import subject_table, xu_partition
    from adgraph.stores import open_store

    store, _ = open_store(amended_root, "lcmv", "B", "lobes12", lock=False)
    dev, holdout = xu_partition(subject_table(store.windows))
    for variant in EXPECTED:
        d = root / variant / "xu_dev_cv"
        pred = pd.read_csv(d / "predictions_participants.csv")
        assert len(pred) == 52 and pred["score"].notna().all() and not set(pred["key"]) & set(holdout)
        folds = pd.read_csv(d / "folds.csv")
        for _, g in folds.groupby("fold"):                                  # participant-disjoint folds
            assert not set(g[g.side == "train"]["key"]) & set(g[g.side == "test"]["key"])
        ident = json.loads((d / "_identity.json").read_text())["identity"]
        assert ident["experiment"] == "exp8a_xu_dual_gcn_surface" and ident["context"]["atlas"] == "lobes12"
        assert json.loads((d / "manifest.json").read_text())["release"]["git_commit"] == man["amendment_commit"]
    assert _snapshot(main) == main_before                                   # the release results: untouched
    assert _snapshot(release.root(amended_root) / "qc") == qc_before        # QC reused, never recomputed
    assert (amended_root / release.LOCK_FILE).read_bytes() == lock_before   # release lock: read only
    assert (amended_root / release.PIN_FILE).read_text() == BASE_PIN + "\n"  # release pin: untouched
    again = amendment.run(amended_root, "A2", "exp8a", "lcmv", "B", "lobes12", "dense")
    assert (again["state"] == "completed").all()                            # resumed from its own checkpoints
    assert not list(amendment.results_dir(amended_root, "A2").glob("**/xu_holdout"))


def test_the_amendment_needs_complete_qc_its_base_pin_and_its_own_commit(amended_root, monkeypatch):
    import shutil

    from adgraph import provenance

    (amended_root / release.PIN_FILE).write_text("someothercommit\n")
    with pytest.raises(amendment.AmendmentError, match="release pin"):
        amendment.initialize(amended_root, "A2")
    (amended_root / release.PIN_FILE).write_text(BASE_PIN + "\n")
    monkeypatch.setattr(provenance, "versions", lambda *a, **k: {"python": "3.0", "torch": "0.1"})
    with pytest.raises(amendment.AmendmentError, match="environment differs"):
        amendment.initialize(amended_root, "A2")
    man = amendment.initialize(amended_root, "A2", accept_environment_difference=True)
    assert man["environment_difference_accepted"] and man["library_differences"]
    monkeypatch.setattr(provenance, "git_commit", lambda *a, **k: "a-later-commit")
    with pytest.raises(amendment.AmendmentError, match="belongs to commit"):
        amendment.require_initialized(amended_root, "A2")
    monkeypatch.undo()
    shutil.rmtree(release.root(amended_root) / "qc" / "wmne_A_lobes12")
    with pytest.raises(amendment.AmendmentError, match="no complete participant QC"):
        amendment.open_context(amended_root, "wmne", "A", "lobes12")
    assert not (release.root(amended_root) / "qc" / "wmne_A_lobes12").exists()   # nothing computed in the release


def test_the_qc_code_is_unchanged_since_the_release_pin():
    """QC reports are reused across the two commits only because the QC identity hashes qc/ only (unchanged)."""
    from adgraph.provenance import PACKAGE_DIR, source_fingerprint

    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE_PIN, "src/adgraph/qc"], cwd=REPO,
                                    text=True).split()
    h = hashlib.sha256()
    for name in sorted(n for n in names if n.endswith(".py")):
        rel = Path(name).relative_to("src/adgraph")
        h.update(str(rel).encode())
        h.update(b"\0")
        h.update(subprocess.check_output(["git", "show", f"{BASE_PIN}:{name}"], cwd=REPO))
        h.update(b"\0")
    assert h.hexdigest() == source_fingerprint(("qc",)) and PACKAGE_DIR.name == "adgraph"


# ------------------------------------------------------------------------------------------- pin, CLI, notebook
def test_the_amendment_pin_is_separate_from_the_release_pin(tmp_path):
    from adgraph.colab import resolve_code_pin

    assert amendment.pin_file("A2") == "FINAL_v1/amendments/A2_exp8_surface/CODE_COMMIT.txt" != release.PIN_FILE
    (tmp_path / release.PIN_FILE).parent.mkdir(parents=True)
    (tmp_path / release.PIN_FILE).write_text(BASE_PIN + "\n")
    commit, _ = resolve_code_pin(tmp_path / amendment.pin_file("A2"), "newcommit", has_work=False)
    assert commit == "newcommit" and (tmp_path / release.PIN_FILE).read_text() == BASE_PIN + "\n"
    setup = (REPO / "scripts" / "colab_setup.py").read_text()
    assert "--amendment" in setup and "amendment.pin_file(args.amendment)" in setup


def test_cli_status_check_and_report_stay_in_the_namespace(amended_root):
    from adgraph.checkpoint import atomic_write_json
    from adgraph.experiments import decisions

    atomic_write_json(decisions.atlas_path(amended_root), {"atlas": "lobes12"})
    atomic_write_json(decisions.graph_path(amended_root), {"selected": "dense", "atlas": "lobes12", "reason": "kept"})
    paths = amended_root / "paths.yaml"
    paths.write_text(yaml.safe_dump({"drive_root": str(amended_root), "processed_dir": str(amended_root),
                                     "cache_dir": str(amended_root / "cache")}))
    env = {"PYTHONPATH": str(REPO / "src"), **{k: v for k, v in __import__("os").environ.items()}}

    def cli(*a, ok=True):
        p = subprocess.run([sys.executable, str(REPO / "scripts" / "final.py"), *a, "--paths", str(paths)], cwd=REPO,
                           env=env, capture_output=True, text=True)
        assert (p.returncode == 0) == ok, p.stdout + p.stderr
        return p.stdout + p.stderr

    out = cli("amendment-check", "--amendment", "A2", "--atlas", "selected")
    assert "ready" in out and str(amendment.results_dir(amended_root, "A2")) in out
    out = cli("status", "--amendment", "A2", "--exp", "exp8a", "--inverse", "wmne", "--mode", "A", "--atlas",
              "selected", "--graph", "selected")
    assert "exp8a_xu_dual_gcn_surface" in out and "not started" in out
    out = cli("report", "--amendment", "A2", "--exp", "exp8a", "--inverse", "wmne", "--mode", "A", "--atlas",
              "selected", "--graph", "selected")
    assert str(amendment.reports_dir(amended_root, "A2")) in out and "0 completed results" in out
    assert "exp8a" not in cli("report")                                     # the release report ignores A2
    cli("status", "--exp", "exp8a", ok=False)                               # exp8a exists only in the amendment


def test_the_exp8a_notebook_uses_its_own_pin_and_namespace_only():
    nb = json.loads((REPO / "notebooks" / "Exp8a_xu_dual_gcn_surface.ipynb").read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert "'scripts/colab_setup.py', '--amendment', 'A2'" in cells[0] and "MPLBACKEND" in cells[0]
    calls = [ln for c in cells for ln in c.splitlines() if "scripts/final.py" in ln]
    assert calls and all("--amendment A2" in ln for ln in calls)
    code = "\n".join(cells)
    for banned in ("xu_holdout", "final.py holdout", "final.py freeze", "choose-atlas", "UPDATE_CODE = True"):
        assert banned not in code
    assert "A2_ROOT}/results/" in code and "exp8a_xu_dual_gcn_surface/_reports" in code
    plan = yaml.safe_load((REPO / "configs" / "final" / "run_plan.yaml").read_text())["steps"]
    names = [s["notebook"] for s in plan]
    assert names.index("Exp8a_xu_dual_gcn_surface.ipynb") == names.index("Exp8_xu_dual_gcn.ipynb") + 1
