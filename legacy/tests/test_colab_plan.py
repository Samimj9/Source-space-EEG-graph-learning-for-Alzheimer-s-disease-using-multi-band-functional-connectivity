"""Colab notebook helpers and the run-everything experiment plan (synthetic only)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from adgraph import colab
from adgraph.config import load_paths
from adgraph.experiments import plan as plan_mod
from adgraph.experiments import runner
from adgraph.experiments.config import load_experiment
from adgraph.experiments.plan import load_plan, plan_status, plan_units, run_plan
from adgraph.legacy.synthetic import make_synthetic_store

REPO = Path(__file__).resolve().parents[1]
TINY = {"evaluation.n_repeats": 1, "evaluation.k": 3, "evaluation.inner_k": 2, "evaluation.n_bootstrap": 20,
        "hparams": {}}


# ======================================================================= data location / paths
def _fake_drive(tmp_path, where="MyDrive/AD-LEB"):
    root = tmp_path / "drive" / where
    (root / "EEGs" / "Dataset 1 (2017)").mkdir(parents=True)
    (root / "EEGs" / "Dataset 2 (2019)" / "RestingState").mkdir(parents=True)
    (root / "Info participants").mkdir()
    return tmp_path / "drive", root


@pytest.mark.parametrize("where", ["MyDrive/AD-LEB", ".shortcut-targets-by-id/1AbC/AD-LEB",
                                   "Shareddrives/Lab/AD-LEB", "MyDrive/projects/AD-LEB"])
def test_find_data_root(tmp_path, where):
    drive, root = _fake_drive(tmp_path, where)
    assert colab.find_data_root(drive) == root


def test_find_data_root_explains_missing_shortcut(tmp_path):
    (tmp_path / "drive" / "MyDrive").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Add shortcut"):
        colab.find_data_root(tmp_path / "drive")


def test_output_root_and_paths_yaml(tmp_path, monkeypatch):
    drive, root = _fake_drive(tmp_path)
    assert colab.choose_output_root(root, drive) == root / "processed"
    real = colab._writable
    monkeypatch.setattr(colab, "_writable", lambda d: False if d == root / "processed" else real(d))
    out = colab.choose_output_root(root, drive)                       # shared folder read-only
    assert out == drive / "MyDrive" / "AD-LEB-processed" and out.is_dir()
    paths = load_paths(colab.write_paths_yaml(tmp_path, root, out))   # spaces in folder names survive
    assert paths["dataset1"]["eeg_dir"] == f"{root}/EEGs/Dataset 1 (2017)"
    assert paths["processed_dir"] == str(out) and paths["cache_dir"] == f"{out}/cache"


def test_code_pin(tmp_path):
    pin = tmp_path / "out" / "CODE_COMMIT.txt"
    assert colab.resolve_code_pin(pin, "aaa") == ("aaa", False)              # first session pins
    assert colab.resolve_code_pin(pin, "bbb") == ("aaa", True)               # newer push ignored (results)
    assert colab.resolve_code_pin(pin, "bbb", update=True) == ("bbb", False)  # deliberate update
    assert colab.resolve_code_pin(pin, "ccc") == ("bbb", True)
    assert "previous pin: aaa" in pin.read_text()
    # a stale pin with no completed work behind it (setup-only session) moves to the current commit
    assert colab.resolve_code_pin(pin, "ddd", has_work=False) == ("ddd", False)
    assert colab.resolve_code_pin(pin, "eee") == ("ddd", True)               # from now on it protects results


def test_completed_work_detection(tmp_path):
    out = tmp_path / "processed"
    assert not colab.has_completed_work(out)
    (out / "graphs_v3_abc" / "subjects" / "D1-1").mkdir(parents=True)
    (out / "graphs_v3_abc" / "subjects" / "D1-1" / "arrays.npz").write_bytes(b"partial")
    assert not colab.has_completed_work(out)                               # unfinished unit: no marker
    (out / "graphs_v3_abc" / "subjects" / "D1-1" / "subject.done.json").write_text("{}")
    assert colab.has_completed_work(out)
    exp = tmp_path / "p2" / "experiments" / "exp0_sensor_psd" / "svm" / "within_D1" / "units"
    exp.mkdir(parents=True)
    (exp / "fold_r000_f00.done.json").write_text("{}")
    assert colab.has_completed_work(tmp_path / "p2")


def _cohort_names(root, d1=(("Sub1", "AD"), ("Sub2", "HC")), d2=(("3_1342019", "HC"),)):
    for sid, _ in d1:
        (root / "EEGs" / "Dataset 1 (2017)" / f"sub{sid[3:]}.edf").write_bytes(b"")
    for sid, _ in d2:
        (root / "EEGs" / "Dataset 2 (2019)" / "RestingState" / f"{sid}_1.edf").write_bytes(b"")
    pd.DataFrame({"ID": [s for s, _ in d1], "Group": [g for _, g in d1], "Age": 70}).to_excel(
        root / "Info participants" / "Info Participants_2017.xlsx", index=False)
    pd.DataFrame({"Id": [s for s, _ in d2], "Group": [g for _, g in d2], "Age": 70}).to_excel(
        root / "Info participants" / "Info Participants_2019.xlsx", index=False)


def test_check_inputs(tmp_path):
    drive, root = _fake_drive(tmp_path)
    _cohort_names(root)
    paths = load_paths(colab.write_paths_yaml(tmp_path, root, tmp_path / "out"))
    expected = {"D1": {"recordings": 2, "AD": 1, "HC": 1}, "D2": {"recordings": 1, "AD": 0, "HC": 1}}
    table = colab.check_inputs(paths, expected)
    assert table["recordings"].tolist() == [2, 1]
    with pytest.raises(RuntimeError, match="D2: found 1 recordings"):
        colab.check_inputs(paths, {**expected, "D2": {"recordings": 23, "AD": 10, "HC": 13}})
    (root / "EEGs" / "Dataset 1 (2017)" / "sub9.edf").write_bytes(b"")      # recording without metadata
    with pytest.raises(RuntimeError, match="without a participant row"):
        colab.check_inputs(paths, {**expected, "D1": {"recordings": 3, "AD": 1, "HC": 1}})


def test_sh_streams_and_raises(capsys):
    colab.sh(f"{sys.executable} -c \"print('hello from child')\"")
    assert "hello from child" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="exit 3"):
        colab.sh(f"{sys.executable} -c \"import sys; sys.exit(3)\"")


def test_notebooks_are_run_all_ready():
    """Every notebook: valid JSON, the shared setup cell first, no shell '!' commands (they neither
    stop 'Run all' on failure nor show subprocess output), no commented-out run cells, no /content
    output paths."""
    for nb_path in sorted((REPO / "notebooks").rglob("*.ipynb")):
        nb = json.loads(nb_path.read_text())
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        assert "scripts/colab_setup.py" in code[0] and "drive.mount" in code[0], nb_path.name
        for src in code:
            compile(src, nb_path.name, "exec")                        # plain Python, no IPython magics
            assert not any(ln.lstrip().startswith(("!", "%", "# !")) for ln in src.splitlines()), nb_path.name
            assert "/content/smoke" not in src
        assert "sh(" in "".join(code[1:]) or "run_tests_once" in "".join(code[1:])


# ======================================================================= run plan
@pytest.fixture
def plan_file(tmp_path):
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({"experiments": [
        {"config": "configs/experiments/exp0_sensor_psd.yaml", "variants": ["logreg"], "overrides": TINY,
         "settings": ["within_D1", "pooled", "cross_D1_to_D2"]},
        {"config": "configs/experiments/exp1_source_psd.yaml", "variants": ["logreg"], "overrides": TINY,
         "settings": ["within_D2"]},
    ]}))
    return p


def test_plan_units_order(plan_file):
    units = plan_units(load_plan(plan_file), REPO)
    assert [(u[2]["experiment"], u[3]) for u in units] == [
        ("exp0_sensor_psd", "within_D1"), ("exp0_sensor_psd", "pooled"), ("exp0_sensor_psd", "cross_D1_to_D2"),
        ("exp1_source_psd", "within_D2")]
    assert all(u[2]["evaluation"]["n_repeats"] == 1 for u in units)


def test_committed_plan_covers_every_config():
    entries = load_plan(REPO / "configs" / "run_plan.yaml")
    listed = {e["config"] for e in entries}
    every = {str(p.relative_to(REPO)) for p in (REPO / "configs" / "experiments").rglob("*.yaml")
             if p.stem not in {"base", "classical_base", "gnn_base", "gnn_fixed_base"}}
    # nested-CV GNN configs are historical records (methodology log entry 42) and are never planned
    historical = {c for c in every if (lambda v: v["input"] == "graph" and not v.get("protocol"))(
        load_experiment(REPO / c))}
    # Dataset 3 experiments use their own graph runs (graph_family D3) and are never part of the D1/D2 plan
    every -= {c for c in every if load_experiment(REPO / c).get("graph_family") == "D3"}
    assert historical and listed == every - historical
    units = plan_units(entries, REPO)
    assert len({(u[2]["name"], u[3]) for u in units}) == len(units)     # no unit twice


def test_run_plan_resumes_skips_gated_and_isolates_failures(plan_file, tmp_path, monkeypatch):
    store = make_synthetic_store({"D1": (6, 6), "D2": (6, 7)}, windows_per_subject=(3, 3))
    entries = load_plan(plan_file)
    real = runner.run_variant
    calls = []

    def flaky(store_, v, setting, out, **kw):
        calls.append((v["experiment"], setting))
        if v["experiment"] == "exp0_sensor_psd" and setting == "pooled":
            raise ValueError("simulated failure")
        return real(store_, v, setting, out, **kw)

    monkeypatch.setattr(plan_mod, "run_variant", flaky)
    monkeypatch.setattr(plan_mod, "gate_failures", lambda s: ["operator verification error None > 1e-06"])
    out = tmp_path / "results"
    st = run_plan(entries, store, out, REPO).set_index("setting")
    assert st.loc["within_D1", "state"] == "completed" and 0 <= st.loc["within_D1", "subject_roc_auc"] <= 1
    assert st.loc["pooled", "state"] == "failed" and "simulated failure" in st.loc["pooled", "message"]
    assert st.loc["within_D2", "state"] == "completed"                 # continued after the failure
    assert st.loc["cross_D1_to_D2", "state"] == "skipped" and "gate" in st.loc["cross_D1_to_D2", "message"]
    assert ("exp0_sensor_psd", "cross_D1_to_D2") not in calls
    assert pd.read_csv(out / "_plan_status.csv").shape[0] == 4

    monkeypatch.setattr(plan_mod, "run_variant", real)
    fits = []
    real_make = runner.make_model
    monkeypatch.setattr(runner, "make_model", lambda v, s: (fits.append(1), real_make(v, s))[1])
    st2 = run_plan(entries, store, out, REPO).set_index("setting")
    assert st2.loc["pooled", "state"] == "completed"                   # failed unit retried
    n_pooled = len(fits)
    assert n_pooled > 0
    fits.clear()
    st3 = run_plan(entries, store, out, REPO)
    assert fits == [] and set(st3["state"]) == {"completed", "skipped"}   # everything reused
    prog = plan_status(entries, store, out, REPO)
    assert set(prog["state"]) == {"completed", "skipped (gate)"}


def test_run_plan_cli_status_and_run(tmp_path, plan_file):
    store = make_synthetic_store({"D1": (6, 6), "D2": (6, 7)}, windows_per_subject=(3, 3))
    processed = tmp_path / "processed"
    store.save(processed / "graphs_v3_synthetic")
    (processed / "graphs_v3_synthetic" / "assembly.done.json").write_text("{}")   # discoverable as assembled
    m = json.loads((processed / "graphs_v3_synthetic" / "manifest.json").read_text())
    m["assembled_at"] = "2026-01-01T00:00:00+00:00"
    m["source_operator"] = {**m.get("source_operator", {}), "shared_across_datasets": False}   # gate fails
    (processed / "graphs_v3_synthetic" / "manifest.json").write_text(json.dumps(m))
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump({"drive_root": str(tmp_path), "processed_dir": str(processed),
                                                         "cache_dir": str(processed / "cache")}))
    base = [sys.executable, "scripts/run_plan.py", "--paths", str(tmp_path / "paths.yaml"), "--plan", str(plan_file)]
    st = subprocess.run(base + ["--status"], capture_output=True, text=True, cwd=REPO)
    assert st.returncode == 0, st.stderr
    assert "not started" in st.stdout and "skipped (gate)" in st.stdout
    r = subprocess.run(base, capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (processed / "experiments" / "_plan_status.csv").exists()
    assert "completed" in r.stdout


def test_d3_and_d1d2_code_pins_are_separate(tmp_path, monkeypatch):
    """D3 work counts only for the D3 pin; the D1/D2 pin is restored to the last pre-D3 pin from its history,
    and never when D1/D2 units were completed under the current pin (methodology log entry 45)."""
    import importlib.util
    import subprocess

    out = tmp_path / "processed"
    (out / "graphsD3_v4_abc" / "subjects" / "D3-1").mkdir(parents=True)
    (out / "graphsD3_v4_abc" / "subjects" / "D3-1" / "subject.done.json").write_text("{}")
    assert colab.has_completed_work(out, "D3") and not colab.has_completed_work(out)
    (out / "graphs_v4_x" / "subjects" / "D1-1").mkdir(parents=True)
    (out / "graphs_v4_x" / "subjects" / "D1-1" / "subject.done.json").write_text('{"git_commit": "e637368"}')
    assert colab.has_completed_work(out) and colab.has_completed_work(out, "D3")

    spec = importlib.util.spec_from_file_location("code_pin", REPO / "scripts" / "code_pin.py")
    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    rev = lambda c: subprocess.check_output(["git", "rev-parse", c], cwd=REPO, text=True).strip()
    pre, d3 = rev(cp.FIRST_D3_COMMIT + "~1"), rev(cp.FIRST_D3_COMMIT)
    pin = out / "CODE_COMMIT.txt"
    colab.resolve_code_pin(pin, pre)
    colab.resolve_code_pin(pin, d3, update=True)                  # a D3 session moved the shared pin
    assert cp.pre_d3_pin(pin) == pre
    exp = out / "experiments" / "exp2" / "gcn" / "within_D1" / "units"
    exp.mkdir(parents=True)
    (exp / "fold.done.json").write_text(json.dumps({"git_commit": d3}))
    assert cp.restore_d1d2(out, None, force=False) == 1 and cp.pinned(pin) == d3     # refused: unit under pin
    (exp / "fold.done.json").unlink()
    assert cp.restore_d1d2(out, None, force=False) == 0 and cp.pinned(pin) == pre
    assert f"previous pin: {d3}" in pin.read_text() and "restored by scripts/code_pin.py" in pin.read_text()
    assert cp.restore_d1d2(out, None, force=False) == 0 and cp.pinned(pin) == pre     # idempotent
    assert not (out / "D3_CODE_COMMIT.txt").exists()
