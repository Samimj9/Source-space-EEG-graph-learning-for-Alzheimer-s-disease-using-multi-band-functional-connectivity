"""Per-experiment Colab workflow: generated notebooks and the single-experiment CLI (synthetic only)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from adgraph.experiments.config import load_experiment
from adgraph.synthetic import make_synthetic_store

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks"


def _generator():
    spec = importlib.util.spec_from_file_location("make_notebooks", REPO / "scripts" / "make_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _code(nb_path):
    nb = json.loads(Path(nb_path).read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_committed_notebooks_match_the_generator(tmp_path):
    gen = _generator()
    outputs = gen.main(tmp_path)
    for path in outputs:
        rel = path.relative_to(tmp_path)
        assert (NB / rel).read_text() == path.read_text(), f"{rel} is stale: run python scripts/make_notebooks.py"
    assert (NB / "README.md").read_text() == (tmp_path / "README.md").read_text()
    committed = {p.relative_to(NB) for p in NB.rglob("*.ipynb")}
    assert committed == {p.relative_to(tmp_path) for p in outputs}       # no stray / old notebooks


def test_one_notebook_per_experiment_config():
    # the LCMV / Xu and wMNE / Xu suites have their own notebooks (two modes per notebook; test_lx_suite_notebooks)
    configs = {str(p.relative_to(REPO)) for p in (REPO / "configs" / "experiments").rglob("*.yaml")
               if p.stem not in {"base", "classical_base", "gnn_base", "gnn_fixed_base"}
               and p.relative_to(REPO / "configs" / "experiments").parts[0] not in ("lx", "wx", "wx_hist", "lx_es", "wx_es")}
    # nested-CV GNN configs are historical records without notebooks (methodology log entry 42)
    configs = {c for c in configs if not (lambda v: v["input"] == "graph" and not v.get("protocol"))(
        load_experiment(REPO / c))}
    found = {}
    for nb in [*(NB / "experiments").glob("*.ipynb"), *(NB / "additional").glob("*.ipynb"),
               *(NB / "d3" / "experiments").glob("*.ipynb")]:
        cfg = [ln.split("'")[1] for src in _code(nb) for ln in src.splitlines() if ln.startswith("CONFIG = ")]
        assert len(cfg) == 1, nb.name
        found[nb.name] = cfg[0]
    assert set(found.values()) == configs
    main = {f"Exp{k}" for k in range(8)}
    assert {n.split("_")[0] for n in (p.name for p in (NB / "experiments").glob("*.ipynb"))} == main


def test_experiment_notebooks_run_only_their_experiment():
    for nb in [*(NB / "experiments").glob("*.ipynb"), *(NB / "additional").glob("*.ipynb"),
               *(NB / "d3" / "experiments").glob("*.ipynb")]:
        src = "\n".join(_code(nb))
        assert "run_plan" not in src and "run_experiment.py" not in src, nb.name
        runs = [ln for ln in src.splitlines() if "scripts/experiment.py run" in ln]
        assert runs == ['    sh(f"python scripts/experiment.py run --config {CONFIG}")'], nb.name
        assert "experiment.py estimate" not in src and "experiment.py report" in src   # no cost stage
        assert "raise RuntimeError" in src.split("experiment.py report")[1]          # failures end visibly


def test_advanced_notebook_starts_nothing_by_default():
    src = "\n".join(_code(NB / "advanced" / "Run_all_experiments_ADVANCED.ipynb"))
    assert "RUN_EVERYTHING = False" in src
    runs = [ln.strip() for ln in src.splitlines() if "run_plan.py" in ln and "--status" not in ln]
    assert runs == ['sh("python scripts/run_plan.py --paths configs/paths.yaml")']
    guarded = src.split("if RUN_EVERYTHING:")[1].splitlines()[1].strip()
    assert guarded == runs[0]
    for other in NB.rglob("*.ipynb"):                                # nowhere else is the full plan run
        if "advanced" not in other.parts:
            assert not any("run_plan.py" in s and "--status" not in s for s in _code(other)), other.name


def test_comparative_figures_notebook_trains_nothing():
    src = "\n".join(_code(NB / "03_comparative_figures.ipynb"))
    assert "make_figures.py" in src and "experiment.py" not in src and "run_plan" not in src


# ------------------------------------------------------------------ the CLI behind every experiment notebook
@pytest.fixture
def workspace(tmp_path):
    store = make_synthetic_store({"D1": (6, 6), "D2": (6, 7)}, windows_per_subject=(3, 3), psd_effect=0.6)
    processed = tmp_path / "processed"
    g = processed / "graphs_v3_synthetic"
    store.save(g)
    (g / "assembly.done.json").write_text("{}")
    m = json.loads((g / "manifest.json").read_text())
    m["assembled_at"] = "2026-01-01T00:00:00+00:00"
    (g / "manifest.json").write_text(json.dumps(m))
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump({"drive_root": str(tmp_path), "processed_dir": str(processed),
                                                         "cache_dir": str(processed / "cache")}))
    return tmp_path, processed


def _tiny(tmp_path, config, **extra):
    cfg = {"extends": str(REPO / "configs" / "experiments" / config),
           "evaluation": {"n_repeats": 1, "k": 3, "inner_k": 2, "n_bootstrap": 20}, **extra}
    p = tmp_path / Path(config).name
    p.write_text(yaml.safe_dump(cfg))
    return p


def _cli(tmp_path, action, config):
    r = subprocess.run([sys.executable, "scripts/experiment.py", action, "--config", str(config),
                        "--paths", str(tmp_path / "paths.yaml")], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
    return r.stdout


@pytest.mark.parametrize("config,extra", [
    ("exp0_sensor_psd.yaml", {}),
    ("exp4_band_mean.yaml", {"training": {"epochs": 1}, "hparams": {"model.hidden": [8, 16]}}),
])
def test_single_experiment_cli_estimate_run_status_report(workspace, config, extra):
    tmp_path, processed = workspace
    cfg = _tiny(tmp_path, config, **extra)
    exp = yaml.safe_load((REPO / "configs" / "experiments" / config).read_text())["experiment"]
    est = _cli(tmp_path, "estimate", cfg)
    assert "ESTIMATED TOTAL" in est and "per variant" in est
    assert list((processed / "experiments" / exp / "_estimates").glob("estimate_*.csv"))
    assert "not started" in _cli(tmp_path, "status", cfg)
    run = _cli(tmp_path, "run", cfg)
    assert "No other experiment is started" in run
    assert {p.name for p in (processed / "experiments").iterdir()} == {exp}   # only this experiment
    status = pd.read_csv(processed / "experiments" / exp / "_run_status.csv")
    assert set(status["state"]) == {"completed"}                    # gate passes on the synthetic run
    order = status["setting"].tolist()
    assert order[:2] == ["within_D1", "within_D2"] and order[-1].startswith("cross")   # primary settings first
    rep = _cli(tmp_path, "report", cfg)
    report = sorted((processed / "experiments" / exp / "_reports").glob("report_*"))[-1]
    table = pd.read_csv(report / "results_summary.csv")
    assert len(table) == len(status) and table["roc_auc_mean"].between(0, 1).all()
    assert (report / "figures" / "D1_performance.png").exists() and (report / "README.txt").exists()
    assert list((report / "figures").glob("D2_roc_within_D1*.png"))
    assert "completed" in _cli(tmp_path, "status", cfg) and "report:" in rep
    _cli(tmp_path, "run", cfg)                                       # rerun: completed results reused
    rerun = pd.read_csv(processed / "experiments" / exp / "_run_status.csv")
    assert set(rerun["state"]) == {"completed"} and len(rerun) == len(status)


def test_lx_suite_notebooks():
    """Every LCMV / wMNE Xu-suite config is run by exactly one suite notebook (modes A and B in one notebook)."""
    exp = REPO / "configs" / "experiments"
    for suite, nbs in (("lx", (NB / "lcmv_xu").glob("LX_Exp*.ipynb")), ("wx", (NB / "wmne_xu").glob("WX_Exp*.ipynb")),
                       ("wx_hist", (NB / "wmne_xu" / "historical").glob("WXH_Exp*.ipynb")),
                       ("lx_es", (NB / "lcmv_xu" / "early_stopping").glob("LX_ES_Exp*.ipynb")),
                       ("wx_es", (NB / "wmne_xu" / "early_stopping").glob("WX_ES_Exp*.ipynb"))):
        want = {str(p.relative_to(REPO)) for p in (exp / suite).rglob("*.yaml")}
        per_nb = [{c for src in _code(nb) for c in want if f"'{c}'" in src} for nb in nbs]
        counts = {c: sum(c in s for s in per_nb) for c in want}
        assert counts and all(n == 1 for n in counts.values()), (suite, counts)


def test_early_stopping_notebooks_pin_their_own_code():
    """ES notebooks keep the suite pins (60-epoch results) and run their own ES200 pin, adopting the other suite's."""
    for nb in [*(NB / "lcmv_xu" / "early_stopping").glob("*.ipynb"), *(NB / "wmne_xu" / "early_stopping").glob("*.ipynb")]:
        src = "\n".join(_code(nb))
        assert "ES200_CODE_COMMIT.txt" in src and "UPDATE_ES_CODE = False" in src and "sibling or head" in src
        compile(src.replace("display(", "print("), str(nb), "exec")      # syntactically valid cells
