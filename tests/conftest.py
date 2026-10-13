import pytest

from dependency_guard import (pytest_collectreport, pytest_runtest_logreport,  # noqa: F401
                              pytest_sessionfinish, pytest_terminal_summary)

from adgraph import storage
from adgraph.synthetic import make_synthetic_store


@pytest.fixture(autouse=True)
def _not_in_colab(monkeypatch):
    """Tests write to pytest's tmp dirs on purpose; when the suite itself runs in Colab, the
    real-run storage guard must not treat them as a real run on the VM disk. Tests of the
    guard patch ``in_colab`` explicitly."""
    monkeypatch.setattr(storage, "in_colab", lambda: False)


@pytest.fixture(scope="session")
def synthetic_results(tmp_path_factory):
    """Synthetic graph run + experiment outputs from the real runner (the costliest fixture),
    built once per session and shared read-only; tests that modify results work on copies."""
    from pathlib import Path

    from adgraph.visualization.demo import build_synthetic_results

    root = tmp_path_factory.mktemp("synthetic_inputs")
    store, results = build_synthetic_results(root, cohort={"D1": (5, 5), "D2": (5, 6)})
    return store, Path(results)


@pytest.fixture(scope="session")
def store():
    """Small synthetic graph run with the real schema and cohort sizes (D1 10/10, D2 10/13)."""
    return make_synthetic_store(psd_effect=0.6, conn_effect=0.6, seed=0)
