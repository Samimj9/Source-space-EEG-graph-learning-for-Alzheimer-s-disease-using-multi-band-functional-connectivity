import pytest

from dependency_guard import (pytest_collectreport, pytest_runtest_logreport,  # noqa: F401
                              pytest_sessionfinish, pytest_terminal_summary)

from adgraph import storage


@pytest.fixture(autouse=True)
def _not_in_colab(monkeypatch):
    """Tests write to pytest's tmp dirs on purpose; when the suite itself runs in Colab, the
    real-run storage guard must not treat them as a real run on the VM disk. Tests of the
    guard patch ``in_colab`` explicitly."""
    monkeypatch.setattr(storage, "in_colab", lambda: False)


@pytest.fixture(scope="session")
def store():
    """Small synthetic graph run of the legacy D1 / D2 schema and cohort sizes (D1 10/10, D2 10/13), for the tests
    of the legacy graph builder and of the shared store / dataset code."""
    from adgraph.legacy.synthetic import make_synthetic_store

    return make_synthetic_store(psd_effect=0.6, conn_effect=0.6, seed=0)


@pytest.fixture(scope="session")
def d3_store():
    """Synthetic store of the final framework: ds004504 cohort (65 participants, 36 AD / 29 HC), five Xu bands,
    68 Desikan-Killiany ROIs, 19 sensors, planted spectral and connectivity effects (read-only)."""
    from adgraph.synthetic import make_store

    return make_store(n_windows=4, psd_effect=0.8, conn_effect=0.8, seed=0)


@pytest.fixture
def release_root(tmp_path):
    """A processed directory holding synthetic LCMV and wMNE stores (modes B and A) at the release's store paths."""
    from adgraph.synthetic import write_release_stores

    write_release_stores(tmp_path, n_windows=3, psd_effect=0.8, conn_effect=0.8, band_direction=0.5)
    return tmp_path
