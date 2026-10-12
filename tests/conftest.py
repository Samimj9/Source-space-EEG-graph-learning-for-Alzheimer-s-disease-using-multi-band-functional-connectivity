import pytest

from dependency_guard import (pytest_collectreport, pytest_runtest_logreport,  # noqa: F401
                              pytest_sessionfinish, pytest_terminal_summary)

from adgraph.synthetic import make_synthetic_store


@pytest.fixture(scope="session")
def store():
    """Small synthetic graph run with the real schema and cohort sizes (D1 10/10, D2 10/13)."""
    return make_synthetic_store(psd_effect=0.6, conn_effect=0.6, seed=0)
