import pytest

from adgraph.synthetic import make_synthetic_store


@pytest.fixture(scope="session")
def store():
    """Small synthetic graph run with the real schema and cohort sizes (D1 10/10, D2 10/13)."""
    return make_synthetic_store(psd_effect=0.6, conn_effect=0.6, seed=0)
