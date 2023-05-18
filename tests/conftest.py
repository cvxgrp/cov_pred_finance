"""global fixtures"""

from pathlib import Path

import pytest
import pandas as pd

@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture()
def prices(resource_dir):
    return pd.read_csv(resource_dir/ "stock_prices.csv", index_col=0, header=0, parse_dates=True).ffill()

@pytest.fixture()
def Sigma_test_iewma(resource_dir):
    return pd.read_csv(resource_dir / "Sigma_iewma.csv", index_col=0, header=0)


@pytest.fixture()
def weights_test_combinator(resource_dir):
    return pd.read_csv(resource_dir / "weights_combinator.csv", index_col=0, header=None).squeeze()

@pytest.fixture()
def Sigma_test_combinator(resource_dir):
    return pd.read_csv(resource_dir / "Sigma_combinator.csv", index_col=0, header=0)
