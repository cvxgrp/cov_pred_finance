from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvx.covariance.ewma import iterated_ewma

np.random.seed(0)


@pytest.fixture()
def returns(resource_dir):
    """
    prices fixture
    """
    returns = (
        pd.read_csv(resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True)
        .ffill()
        .pct_change()
        .iloc[1:]
    )
    np.random.seed(0)
    mask = np.random.choice([False, True], size=returns.shape, p=[0.9, 0.1])
    returns[mask] = np.nan

    return returns


def test_iewma_nans(returns, resource_dir):
    iewma_pair = (63, 125)
    iewma = list(
        iterated_ewma(
            returns,
            vola_halflife=iewma_pair[0],
            cov_halflife=iewma_pair[1],
            min_periods_vola=0,
            min_periods_cov=0,
            nan_to_num=True,
        )
    )
    iewma = {iterate.time: iterate.covariance for iterate in iewma}

    covariance = iewma[returns.index[-1]]
    covariance_reference = pd.read_csv(resource_dir / "cov_iewma_nan.csv", index_col=0, header=0)

    pd.testing.assert_frame_equal(covariance, covariance_reference)
