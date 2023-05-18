import numpy as np
import pandas as pd
import pytest

from cvx.covariance.iterated_ewma import iterated_ewma


@pytest.fixture()
def prices(prices):
    return prices[["GOOG", "AAPL", "FB"]]

@pytest.fixture()
def returns(prices):
    return prices.pct_change().dropna(axis=0, how="all")

def test_iterated_ewma(returns, Sigma_test_iewma):
    iewma = iterated_ewma(returns, vola_halflife=10, cov_halflife=21, min_periods=40, lower=-4.2, upper=4.2)

    Sigma_hat = iewma[returns.index[-1]]

    pd.testing.assert_frame_equal(Sigma_hat, Sigma_test_iewma)

