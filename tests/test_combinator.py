import pytest
import pandas as pd

from cvx.covariance.covariance_combination import CovarianceCombination
from cvx.covariance.ewma import iterated_ewma


@pytest.fixture()
def prices(prices):
    return prices[["GOOG", "AAPL", "FB"]].ffill()


@pytest.fixture()
def returns(prices):

    return prices.pct_change().dropna(axis=0, how="all")


@pytest.fixture()
def combinator(returns):
    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    # compute the covariance matrices, one time series for each pair
    Sigmas = {f"{pair[0]}-{pair[1]}": iterated_ewma(returns, vola_halflife=pair[0], cov_halflife=pair[1], lower=-4.2, upper=4.2) for pair in pairs}


    # combination of covariance matrix valued time series
    return CovarianceCombination(Sigmas=Sigmas, returns=returns)


def test_combine_all(combinator, returns, weights_test_combinator, Sigma_test_combinator):
    """
    Tests the covariance combination function
    """
    time = returns.index[-1]
    results = combinator.solve(time=time, window=None)


    pd.testing.assert_series_equal(results.weights, weights_test_combinator,\
         check_names=False, atol=1e-6)
    pd.testing.assert_frame_equal(results.covariance, Sigma_test_combinator, rtol=1e-2)
    

