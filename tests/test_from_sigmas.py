from __future__ import annotations

import pandas as pd
import pytest

from cvx.covariance.combination import from_sigmas
from cvx.covariance.ewma import iterated_ewma


@pytest.fixture()
def prices(prices):
    """
    prices fixture
    """
    return prices[["GOOG", "AAPL", "FB"]].ffill()


@pytest.fixture()
def returns(prices):
    """
    returns fixture
    """
    return prices.pct_change().dropna(axis=0, how="all")


@pytest.fixture()
def combinator(returns):
    """
    combinator fixture
    """
    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    # compute the covariance matrices, one time series for each pair
    sigmas = {
        f"{pair[0]}-{pair[1]}": {
            result.time: result.covariance
            for result in iterated_ewma(
                returns, vola_halflife=pair[0], cov_halflife=pair[1], clip_at=4.2
            )
        }
        for pair in pairs
    }

    # combination of covariance matrix valued time series
    return from_sigmas(sigmas=sigmas, returns=returns)


def test_combine_all(
    combinator, returns, weights_test_combinator, Sigma_test_combinator
):
    """
    Tests the covariance combination function
    """
    r = {results.time: results for results in combinator.solve(window=None)}
    r = r[returns.index[-1]]

    pd.testing.assert_series_equal(
        r.weights, weights_test_combinator, check_names=False, atol=1e-6
    )
    pd.testing.assert_frame_equal(r.covariance, Sigma_test_combinator, rtol=1e-2)


def test_number_of_experts(combinator):
    """
    Tests the number of experts
    """
    assert combinator.K == 3


def test_assets(combinator):
    """
    Tests the assets
    """
    assert set(combinator.assets) == {"GOOG", "AAPL", "FB"}
