from __future__ import annotations

import pandas as pd

from cvx.covariance.combination import from_ewmas


def test_covariance_estimator_mean(prices):
    # Frame of returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    combinator = from_ewmas(returns, pairs, clip_at=4.2, min_periods_cov=21)
    results_full = {
        result.time: result.weights
        for result in combinator.solve(window=10)
        if result is not None
    }

    # test times
    times = returns.index[-20:-10]
    results_test = {
        result.time: result.weights
        for result in combinator.solve(window=10, times=times)
        if result is not None
    }

    assert list(times) == list(results_test.keys())

    for time in times:
        pd.testing.assert_series_equal(
            results_full[time],
            results_test[time],
        )
