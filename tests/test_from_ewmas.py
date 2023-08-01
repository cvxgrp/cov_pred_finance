# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import pandas as pd

from cvx.covariance.combination import from_ewmas


def test_covariance_estimator(prices):
    # Frame of returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    combinator = from_ewmas(returns, pairs, clip_at=4.2)
    results = {}

    for result in combinator.solve(window=10):
        try:
            results[result.time] = result.weights
        except AttributeError:
            pass
        #    pass
    # results = {result.time: result.weights for result in combinator.solve(window=10)}

    pd.testing.assert_series_equal(
        results[pd.Timestamp("2018-04-11")],
        pd.Series(
            index=["10-10", "21-21", "21-63"], data=[0.060104, 0.205062, 0.734834]
        ),
        atol=1e-3,
    )


def test_covariance_estimator_mean(prices):
    # Frame of returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    combinator = from_ewmas(returns, pairs, clip_at=4.2, mean=True, min_periods_cov=21)
    results = {
        result.time: result.weights
        for result in combinator.solve(window=10)
        if result is not None
    }

    pd.testing.assert_series_equal(
        results[pd.Timestamp("2018-04-11")],
        pd.Series(
            index=["10-10", "21-21", "21-63"],
            # data=[0.090970, 0.029890, 0.879140],
            # i fixed a bug in iterated_ewma and these values changed
            data=[0.089445, 0.038577, 0.871978],
        ),
        atol=1e-3,
    )


def test_covariance_estimator_no_clipping(prices):
    # Frame of returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    combinator = from_ewmas(returns, pairs)
    results = {}
    for result in combinator.solve(window=10):
        try:
            results[result.time] = result.weights
        except AttributeError:
            pass

    pd.testing.assert_series_equal(
        results[pd.Timestamp("2018-04-11")],
        pd.Series(
            index=["10-10", "21-21", "21-63"],
            data=[0.167476, 0.016157, 0.816367],
        ),
        atol=1e-3,
    )
