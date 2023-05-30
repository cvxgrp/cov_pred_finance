import pandas as pd

from cvx.covariance.covariance_combination import CovarianceCombination


def test_covariance_estimator(prices):
    # Frame of returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    combinator = CovarianceCombination.from_ewmas(returns, pairs, clip_at=4.2)
    results = {
        result.time: result.weights
        for result in combinator.solve(window=10)
    }

    pd.testing.assert_series_equal(
        results[pd.Timestamp("2018-04-11")],
        pd.Series(
            index=["10-10", "21-21", "21-63"], 
            data=[0.060104, 0.205062, 0.734834]
        ),
        atol=1e-3,
    )


def test_covariance_estimator_mean(prices):
    # Frame of returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    combinator = CovarianceCombination.from_ewmas(returns, pairs, clip_at=4.2, mean=True)
    results = {
        result.time: result.weights
        for result in combinator.solve(window=10)
    }

    pd.testing.assert_series_equal(
        results[pd.Timestamp("2018-04-11")],
        pd.Series(
            index=["10-10", "21-21", "21-63"],
            data=[0.090970, 0.029890, 0.879140],
        ),
        atol=1e-3,
    )


def test_covariance_estimator_no_clipping(prices):
    # Frame of returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    combinator = CovarianceCombination.from_ewmas(returns, pairs)
    results = {
        result.time: result.weights
        for result in combinator.solve(window=10)
    }

    pd.testing.assert_series_equal(
        results[pd.Timestamp("2018-04-11")],
        pd.Series(
            index=["10-10", "21-21", "21-63"],
            data=[0.167476, 0.016157, 0.816367],
        ),
        atol=1e-3,
    )
