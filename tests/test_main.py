import pandas as pd

from cvx.covariance import covariance_estimator


def test_covariance_estimator(prices):
    # Frame of returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    # Here we just combine all available matrices in one optimization problem
    results = {result.time: result.weights for result in covariance_estimator(returns, pairs, clip_at=4.2, window=10)}

    pd.testing.assert_series_equal(results[pd.Timestamp("2018-04-11")],
                                   pd.Series(index=["10-10", "21-21", "21-63"], data=[0.060104, 0.205062, 0.734834]),
                                   atol=1e-3)


def test_covariance_estimator_mean(prices):
    # Frame of returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    # Here we just combine all available matrices in one optimization problem
    results = {result.time: result.weights for result in covariance_estimator(returns, pairs, clip_at=4.2, window=10, mean=True)}

    pd.testing.assert_series_equal(results[pd.Timestamp("2018-04-11")],
                                   pd.Series(index=["10-10", "21-21", "21-63"], data=[0.09097005587922206, 0.029890296600343525, 0.8791396456545304]),
                                   atol=1e-3)


def test_covariance_estimator_no_clipping(prices):
    # Frame of returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    # Here we just combine all available matrices in one optimization problem
    results = {result.time: result.weights for result in covariance_estimator(returns, pairs, window=10)}

    pd.testing.assert_series_equal(results[pd.Timestamp("2018-04-11")],
                                   pd.Series(index=["10-10", "21-21", "21-63"], data=[0.16747553827969558, 0.016157108129744458, 0.8163673515094994]),
                                   atol=1e-3)





