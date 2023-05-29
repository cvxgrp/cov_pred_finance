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
                                   pd.Series(index=["10-10", "21-21", "21-63"], data=[0.060104, 0.205062, 0.734834]))





