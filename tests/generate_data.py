from __future__ import annotations

import pandas as pd

from cvx.covariance import covariance_estimator
from cvx.covariance.ewma import iterated_ewma

# import test data
returns = (
    pd.read_csv("resources/stock_prices.csv", index_col=0, header=0, parse_dates=True)
    .ffill()[["GOOG", "AAPL", "FB"]]
    .pct_change()
    .dropna(axis=0, how="all")
)

# Iterated EWWMA covariance prediction
iewma = {
    result.time: result.covariance
    for result in iterated_ewma(
        returns,
        vola_halflife=10,
        cov_halflife=21,
        min_periods_vola=20,
        min_periods_cov=20,
        clip_at=4.2,
    )
}
sigma = iewma[returns.index[-1]]
sigma.to_csv("resources/Sigma_iewma.csv")

# Combination of covariance matrix valued time series
pairs = [(10, 10), (21, 21), (21, 63)]

results = {result.time: result for result in covariance_estimator(returns, pairs, clip_at=4.2, window=None)}

weights = results[returns.index[-1]].weights
sigma = results[returns.index[-1]].covariance

weights.to_csv("resources/weights_combinator.csv", header=False)
sigma.to_csv("resources/Sigma_combinator.csv")
