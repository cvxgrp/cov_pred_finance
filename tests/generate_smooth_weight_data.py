from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvx.covariance.combination import from_ewmas

save_data = False

# import test data
resource_dir = Path(__file__).parent / "resources"
returns = (
    pd.read_csv(
        resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True
    )
    .ffill()
    .pct_change()
    .iloc[1:]
)

time_stamp = pd.Timestamp("2017-11-15")
half_life_pairs = [(10, 21), (21, 63)]

iterator_l2 = from_ewmas(returns, half_life_pairs).solve(
    window=5, smoother="l2", gamma=1, solver="CLARABEL"
)
covariances_l2 = {}
weights_l2 = {}
for iterate in iterator_l2:
    covariances_l2[iterate.time] = iterate.covariance
    weights_l2[iterate.time] = iterate.weights
cov_l2 = covariances_l2[time_stamp]
weights_l2 = pd.DataFrame(weights_l2).T

iterator_l1 = from_ewmas(returns, half_life_pairs).solve(
    window=5, smoother="l1", gamma=1, solver="CLARABEL"
)
covariances_l1 = {}
weights_l1 = {}
for iterate in iterator_l1:
    covariances_l1[iterate.time] = iterate.covariance
    weights_l1[iterate.time] = iterate.weights
cov_l1 = covariances_l1[time_stamp]
weights_l1 = pd.DataFrame(weights_l1).T

iterator_sum_squares = from_ewmas(returns, half_life_pairs).solve(
    window=5, smoother="sum_squares", gamma=1, solver="CLARABEL"
)
covariances_sum_squares = {}
weights_sum_squares = {}
for iterate in iterator_sum_squares:
    covariances_sum_squares[iterate.time] = iterate.covariance
    weights_sum_squares[iterate.time] = iterate.weights

cov_sum_squares = covariances_sum_squares[time_stamp]
weights_sum_squares = pd.DataFrame(weights_sum_squares).T

if save_data:
    print("Saving data")
    cov_l2.to_csv(resource_dir / "cov_smooth_weights_l2.csv")
    cov_l1.to_csv(resource_dir / "cov_smooth_weights_l1.csv")
    cov_sum_squares.to_csv(resource_dir / "cov_smooth_weights_sum_squares.csv")
    weights_l2.to_csv(resource_dir / "weights_smooth_weights_l2.csv")
    weights_l1.to_csv(resource_dir / "weights_smooth_weights_l1.csv")
    weights_sum_squares.to_csv(resource_dir / "weights_smooth_weights_sum_squares.csv")
else:
    print("Not saving data")
