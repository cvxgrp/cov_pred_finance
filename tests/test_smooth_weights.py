from __future__ import annotations

import pandas as pd
import pytest

from cvx.covariance.combination import from_ewmas


@pytest.fixture()
def returns(resource_dir):
    """
    prices fixture
    """
    return (
        pd.read_csv(
            resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True
        )
        .ffill()
        .pct_change()
        .iloc[1:]
    )


@pytest.fixture()
def time_stamp():
    return pd.Timestamp("2017-11-15")


@pytest.fixture()
def half_life_pairs():
    return [(10, 21), (21, 63)]


def test_l2(returns, resource_dir, time_stamp, half_life_pairs):
    iterator = from_ewmas(returns, half_life_pairs).solve(
        window=5, smoother="l2", gamma=1
    )

    covariances = {}
    weights = {}
    for iterate in iterator:
        covariances[iterate.time] = iterate.covariance
        weights[iterate.time] = iterate.weights
    cov = covariances[time_stamp]
    weights = pd.DataFrame(weights).T

    cov_reference = pd.read_csv(
        resource_dir / "cov_smooth_weights_l2.csv", index_col=0, header=0
    )
    weights_reference = pd.read_csv(
        resource_dir / "weights_smooth_weights_l2.csv",
        index_col=0,
        header=0,
        parse_dates=True,
    )

    print(weights.iloc[:, 0].name)
    print(weights_reference.iloc[:, 0].name)

    pd.testing.assert_frame_equal(cov, cov_reference, atol=1e-3)
    pd.testing.assert_frame_equal(weights, weights_reference, atol=1e-3)


def test_l1(returns, resource_dir, time_stamp, half_life_pairs):
    iterator = from_ewmas(returns, half_life_pairs).solve(
        window=5, smoother="l1", gamma=1
    )

    covariances = {}
    weights = {}
    for iterate in iterator:
        covariances[iterate.time] = iterate.covariance
        weights[iterate.time] = iterate.weights
    cov = covariances[time_stamp]
    weights = pd.DataFrame(weights).T

    cov_reference = pd.read_csv(
        resource_dir / "cov_smooth_weights_l1.csv", index_col=0, header=0
    )
    weights_reference = pd.read_csv(
        resource_dir / "weights_smooth_weights_l1.csv",
        index_col=0,
        header=0,
        parse_dates=True,
    )

    pd.testing.assert_frame_equal(cov, cov_reference, atol=1e-3)
    pd.testing.assert_frame_equal(weights, weights_reference, atol=1e-3)


def test_sum_squares(returns, resource_dir, time_stamp, half_life_pairs):
    iterator = from_ewmas(returns, half_life_pairs).solve(
        window=5, smoother="sum_squares", gamma=1
    )

    covariances = {}
    weights = {}
    for iterate in iterator:
        covariances[iterate.time] = iterate.covariance
        weights[iterate.time] = iterate.weights
    cov = covariances[time_stamp]
    weights = pd.DataFrame(weights).T

    cov_reference = pd.read_csv(
        resource_dir / "cov_smooth_weights_sum_squares.csv", index_col=0, header=0
    )
    weights_reference = pd.read_csv(
        resource_dir / "weights_smooth_weights_sum_squares.csv",
        index_col=0,
        header=0,
        parse_dates=True,
    )

    pd.testing.assert_frame_equal(cov, cov_reference, atol=1e-3)
    pd.testing.assert_frame_equal(weights, weights_reference, atol=1e-3)
