from __future__ import annotations

import numpy as np
import pandas as pd

from cvx.covariance.ewma import center, clip


def test_center_inactive():
    # Test case 1
    returns = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    halflife = 1
    min_periods = 0
    mean_adj = False
    expected_centered_returns = returns
    centered_returns, mean = center(returns, halflife, min_periods, mean_adj)
    pd.testing.assert_frame_equal(centered_returns, expected_centered_returns)
    pd.testing.assert_frame_equal(mean, 0.0 * returns)


def test_center_active():
    # Test case 2
    returns = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    halflife = 1
    min_periods = 0
    mean_adj = True
    expected_mean = pd.DataFrame(
        {"a": [1.0, 1.666667, 2.428571], "b": [4.0, 4.666667, 5.4285715]}
    )
    expected_centered_returns = returns.sub(expected_mean)

    centered_returns, mean = center(returns, halflife, min_periods, mean_adj)
    pd.testing.assert_frame_equal(centered_returns, expected_centered_returns)
    pd.testing.assert_frame_equal(mean, expected_mean)


def test_clip_1():
    # Test case 1
    data = pd.DataFrame({"a": [1, 2, 3, -4], "b": [4, -5, 6, 7]})
    clip_at = 5
    expected_data = pd.DataFrame({"a": [1, 2, 3, -4], "b": [4, -5, 5, 5]})
    clipped_data = clip(data, clip_at)
    assert clipped_data.equals(expected_data)


def test_clip_2():
    # Test case 2
    data = pd.DataFrame(np.random.randn(10, 5), columns=["a", "b", "c", "d", "e"])
    clip_at = None
    expected_data = data
    clipped_data = clip(data, clip_at)
    assert clipped_data.equals(expected_data)
