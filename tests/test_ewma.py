import numpy as np
import pandas as pd
import pytest

from cvx.covariance.ewma import _ewma_cov, _ewma_mean, volatility, center, clip


@pytest.fixture()
def prices(prices):
    return prices[["GOOG", "AAPL", "FB"]].head(100)

@pytest.fixture()
def prices_big():
    return pd.DataFrame(data=np.random.randn(1000, 300) + 100)

# Thomas: i think this implementation is cleaner and hence more maintainable than the one in
# cvx.covariance.ewma...
def cov_no_mean_correction(data, halflife, min_periods=None):
    covariances = data.ewm(halflife=halflife, min_periods=min_periods)\
        .cov(bias=True)
    means = data.ewm(halflife=halflife, min_periods=min_periods).mean()

    # correct the covariance estimate by adding back the outer product of the current means
    for t, mean_vector in means.iterrows():
        if not np.isnan(mean_vector.values).all():
            yield t, covariances.loc[t] + np.outer(mean_vector.values, mean_vector.values)

def mean(data, halflife, min_periods=None):
    means = data.ewm(halflife=halflife, min_periods=min_periods).mean()
    # correct the covariance estimate by adding back the outer product of the current means
    for t, mean_vector in means.iterrows():
        if not np.isnan(mean_vector).all():
            yield t, mean_vector



@pytest.mark.parametrize("halflife", [1, 2, 4, 8, 16, 32, 64])
def test_ewma_cov(prices, halflife):
    """
    tests ewma implementation against pandas implementation\
        when assuming zero mean
    """
    min_periods = 5
    d1 = dict(cov_no_mean_correction(prices, halflife=halflife, min_periods=min_periods))
    d2 = dict(_ewma_cov(prices, halflife=halflife, min_periods=min_periods))

    assert d1.keys() == d2.keys()

    for time in d1.keys():
        pd.testing.assert_frame_equal(d1[time], d2[time], rtol=1e-10, atol=1e-5)



@pytest.mark.parametrize("halflife", [1, 2, 4, 8, 16, 32, 64])
def test_ewma_mean(prices, halflife):
    """
    tests ewma implementation against pandas implementation\
        when assuming zero mean
    """
    d1 = pd.Series(dict(mean(prices, halflife=halflife)))
    d2 = pd.Series(dict(_ewma_mean(prices, halflife=halflife)))
    pd.testing.assert_series_equal(d1, d2, rtol=1e-10, atol=1e-5)

    min_periods = 5
    d1 = pd.Series(dict(mean(prices, halflife=halflife,\
         min_periods=min_periods)))
    d2 = pd.Series(dict(_ewma_mean(prices, halflife=halflife, \
                                   min_periods=min_periods)))
    pd.testing.assert_series_equal(d1, d2, rtol=1e-10, atol=1e-5)




def test_center_inactive():
    # Test case 1
    returns = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    halflife = 1
    min_periods = 0
    mean_adj = False
    expected_centered_returns = returns
    centered_returns, mean = center(returns, halflife, min_periods, mean_adj)
    pd.testing.assert_frame_equal(centered_returns, expected_centered_returns)
    pd.testing.assert_frame_equal(mean, 0.0 * returns)


def test_center_active():
    # Test case 2
    returns = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    halflife = 1
    min_periods = 0
    mean_adj = True
    expected_mean = pd.DataFrame({'a': [1.0, 1.666667, 2.428571], 'b': [4.0, 4.666667, 5.4285715]})
    expected_centered_returns = returns.sub(expected_mean)

    centered_returns, mean = center(returns, halflife, min_periods, mean_adj)
    pd.testing.assert_frame_equal(centered_returns, expected_centered_returns)
    pd.testing.assert_frame_equal(mean, expected_mean)


def test_clip_1():
    # Test case 1
    data = pd.DataFrame({'a': [1, 2, 3, -4], 'b': [4, -5, 6, 7]})
    clip_at = 5
    expected_data = pd.DataFrame({'a': [1, 2, 3, -4], 'b': [4, -5, 5, 5]})
    clipped_data = clip(data, clip_at)
    assert clipped_data.equals(expected_data)


def test_clip_2():
    # Test case 2
    data = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
    clip_at = None
    expected_data = data
    clipped_data = clip(data, clip_at)
    assert clipped_data.equals(expected_data)
