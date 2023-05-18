import numpy as np
import pandas as pd
import pytest

from cvx.covariance.ewma import ewma_cov, ewma_mean, volatility


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
    d2 = dict(ewma_cov(prices, halflife=halflife,min_periods=min_periods))

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
    d2 = pd.Series(dict(ewma_mean(prices, halflife=halflife)))
    pd.testing.assert_series_equal(d1, d2, rtol=1e-10, atol=1e-5)

    min_periods = 5
    d1 = pd.Series(dict(mean(prices, halflife=halflife,\
         min_periods=min_periods)))
    d2 = pd.Series(dict(ewma_mean(prices, halflife=halflife,\
         min_periods=min_periods)))
    pd.testing.assert_series_equal(d1, d2, rtol=1e-10, atol=1e-5)


def test_ewma_speed(prices_big):
    # repeat 20 times
    for i in range(20):
        for time, ewma in ewma_mean(prices_big, halflife=10):
            pass


def test_ewma_pandas_speed(prices_big):
    # repeat 20 times
    for i in range(20):
        for time, ewma in mean(prices_big, halflife=10):
            # ewma is a pandas series
            pass

def test_volatility_pandas_speed(prices_big):
    # repeat 5 times
    for i in range(5):
        returns = prices_big.pct_change().fillna(0.0)
        frame = returns.ewm(halflife=10).std()

def test_volatility_speed(prices_big):
    # repeat 5 times
    for i in range(5):
        vola = volatility(prices_big, halflife=10)


