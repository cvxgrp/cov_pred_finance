import numpy as np
import pandas as pd
import pytest

from cvx.covariance.iterated_ewma import ewma

@pytest.fixture()
def prices(prices):
    return prices[["GOOG", "AAPL", "FB"]].head(100)

def test_ewma(prices):
    halflife = 10
    pandas_ewma = prices.ewm(halflife=halflife).mean()

    my_ewma = pd.DataFrame(ewma(prices.values, halflife=halflife), index = prices.index, columns=prices.columns)

    pd.testing.assert_frame_equal(my_ewma, pandas_ewma)




