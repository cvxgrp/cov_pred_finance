import pandas as pd
import pytest
from collections import namedtuple

from cvx.covariance.ewma import iterated_ewma as iterated_ewma_1
from cvx.covariance.iterated_ewma_vec import iterated_ewma as iterated_ewma_2

IEWMA = namedtuple("iewma", ["mean", "covariance"])

@pytest.fixture
def returns(prices):
    return 100*prices.pct_change().fillna(0.0)

def _evaluate_nonzero_mean(iewma1, iewma2):
    # Test mean 
    mean1 = iewma1.mean
    mean2 = iewma2.mean
    print(mean1.keys())
    #print(mean2.keys())
    #assert False

    #assert mean1.keys() == mean2.keys()
    for time in set(mean1.keys()).intersection(set(mean2.keys())):
        pd.testing.assert_series_equal(mean1[time], mean2[time], check_names=False)
    
    # Test covariance
    covariance1 = iewma1.covariance
    covariance2 = iewma2.covariance
    #assert covariance1.keys() == covariance2.keys()
    for time in set(covariance1.keys()).intersection(set(covariance2.keys())):
    #for time in covariance1.keys():
        pd.testing.assert_frame_equal(covariance1[time], covariance2[time])

def test_cov_zero_mean(returns):
    iewma1 = list(iterated_ewma_1(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, mean=False, clip_at=4.2))
    iewma1 = {item.time: item.covariance for item in iewma1}

    iewma2 = iterated_ewma_2(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, clip_at=4.2)

    assert iewma1.keys() == iewma2.keys()
    for time in iewma1.keys():
        pd.testing.assert_frame_equal(iewma1[time], iewma2[time])


def test_cov_nonzero_mean(returns):

    iewma1 = list(iterated_ewma_1(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, clip_at=4.2, mean =True))
    means = {item.time: item.mean for item in iewma1}
    covariances = {item.time: item.covariance for item in iewma1}
    iewma1 = IEWMA(mean=means, covariance=covariances)

    iewma2 = iterated_ewma_2(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, clip_at=4.2, mean=True)

    _evaluate_nonzero_mean(iewma1, iewma2)


