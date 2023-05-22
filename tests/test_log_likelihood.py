import numpy as np
import pytest
from scipy.stats import multivariate_normal

from experiments.experiment_utils import log_likelihood


@pytest.fixture
def returns(prices):
    return 100*prices.pct_change().fillna(0.0)


def log_likelihood_benchmark(returns, Sigmas, means=None):
    log_likelihood = np.zeros(returns.shape[0])

    if means is None:
        means = [None]*returns.shape[0]

    for t in range(len(returns)):

        log_likelihood[t] = multivariate_normal.logpdf(returns[t], mean=means[t], cov=Sigmas[t])

    return log_likelihood

def test_log_likelihood(returns):
    np.random.seed(1)
    means = np.random.normal(size=returns.shape)

    A = np.random.normal(size = (returns.shape[0], returns.shape[1], returns.shape[1]))
    Sigmas = np.array([A[i] @ A[i].T for i in range(A.shape[0])])


    # Zero mean
    ll_benchmark = log_likelihood_benchmark(returns.values, Sigmas)
    ll_my = log_likelihood(returns.values, Sigmas)
    assert np.allclose(ll_benchmark, ll_my)


    # Nonzero mean
    ll_benchmark = log_likelihood_benchmark(returns.values, Sigmas, means)
    ll_my = log_likelihood(returns.values, Sigmas, means)
    assert np.allclose(ll_benchmark, ll_my)

