import collections

from cvx.covariance.covariance_combination import CovarianceCombination
from cvx.covariance.ewma import iterated_ewma


def _map_nested_dicts(ob, func):
    """
    Recursively applies a function to a nested dictionary
    """
    if isinstance(ob, collections.Mapping):
        return {k: _map_nested_dicts(v, func) for k, v in ob.items()}
    else:
        return func(ob)


def covariance_estimator(returns, pairs, min_periods_vola=20, min_periods_cov=20, clip_at=None, window=10, mean=False, **kwargs):
    """
    Estimate a series of covariance matrices using the iterated EWMA method

    param returns: Frame of returns
    param pairs: list of pairs of EWMA half lives, e.g. [(20, 20), (10, 50), (20, 60)],
                 pair[0] is the half life for volatility estimation
                 pair[1] is the half life for covariance estimation
    param min_periods_vola: minimum number of observations to start EWMA for volatility estimation (optional)
    param min_periods_cov: minimum number of observations to start EWMA for covariance estimation (optional)
    param clip_at: clip volatility adjusted returns at +- clip_at (optional)
    param mean: subtract EWMA mean from returns and volatility adjusted returns (optional)

    return: Yields tuples with time, mean, covariance matrix, weights
    """
    # compute the covariance matrices, one time series for each pair
    results = {f"{pair[0]}-{pair[1]}":
                 {result.time: result for result in iterated_ewma(returns=returns,
                                                                  vola_halflife=pair[0],
                                                                  cov_halflife=pair[1],
                                                                  min_periods_vola=min_periods_vola,
                                                                  min_periods_cov=min_periods_cov,
                                                                  clip_at=clip_at,
                                                                  mean=mean)}
              for pair in pairs}

    sigmas = _map_nested_dicts(results, lambda result: result.covariance)
    means = _map_nested_dicts(results, lambda result: result.mean)

    # combination of covariance matrix valued time series
    combination = CovarianceCombination(sigmas=sigmas, returns=returns, means=means)

    for result in combination.solve(window=window, **kwargs):
        yield result
