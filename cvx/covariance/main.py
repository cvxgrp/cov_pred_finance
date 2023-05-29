from cvx.covariance.covariance_combination import CovarianceCombination
from cvx.covariance.ewma import iterated_ewma



def covariance_estimator(returns, pairs, min_periods_vola=20, min_periods_cov=20, clip_at=None, window=10, mean=False, **kwargs):
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

    sigmas = {pair: {time: result.covariance for time, result in results[pair].items()} for pair in results.keys()}
    means = {pair: {time: result.mean for time, result in results[pair].items()} for pair in results.keys()}

    # combination of covariance matrix valued time series
    combination = CovarianceCombination(sigmas=sigmas, returns=returns, means=means)

    for result in combination.solve(window=window, **kwargs):
        yield result
