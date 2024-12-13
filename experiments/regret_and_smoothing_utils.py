import pandas as pd
import numpy as np


################## REGRET CODE ##################


def get_prescient(returns):
    """
    Returns the prescient predictor (sample covariance of each quarter).
    """

    prescient = {}
    quarter_prev = None
    for t in returns.index:
        # get sample covariance matrix for corresponding quarter
        quarter = (t.month - 1) // 3 + 1
        if quarter != quarter_prev:
            cov = np.cov(
                returns.loc[
                    (returns.index.year == t.year) & (returns.index.quarter == quarter)
                ].values,
                rowvar=False,
            )
            mean = np.mean(
                returns.loc[
                    (returns.index.year == t.year) & (returns.index.quarter == quarter)
                ].values,
                axis=0,
            )
            quarter_prev = quarter
        prescient[t] = pd.DataFrame(
            cov + np.outer(mean, mean), index=returns.columns, columns=returns.columns
        )

    return prescient


def log_likelihood(returns, Sigmas, means=None, scale=1):
    """
    Computes the log likelihhod assuming Gaussian returns with covariance matrix
    Sigmas and mean vector means

    param returns: numpy array where rows are vector of asset returns
    param Sigmas: numpy array of covariance matrix
    param means: numpy array of mean vector; if None, assumes zero mean
    """
    if means is None:
        means = np.zeros_like(returns)

    T, n = returns.shape

    returns = returns.reshape(T, n, 1)
    means = means.reshape(T, n, 1)

    returns = returns * scale
    means = means * scale
    Sigmas = Sigmas * scale**2

    dets = np.linalg.det(Sigmas).reshape(len(Sigmas), 1, 1)
    Sigma_invs = np.linalg.inv(Sigmas)

    return (
        -n / 2 * np.log(2 * np.pi)
        - 1 / 2 * np.log(dets)
        - 1
        / 2
        * np.transpose(returns - means, axes=(0, 2, 1))
        @ Sigma_invs
        @ (returns - means)
    ).flatten()


def compute_regrets(returns, covariance_predictors):
    """
    Computes the regrets (for each period) of the covariance predictors
    """
    prescient = get_prescient(returns)

    # Compute prescient log likelihood
    times_pres = list(prescient.keys())
    returns_pres = returns.loc[times_pres].values
    Sigmas_pres = np.stack([prescient[t].values for t in times_pres])
    ll_pres = pd.Series(log_likelihood(returns_pres, Sigmas_pres), index=times_pres)

    # Compute log likelihood for covariance_predictors
    times_pred = list(covariance_predictors.keys())
    returns_pred = returns.loc[times_pred].values
    Sigmas_pred = np.stack([covariance_predictors[t].values for t in times_pred])
    ll_pred = pd.Series(log_likelihood(returns_pred, Sigmas_pred), index=times_pred)

    # Return regrets
    return ll_pres - ll_pred


################## SMOOTHING CODE ##################


def get_next_ewma(EWMA, y_last, t, beta, clip_at=None, min_periods=None):
    """
    param EWMA: EWMA at time t-1
    param y_last: observation at time t-1
    param t: current time step
    param beta: EWMA exponential forgetting parameter
    param clip_at: clip y_last at  +- clip_at*EWMA (optional)

    returns: EWMA estimate at time t (note that this does not depend on y_t)
    """

    old_weight = (beta - beta**t) / (1 - beta**t)
    new_weight = (1 - beta) / (1 - beta**t)

    if clip_at:
        assert min_periods, "min_periods must be specified if clip_at is specified"
        if t >= min_periods + 2:
            return old_weight * EWMA + new_weight * np.clip(
                y_last, -clip_at * EWMA, clip_at * EWMA
            )
    return old_weight * EWMA + new_weight * y_last


def ewma(y, halflife, clip_at=None, min_periods=None):
    """
    param y: array with measurements for times t=1,2,...,T=len(y)
    halflife: EWMA half life
    param clip_at: clip y_last at  +- clip_at*EWMA (optional)

    returns: list of EWMAs for times t=2,3,...,T+1 = len(y)


    Note: We define EWMA_t as a function of the
    observations up to time t-1. This means that
    y = [y_1,y_2,...,y_T] (for some T), while
    EWMA = [EWMA_2, EWMA_3, ..., EWMA_{T+1}]
    This way we don't get a "look-ahead bias" in the EWMA
    """
    times = [*y.keys()]
    beta = np.exp(-np.log(2) / halflife)
    EWMAs = {}
    EWMAs[times[0]] = y[times[0]].fillna(0)

    t = 1

    for t_prev, t_curr in zip(times[:-1], times[1:]):  # First EWMA is for t=2
        EWMAs[t_curr] = get_next_ewma(
            EWMAs[t_prev], y[t_curr].fillna(0), t + 1, beta, clip_at, min_periods
        )
        t += 1

    return EWMAs
