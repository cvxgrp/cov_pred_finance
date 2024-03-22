from __future__ import annotations

import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
from pandas._typing import TimedeltaConvertibleTypes

IEWMA = namedtuple("IEWMA", ["time", "mean", "covariance", "volatility"])


def _generator2frame(generator):
    return pd.DataFrame(dict(generator)).transpose()


def _variance(returns, halflife, min_periods=0, clip_at=None, nan_to_num=False):
    """
    Estimate variance from returns using an exponentially weighted moving average (EWMA)

    param returns: 1 dim. numpy array of returns
    param halflife: EWMA half life
    param min_periods: minimum number of observations to start EWMA (optional)
    param clip_at: clip y_last at  +- clip_at*EWMA (optional)
    """
    yield from _ewma_mean(
        np.power(returns, 2),
        halflife=halflife,
        min_periods=min_periods,
        clip_at=clip_at,
        nan_to_num=nan_to_num,
    )


def volatility(returns, halflife, min_periods=0, clip_at=None):
    """
    Estimate volatility from returns using an exponentially weighted moving average (EWMA)

    param returns: 1 dim. numpy array of returns
    param halflife: EWMA half life
    param min_periods: minimum number of observations to start EWMA (optional)
    param clip_at: clip y_last at  +- clip_at*EWMA (optional)
    """
    if clip_at:
        clip_at_var = clip_at**2
    else:
        clip_at_var = None

    return np.sqrt(
        _generator2frame(
            _variance(returns, halflife, min_periods=min_periods, clip_at=clip_at_var)
        )
    )


def center(returns, halflife, min_periods=0, mean_adj=False, nan_to_num=False):
    """
    Center returns by subtracting their exponentially weighted moving average (EWMA)

    param returns: Frame of returns
    param halflife: EWMA half life
    param min_periods: minimum number of observations to start EWMA (optional)
    param mean_adj: subtract EWMA mean from returns (optional), otherwise function returns original returns and 0.0 as mean

    return: the centered returns and their mean
    """
    if mean_adj:
        mean = _generator2frame(
            _ewma_mean(
                data=returns,
                halflife=halflife,
                min_periods=min_periods,
                nan_to_num=nan_to_num,
            )
        )
        return (returns - mean).dropna(how="all", axis=0), mean
    else:
        return returns, 0.0 * returns


def clip(data, clip_at=None):
    """
    Clip data +- clip_at

    param data: Frame of data
    param clip_at: clip data at  +- clip_at
    """
    if clip_at:
        return data.clip(lower=-clip_at, upper=clip_at).dropna(axis=0, how="all")
    else:
        return data.dropna(axis=0, how="all")


def iterated_ewma(
    returns,
    vola_halflife,
    cov_halflife,
    min_periods_vola=20,
    min_periods_cov=20,
    mean=False,
    mu_halflife1=None,
    mu_halflife2=None,
    clip_at=None,
    nan_to_num="default",
):
    mu_halflife1 = mu_halflife1 or vola_halflife
    mu_halflife2 = mu_halflife2 or cov_halflife

    def scale_cov(vola, matrix):
        index = matrix.index
        columns = matrix.columns
        matrix = matrix.values

        # Convert (covariance) matrix to correlation matrix
        v = 1 / np.sqrt(np.diagonal(matrix)).reshape(-1, 1)
        matrix = v * matrix * v.T

        cov = vola.reshape(-1, 1) * matrix * vola.reshape(1, -1)

        return pd.DataFrame(cov, index=index, columns=columns)

    def scale_mean(vola, vec1, vec2):
        return vec1 + vola * vec2

    # compute the moving mean of the returns

    # TODO: Check if this is correct half life
    returns, returns_mean = center(
        returns=returns, halflife=mu_halflife1, min_periods=0, mean_adj=mean
    )

    # estimate the volatility, clip some returns before they enter the estimation
    vola = volatility(
        returns=returns,
        halflife=vola_halflife,
        min_periods=min_periods_vola,
        clip_at=clip_at,
    )

    # wherever vola is not zero: adj = clip((returns / vola), clip_at=clip_at)
    # wherever vola is zero: adj = NaN

    # adj = returns / vola

    # adj the returns
    # make sure adj is NaN where vola is zero or returns are
    # NaN. This handles NaNs (which in some sense is the same as a constant
    # return series, i.e., vola = 0)
    adj = clip((returns / vola), clip_at=clip_at)  # if vola is zero, adj is NaN
    # print((adj.isna()).sum())
    # adj[vola == 0] = np.nan

    # center the adj returns again
    adj, adj_mean = center(adj, halflife=mu_halflife2, min_periods=0, mean_adj=mean)

    for t, matrix in _ewma_cov(
        data=adj,
        halflife=cov_halflife,
        min_periods=min_periods_cov,
        nan_to_num=nan_to_num,
    ):
        if mean:
            m = scale_mean(
                vola=vola.loc[t].values, vec1=returns_mean.loc[t], vec2=adj_mean.loc[t]
            )
        else:
            m = pd.Series(np.zeros_like(returns.shape[1]), index=returns.columns)
        yield IEWMA(
            time=t,
            mean=m,
            covariance=scale_cov(vola=vola.loc[t].values, matrix=matrix),
            volatility=vola.loc[t],
        )


def _ewma_cov(data, halflife, min_periods=0, nan_to_num=True):
    """
    param data: Txn pandas DataFrame of returns
    param halflife: float, halflife of the EWMA
    """

    for t, ewma in _general(
        data.values,
        times=data.index,
        halflife=halflife,
        fct=lambda x: np.outer(x, x),
        min_periods=min_periods,
        nan_to_num=nan_to_num,
    ):
        if not np.isnan(ewma).all():
            yield t, pd.DataFrame(index=data.columns, columns=data.columns, data=ewma)


def _ewma_mean(data, halflife, min_periods=0, clip_at=None, nan_to_num=False):
    for t, ewma in _general(
        data.values,
        times=data.index,
        halflife=halflife,
        min_periods=min_periods,
        clip_at=clip_at,
        nan_to_num=nan_to_num,
    ):
        # converting back into a series costs some performance but adds robustness
        if not np.isnan(ewma).all():
            yield t, pd.Series(index=data.columns, data=ewma)


def _general(
    y,
    times,
    halflife: float | TimedeltaConvertibleTypes | None = None,
    alpha: float | None = None,
    fct=lambda x: x,
    min_periods=0,
    clip_at=None,
    nan_to_num="default",
):
    """
    y: frame with measurements for times t=t_1,t_2,...,T
    halflife: EWMA half life
    alpha: EWMA alpha
    fct: function to apply to y
    min_periods: minimum number of observations to start EWMA
    clip_at: clip y_last at  +- clip_at*EWMA (optional)
    nan_to_num: if True, replace NaNs in y with 0.0; "default" means True if
    y[i] is 2D and False otherwise

    returns: a generator over (t_i, EWMA of fct(y_i))
    """

    def f(k):
        if k < min_periods - 1:
            return times[k], _ewma * np.nan

        return times[k], _ewma

    if nan_to_num == "default":
        if y[0].ndim == 2:
            nan_to_num = True
            y = np.nan_to_num(y)  # ensures covariances are PSD
        else:
            nan_to_num = False

    if halflife:
        alpha = 1 - np.exp(-np.log(2) / halflife)

    beta = 1 - alpha

    # first row, important to initialize the _ewma variable
    _ewma = fct(y[0])
    yield f(k=0)

    # iterate through all the remaining rows of y. Start in the 2nd row
    for n, row in enumerate(y[1:], start=1):
        # replace NaNs in row with non-NaN values from _ewma and vice versa

        next_val = fct(row)

        # ### Merge NaNs from next_val and _ewma
        nans = False
        mask_next_val = np.isnan(next_val)
        mask_ewma = np.isnan(_ewma)

        if (mask_next_val != mask_ewma).any():
            nans = True
            next_val[mask_next_val] = _ewma[mask_next_val]
            _ewma[mask_ewma] = next_val[mask_ewma]

        # update the moving average
        if clip_at and n >= min_periods + 1:
            _ewma = _ewma + (1 - beta) * (
                np.clip(next_val, -clip_at * _ewma, clip_at * _ewma) - _ewma
            ) / (1 - np.power(beta, n + 1))
        else:
            _ewma = _ewma + (1 - beta) * (next_val - _ewma) / (
                1 - np.power(beta, n + 1)
            )

        # if nans and _ewma.ndim == 2: # project _ewma onto PSD cone
        #     nan_entries = np.isnan(_ewma).all(axis=0)
        #     _ewma_clean = _ewma[~nan_entries][:, ~nan_entries]

        #     print(pd.DataFrame(_ewma_clean))

        #     eigvals, eigvecs = np.linalg.eigh(_ewma_clean)
        #     eigvals = np.maximum(eigvals, 0)
        #     _ewma_clean = eigvecs @ np.diag(eigvals) @ eigvecs.T

        #     # Update _ewma with the cleaned data
        #     non_nan_indices = np.where(~nan_entries)[0]  # Get indices of non-NaN entries
        #     _ewma[np.ix_(non_nan_indices, non_nan_indices)] = _ewma_clean
        if nans:
            warnings.warn(
                f"A new asset entered at time {times[n]} (or the first time "
                f"the return of this asset changed). This may result in "
                f"non-PSD covariance matrices. Consider using nan_to_num=True.",
                Warning,
            )

        yield f(k=n)
