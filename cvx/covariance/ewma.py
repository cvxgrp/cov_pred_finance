import numpy as np
import pandas as pd
from pandas._typing import TimedeltaConvertibleTypes
from typing import Union


def _generator2frame(generator):
    return pd.DataFrame(dict(generator)).transpose()


def variance(returns, halflife, min_periods=0):
    for time, variance in ewma_mean(np.power(returns, 2), halflife=halflife, min_periods=min_periods):
        yield time, variance


def volatility(returns, halflife, min_periods=0):
    return np.sqrt(_generator2frame(variance(returns, halflife, min_periods=min_periods)))


def iterated_ewma(returns, vola_halflife, cov_halflife, lower=None, upper=None,\
    min_periods_vola=20, min_periods_cov=20):
    def scale(vola, matrix):
        # Convert (covariance) matrix to correlation matrix
        v = np.sqrt(np.diagonal(matrix))
        matrix = matrix / np.outer(v,v)


        d = pd.DataFrame(np.diag(vola), index=matrix.index, columns=matrix.columns)
        return d @ matrix @ d

    # do not adjust for the moving mean
    vola = volatility(returns=returns, halflife=vola_halflife,\
         min_periods=min_periods_vola)

    # adj the returns
    adj = (returns / vola).clip(lower=lower, upper=upper)
    # remove all the leading NaNs
    adj = adj.dropna(axis=0, how="all")

    return {t: scale(vola=vola.loc[t].values, matrix=matrix) for t, matrix in ewma_cov(data=adj, halflife=cov_halflife, min_periods=min_periods_cov)}


def ewma_cov(data, halflife, min_periods=0):
    for t, ewma in _general(data.values, times=data.index, halflife=halflife, fct=lambda x: np.outer(x,x),\
         min_periods=min_periods):
        if not np.isnan(ewma).all():
            yield t, pd.DataFrame(index=data.columns, columns=data.columns, data=ewma)


def ewma_mean(data, halflife, min_periods=0):
    for t, ewma in _general(data.values, times=data.index, halflife=halflife, min_periods=min_periods):
        # converting back into a series costs some performance but adds robustness
        if not np.isnan(ewma).all():
            yield t, pd.Series(index=data.columns, data=ewma)


def _general(y,
             times,
             com: Union[float, None] = None,
             span: Union[float, None] = None,
             halflife: Union[float, TimedeltaConvertibleTypes, None] = None,
             alpha: Union[float, None] = None,
             fct=lambda x: x,
             min_periods=0):
    """
    y: frame with measurements for times t=t_1,t_2,...,T
    halflife: EWMA half life

    returns: list of EWMAs for times t=t_1,t_2,...,T
    serving as predictions for the following timestamp

    The function returns a generator over
    t_i, EWMA of fct(y_i)
    """

    def f(k):
        if k < min_periods - 1:
            return times[k], _ewma * np.nan

        return times[k], _ewma

    if com:
        alpha = 1 / (1 + com)

    if span:
        alpha = 2 / (span + 1)

    if halflife:
        alpha = 1 - np.exp(-np.log(2)/halflife)

    beta = 1 - alpha

    # first row, important to initialize the _ewma variable
    _ewma = fct(y[0])
    yield f(k=0)

    # iterate through all the remaining rows of y. Start in the 2nd row
    for n, row in enumerate(y[1:], start=1):
        # update the moving average
        _ewma = _ewma + (1 - beta) * (fct(row) - _ewma) / (1 - np.power(beta, n+1))

        yield f(k=n)
