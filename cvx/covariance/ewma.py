import numpy as np
import pandas as pd
from pandas._typing import TimedeltaConvertibleTypes
from typing import Union
from collections import namedtuple

IEWMA = namedtuple('IEWMA', ['time', 'mean', 'covariance'])


def _generator2frame(generator):
    return pd.DataFrame(dict(generator)).transpose()


def variance(returns, halflife, min_periods=0, clip_at=None):
    for time, variance in ewma_mean(np.power(returns, 2), halflife=halflife, min_periods=min_periods, clip_at=clip_at):
        yield time, variance

def volatility(returns, halflife, min_periods=0, clip_at=None):
    return np.sqrt(_generator2frame(variance(returns, halflife, min_periods=min_periods, clip_at=clip_at)))

def _regularize_correlation(R, r):
    """
    param Rs: Txnxn numpy array of correlation matrices
    param r: float, rank of low rank component 

    returns: low rank + diag approximation of R\
        R_hat = sum_i^r lambda_i q_i q_i' + E, where E is diagonal,
        defined so that R_hat has unit diagonal; lamda_i, q_i are eigenvalues
        and eigenvectors of R (the r first, in descending order)
    """
    Lamda, Q = np.linalg.eigh(R)

    # Sort eigenvalues in descending order
    Lamda = np.diag(Lamda[::-1])
    # reshuffle the eigenvectors accordingly
    Q = Q[ :, ::-1]

    # Get low rank component
    Lamda_r = Lamda[:r, :r]
    Q_r = Q[:, :r]
    R_lo = Q_r @ Lamda_r @ Q_r.T

    # Get diagonal component 
    D = np.diag(np.diag(R-R_lo))
    
    # Create low rank approximation
    return R_lo + D

def regularize_covariance(Sigmas, r):
    """
    param Sigmas: dictionary of covariance matrices
    param r: float, rank of low rank component 

    returns: regularized covariance matrices according to "Factor form
    regularization." of Section 7.2 in the paper "A Simple Method for Predicting Covariance Matrices of Financial Returns"
    """
    for time, Sigma in Sigmas.items():
        vola = np.sqrt(np.diag(Sigma))
        R = Sigma / np.outer(vola, vola)
        R = _regularize_correlation(R, r)
        cov = vola.reshape(-1,1) * R * vola.reshape(1,-1)
        yield time, pd.DataFrame(cov, index=Sigma.columns, columns=Sigma.columns)

def iterated_ewma(returns, vola_halflife, cov_halflife,\
    min_periods_vola=20, min_periods_cov=20, mean=False, clip_at=None):
    def scale_cov(vola, matrix):
        index = matrix.index
        columns = matrix.columns
        matrix = matrix.values

        # Convert (covariance) matrix to correlation matrix
        v = np.sqrt(np.diagonal(matrix))
        matrix = matrix / np.outer(v,v)

        cov = vola.reshape(-1,1) * matrix * vola.reshape(1,-1)

        return pd.DataFrame(cov, index=index, columns=columns)
    def scale_mean(vola, vec1, vec2):
        return vec1 + vola * vec2

    if mean:
        returns_mean = _generator2frame(ewma_mean(data=returns, halflife=vola_halflife, min_periods=0))
    else:
        returns_mean = 0.0*returns
    if clip_at:
        clip_at_var = clip_at**2
    else:
        clip_at_var = None
    vola = volatility(returns=returns-returns_mean, halflife=vola_halflife,\
         min_periods=min_periods_vola, clip_at=clip_at_var)

    if clip_at: # Clip returns
        returns = returns.clip(returns_mean-clip_at*vola, returns_mean+clip_at*vola) 


    # adj the returns
    if clip_at:
        adj = ((returns-returns_mean) / vola).clip(lower=-clip_at, upper=clip_at)
    else:
        adj = (returns-returns_mean) / vola
    # remove all the leading NaNs
    adj = adj.dropna(axis=0, how="all")

    if mean:
        adj_mean = _generator2frame(ewma_mean(data=adj, halflife=cov_halflife, min_periods=0))
    else:
        adj_mean = 0.0*adj

    if mean:

        covs = {t: scale_cov(vola=vola.loc[t].values, matrix=matrix) for t, matrix in ewma_cov(data=adj-adj_mean, halflife=cov_halflife, min_periods=min_periods_cov)}

        means = {t: scale_mean(vola=vola.loc[t].values, vec1=returns_mean.loc[t], vec2=adj_mean.loc[t]) for t in covs.keys()}

        for time in covs.keys():
            yield IEWMA(time=time, mean=means[time], covariance=covs[time]) 

    else:
        zero_mean = pd.Series(np.zeros_like(returns.shape[1]), index=returns.columns)

        for t, matrix in ewma_cov(data=adj-adj_mean, halflife=cov_halflife, min_periods=min_periods_cov):
            yield IEWMA(time=t, mean=zero_mean, covariance=scale_cov(vola=vola.loc[t].values, matrix=matrix))


def ewma_cov(data, halflife, min_periods=0):
    for t, ewma in _general(data.values, times=data.index, halflife=halflife, fct=lambda x: np.outer(x,x),\
         min_periods=min_periods):
        if not np.isnan(ewma).all():
            yield t, pd.DataFrame(index=data.columns, columns=data.columns, data=ewma)


def ewma_mean(data, halflife, min_periods=0, clip_at=None):
    for t, ewma in _general(data.values, times=data.index, halflife=halflife, min_periods=min_periods, clip_at=clip_at):
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
             min_periods=0,
             clip_at=None):
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
        if clip_at and n>=min_periods+1: 
            _ewma = _ewma + (1 - beta) * (np.clip(fct(row), -clip_at*_ewma, clip_at*_ewma) - _ewma) / (1 - np.power(beta, n+1))
        else:
            _ewma = _ewma + (1 - beta) * (fct(row) - _ewma) / (1 - np.power(beta, n+1))

        yield f(k=n)