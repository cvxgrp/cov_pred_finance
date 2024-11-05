from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvx.covariance.regularization import em_regularize_covariance
from cvx.covariance.regularization import regularize_covariance
from experiments.utils.iterated_ewma_vec import ewma


def _realized_covariance(returns):
    for time in returns.index.unique():
        returns_temp = returns.loc[time]
        T_temp = len(returns_temp)

        yield time, returns_temp.cov(ddof=0) * T_temp


def realized_ewma(returns, halflife, clip_at=None, min_periods=None):
    realized_covariances = dict(_realized_covariance(returns))

    return ewma(
        realized_covariances,
        halflife,
        clip_at=clip_at,
        min_periods=min_periods,
    )


def realized_volas(returns):
    for time in returns.index.unique():
        yield time, np.sqrt((returns.loc[time] ** 2).sum(axis=0))


def MSE(returns, covariances):
    returns_shifted = returns.shift(-1)

    MSEs = []
    for time, cov in covariances.items():
        realized_cov = returns_shifted.loc[time].values.reshape(
            -1, 1
        ) @ returns_shifted.loc[time].values.reshape(1, -1)
        MSEs.append(np.linalg.norm(cov - realized_cov) ** 2)

    return pd.Series(MSEs, index=covariances.keys())


def yearly_SR(trader, plot=True, regression_line=True):
    rets = pd.Series(trader.rets.flatten(), index=trader.returns.index)

    # Only keep years with more than 100 trading days
    rets = rets.groupby(rets.index.year).filter(lambda x: len(x) > 100)

    means = rets.resample("Y").mean() * 252
    stds = rets.resample("Y").std() * np.sqrt(252)

    SRs = means / stds

    if plot:
        # Fit regression line to SRs
        coefficients = np.polyfit(SRs.index.year + 1, SRs.values, 1)
        sr_pred = np.polyval(coefficients, SRs.index.year)

        plt.plot(SRs, marker="o")
        if regression_line:
            plt.plot(SRs.index, sr_pred, "--", color="red", alpha=0.5, label="Trend")
        plt.ylabel("Sharpe ratio")
        plt.legend()
    return SRs


def turnover(weights):
    """
    Computes average turnover for a sequence of weights,
    i.e., mean of |w_{t+1}-w_{t}|_1
    """
    w_diff = weights[1:] - weights[:-1]
    w_old = weights[:-1]

    daily_turnover = np.mean(
        np.sum(np.abs(w_diff), axis=1) / np.sum(np.abs(w_old), axis=1)
    )

    yearly_turnover = 252 * daily_turnover

    return yearly_turnover * 100


def _single_log_likelihood(r, Sigma, m):
    n = len(r)
    Sigma_inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    return (
        -n / 2 * np.log(2 * np.pi)
        - 1 / 2 * np.log(det)
        - 1 / 2 * (r - m).T @ Sigma_inv @ (r - m)
    )


def log_likelihood_low_rank(returns, Sigmas, means=None):
    """
    Computes the log-likelihoods

    param returns: pandas DataFrame of returns
    param Sigmas: dictionary of covariance matrices where each covariance matrix
                  is a namedtuple with fields "F" and "d"
    param means: pandas DataFrame of means

    Note: Sigmas[time] is covariance prediction for returns[time+1]
        same for means.loc[time]
    """
    returns = returns.shift(-1)

    ll = []
    m = np.zeros_like(returns.iloc[0].values).reshape(-1, 1)

    times = []

    for time, low_rank in Sigmas.items():
        # TODO: forming the covariance matrix is bad...
        cov = low_rank.F @ (low_rank.F).T + np.diag(low_rank.d)

        if not returns.loc[time].isna()[0]:
            if means is not None:
                m = means.loc[time].values.reshape(-1, 1)
            ll.append(
                _single_log_likelihood(
                    returns.loc[time].values.reshape(-1, 1), cov.values, m
                )
            )
            times.append(time)

    return pd.Series(ll, index=times).astype(float)


def log_likelihood_regularized(returns, Sigmas, means=None, r=None):
    """
    Helper function to avoid storing all the covariance matrices in memory for
    large universe experiments

    param returns: pandas DataFrame of returns
    param Sigmas: dictionary of covariance matrices
    param r: float, rank of low rank component

    Note: Sigmas[time] is covariance prediction for returns[time+1]
    """
    returns = returns.shift(-1)

    ll = []
    m = np.zeros_like(returns.iloc[0].values).reshape(-1, 1)

    times = []

    if r is not None:
        for time, cov in regularize_covariance(Sigmas, r=r):
            if not returns.loc[time].isna()[0]:
                if means is not None:
                    m = means.loc[time].values.reshape(-1, 1)
                ll.append(
                    _single_log_likelihood(
                        returns.loc[time].values.reshape(-1, 1), cov.values, m
                    )
                )
                times.append(time)
    else:
        for time, cov in Sigmas.items():
            if not returns.loc[time].isna()[0]:
                if means is not None:
                    m = means.loc[time].values.reshape(-1, 1)
                ll.append(
                    _single_log_likelihood(
                        returns.loc[time].values.reshape(-1, 1), cov.values, m
                    )
                )
                times.append(time)

    return pd.Series(ll, index=times).astype(float)


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


def log_likelihood_sequential(returns, Sigmas, means=None, scale=1):
    """
    Computes Gaussian log likelihood sequentially
    """
    if means is None:
        means = np.zeros_like(returns)

    T, n = returns.shape

    returns = returns.reshape(T, n, 1)
    means = means.reshape(T, n, 1)

    returns = returns * scale
    means = means * scale
    Sigmas = Sigmas * scale**2

    ll = np.zeros(T)

    for t in range(T):
        r = returns[t].reshape(n, 1)
        m = means[t].reshape(n, 1)
        Sigma = Sigmas[t]
        det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)

        ll[t] = (
            -n / 2 * np.log(2 * np.pi)
            - 1 / 2 * np.log(det)
            - 1 / 2 * (r - m).T @ Sigma_inv @ (r - m)
        )

    return ll


def rolling_window(returns, memory, min_periods=20):
    min_periods = max(min_periods, 1)

    times = returns.index
    assets = returns.columns

    returns = returns.values

    Sigmas = np.zeros((returns.shape[0], returns.shape[1], returns.shape[1]))
    Sigmas[0] = np.outer(returns[0], returns[0])

    for t in range(1, returns.shape[0]):
        alpha_old = 1 / min(t + 1, memory)
        alpha_new = 1 / min(t + 2, memory)

        if t >= memory:
            Sigmas[t] = alpha_new / alpha_old * Sigmas[t - 1] + alpha_new * (
                np.outer(returns[t], returns[t])
                - np.outer(returns[t - memory], returns[t - memory])
            )
        else:
            Sigmas[t] = alpha_new / alpha_old * Sigmas[t - 1] + alpha_new * (
                np.outer(returns[t], returns[t])
            )

    Sigmas = Sigmas[min_periods - 1 :]
    times = times[min_periods - 1 :]

    return {
        times[t]: pd.DataFrame(Sigmas[t], index=assets, columns=assets)
        for t in range(len(times))
    }


def from_row_to_covariance(row, n):
    """
    Convert upper diagonal part of covariance matrix to a covariance matrix
    """
    Sigma = np.zeros((n, n))

    # set upper triangular part
    upper_mask = np.triu(np.ones((n, n)), k=0).astype(bool)
    Sigma[upper_mask] = row

    # set lower triangular part
    lower_mask = np.tril(np.ones((n, n)), k=0).astype(bool)
    Sigma[lower_mask] = Sigma.T[lower_mask]
    return Sigma


def from_row_matrix_to_covariance(M, n):
    """
    Convert Tx(n(n+1)/2) matrix of upper diagonal parts of covariance matrices to a Txnxn matrix of covariance matrices
    """
    Sigmas = []
    T = M.shape[0]
    for t in range(T):
        Sigmas.append(from_row_to_covariance(M[t], n))
    return np.array(Sigmas)


def add_to_diagonal(Sigmas, lamda):
    """
    Adds lamda*diag(Sigma) to each covariance (Sigma) matrix in Sigmas

    param Sigmas: dictionary of covariance matrices
    param lamda: scalar
    """
    for key in Sigmas.keys():
        Sigmas[key] = Sigmas[key] + lamda * np.diag(np.diag(Sigmas[key]))

    return Sigmas


def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y


def em_low_rank_log_likelihood(returns, Sigmas, rank):
    Sigmas_low_rank = dict(regularize_covariance(Sigmas, r=rank, low_rank_format=True))
    Sigmas_em = dict(em_regularize_covariance(Sigmas, Sigmas_low_rank))

    return log_likelihood_low_rank(returns, Sigmas_em)
