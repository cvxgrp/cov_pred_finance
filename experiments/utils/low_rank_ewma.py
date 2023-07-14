# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import namedtuple
from typing import Union

import numpy as np
import pandas as pd
from pandas._typing import TimedeltaConvertibleTypes
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator

from cvx.covariance.ewma import center
from cvx.covariance.ewma import clip
from cvx.covariance.ewma import volatility

LowRankCovariance = namedtuple("LowRankCovariance", ["F", "d"])
IEWMA = namedtuple("IEWMA", ["time", "mean", "covariance", "volatility"])


def _get_diagonal(F, d):
    return np.sum(F**2, axis=1) + d


def low_rank_iterated_ewma(
    returns,
    vola_halflife,
    cov_halflife,
    rank,
    min_periods_vola=20,
    min_periods_cov=20,
    mean=False,
    mu_halflife1=None,
    mu_halflife2=None,
    clip_at=None,
):
    mu_halflife1 = mu_halflife1 or vola_halflife
    mu_halflife2 = mu_halflife2 or cov_halflife

    def scale_low_rank(vola, low_rank):
        F_temp = low_rank.F
        d_temp = low_rank.d

        index = F_temp.index
        columns = F_temp.columns
        F_temp = F_temp.values
        d_temp = d_temp.values

        # Convert (covariance) matrix to correlation matrix
        v = 1 / np.sqrt(_get_diagonal(F_temp, d_temp).reshape(-1, 1))
        F = v * F_temp
        d = (v.flatten() ** 2) * d_temp

        F = vola.reshape(-1, 1) * F
        d = (vola**2) * d

        return LowRankCovariance(
            F=pd.DataFrame(F, index=index, columns=columns), d=pd.Series(d, index=index)
        )

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

    # adj the returns
    adj = clip((returns / vola), clip_at=clip_at)

    # center the adj returns again? Yes, I think so
    # TODO: Check if this is correct half life

    adj, adj_mean = center(adj, halflife=mu_halflife2, min_periods=0, mean_adj=mean)
    # if mean:
    #     print(adj)
    #     print(adj_mean)
    #     assert False

    m = pd.Series(np.zeros_like(returns.shape[1]), index=returns.columns)

    for t, low_rank in _ewma_low_rank(
        data=adj, halflife=cov_halflife, min_periods=min_periods_cov, rank=rank
    ):
        if mean:
            m = scale_mean(
                vola=vola.loc[t].values, vec1=returns_mean.loc[t], vec2=adj_mean.loc[t]
            )

        yield IEWMA(
            time=t,
            mean=m,
            covariance=scale_low_rank(vola=vola.loc[t].values, low_rank=low_rank),
            volatility=vola.loc[t],
        )


def _ewma_low_rank(data, halflife, min_periods=0, rank=5):
    """
    param data: Txn pandas DataFrame of returns
    param halflife: float, halflife of the EWMA
    """
    for t, low_rank_covariance in _general_low_rank(
        data.values,
        times=data.index,
        halflife=halflife,
        min_periods=min_periods,
        rank=rank,
    ):
        F_t = low_rank_covariance.F
        d_t = low_rank_covariance.d
        if not np.isnan(F_t).all():
            yield t, LowRankCovariance(
                pd.DataFrame(
                    index=data.columns, columns=np.arange(F_t.shape[1]), data=F_t
                ),
                pd.Series(index=data.columns, data=d_t),
            )


def _low_rank_sum(F, d, r):
    """
    D = diag(d)

    Approximates sum FF^T + D + rr^T as low rank plus diagonal matrix:
        \hat{F}\hat{F}^T + \hat{D}

    returns \hat{F}, diag(\hat{D})
    """
    n = F.shape[0]
    rank = F.shape[1]
    r = r.reshape(-1, 1)
    d = d.reshape(-1, 1)

    def _mv(x):
        """ "
        returns (FF^T + D + rr^T)x
        """
        shape = x.shape
        x = x.reshape(-1, 1)

        return (
            ((scale * F) @ ((F.T * scale.T) @ x)) + (scale * r) * (r.T * scale.T) @ x
        ).reshape(shape)

    def _get_new_diag(F_hat):
        """
        returns diag(\hat{D})
        """

        return (
            np.ones((n, 1))
            + d * (scale**2)
            - np.sum(F_hat**2, axis=1).reshape(-1, 1)
        )

    # Get scaling toward correlation matrix
    scale = 1 / np.sqrt(np.sum(F**2, axis=1).reshape(-1, 1) + r**2)

    A = LinearOperator((n, n), matvec=_mv)

    # Compute top eigenvalues
    lamda, Q = eigsh(A, k=rank, which="LM")

    # Replace negative lamda with 0
    lamda[lamda < 0] = 0

    # Get new low rank component (unscaled)
    F_hat_temp = Q * np.sqrt(lamda).reshape(1, -1)

    # Get new diagonal
    d_hat = 1 / (scale) * _get_new_diag(F_hat_temp) * 1 / (scale)

    # Scale back low rank component
    F_hat = (1 / scale) * F_hat_temp

    # Replace negative d_hat with 0
    d_hat[d_hat < 0] = 0

    return LowRankCovariance(F_hat, d_hat.flatten())


def _general_low_rank(
    y,
    times,
    halflife: Union[float, TimedeltaConvertibleTypes, None] = None,
    alpha: Union[float, None] = None,
    rank=5,
    min_periods=0,
):
    """
    y: frame with measurements for times t=t_1,t_2,...,T
    halflife: EWMA half life

    returns: list of EWMAs for times t=t_1,t_2,...,T
    serving as predictions for the following timestamp

    The function returns a generator over
    t_i, EWMA of fct(y_i)
    """
    n = y.shape[1]

    def f(k):
        if k < min_periods - 1:
            return times[k], LowRankCovariance(
                np.nan * _low_rank.F, np.nan * _low_rank.d
            )

        return times[k], LowRankCovariance(_low_rank.F, _low_rank.d)

    if halflife:
        alpha = 1 - np.exp(-np.log(2) / halflife)

    beta = 1 - alpha

    # first row, important to initialize the _ewma variable
    F_init = np.zeros((n, rank))
    F_init[:, 0] = y[0]
    d_init = np.zeros(n)
    _low_rank = LowRankCovariance(F_init, d_init)
    yield f(k=0)

    # iterate through all the remaining rows of y. Start in the 2nd row
    for n, row in enumerate(y[1:], start=1):
        coef_old = (beta - beta ** (n + 1)) / (1 - beta ** (n + 1))
        coef_new = (1 - beta) / (1 - beta ** (n + 1))

        F_scaled = _low_rank.F * np.sqrt(coef_old)
        d_scaled = _low_rank.d * coef_old
        r_scaled = row * np.sqrt(coef_new)

        _low_rank = _low_rank_sum(F_scaled, d_scaled, r_scaled)

        yield f(k=n)
