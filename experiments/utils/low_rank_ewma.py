# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import namedtuple
from typing import Union

import numpy as np
import pandas as pd
from pandas._typing import TimedeltaConvertibleTypes
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator

LowRankCovariance = namedtuple("LowRankCovariance", ["F", "d"])


def ewma_low_rank(data, halflife, min_periods=0, rank=5):
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
