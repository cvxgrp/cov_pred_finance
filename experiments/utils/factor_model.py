# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from tqdm import trange

from cvx.covariance.combination import from_sigmas
from cvx.covariance.ewma import iterated_ewma
from cvx.linalg import pca
from experiments.utils.experiment_utils import _single_log_likelihood
from experiments.utils.experiment_utils import add_to_diagonal


def construct_factor_covariance(factors, cm_iewma_pairs, start_date):
    # CM-IEWMA
    iewmas = {
        f"{pair[0]}-{pair[1]}": list(
            iterated_ewma(
                factors,
                vola_halflife=pair[0],
                cov_halflife=pair[1],
                min_periods_vola=factors.shape[1],
                min_periods_cov=2 * factors.shape[1],
            )
        )
        for pair in cm_iewma_pairs
    }

    Sigmas = {
        key: {item.time: item.covariance for item in iewma}
        for key, iewma in iewmas.items()
    }

    # Only keep Sigmas after start_date
    Sigmas = {
        key: {time: Sigma for time, Sigma in Sigma_dict.items() if time >= start_date}
        for key, Sigma_dict in Sigmas.items()
    }

    # Regularize the first covariance matrix
    fast = cm_iewma_pairs[0]
    fast = f"{fast[0]}-{fast[1]}"
    Sigmas[fast] = add_to_diagonal(Sigmas[fast], lamda=0.05)
    results = list(from_sigmas(Sigmas, factors, means=None).solve(window=10))

    cm_iewma = {result.time: result.covariance for result in results}

    return cm_iewma


FactorModel = namedtuple(
    "FactorModel", ["time", "exposure", "factor_covariance", "diagonal"]
)


def construct_cmiewma_factor_model(
    returns, k, cm_iewma_pairs, update_frequency=20, memory=250
):
    """
    Constructs the causal factor model

    param returns: pd.DataFrame
    param k: int
    param cm_iewma_pairs: list of tuples
    param update_frequency: int
    param memory: int
    """
    # Loop over time every update_frequency days
    times = returns.index
    T = len(times)

    for t in trange(memory, T, update_frequency):
        train_end = times[t - 1]
        train_start = times[t - memory]

        test_start = times[t - 10]  # -10 for CM-IEWMA
        test_end = times[min(t + update_frequency - 1, T - 1)]

        returns_train = returns.loc[
            (returns.index <= train_end) & (returns.index >= train_start)
        ]
        returns_test = returns.loc[returns.index <= test_end]

        factor_model_temp = pca(returns_train, n_components=k)
        exposure_temp = factor_model_temp.exposure
        factors_temp = returns_test @ np.transpose(exposure_temp)
        unexplained_temp = pd.DataFrame(
            returns_test.values - factors_temp.values @ exposure_temp.values,
            index=returns_test.index,
            columns=returns.columns,
        )

        factor_covariance_temp = construct_factor_covariance(
            factors_temp, cm_iewma_pairs, test_start
        )
        diag_temp = (unexplained_temp**2).ewm(halflife=21, min_periods=0).mean()

        for time, covariance in factor_covariance_temp.items():
            factor_covariance = covariance
            diagonal = diag_temp.loc[time]

            yield FactorModel(time, exposure_temp, factor_covariance, diagonal)


def from_factor_to_full_covariance(factor_model):
    covariance = {}

    for item in factor_model:
        covariance[
            item.time
        ] = item.exposure.T @ item.factor_covariance @ item.exposure + np.diag(
            item.diagonal
        )

    return covariance


def factor_log_likelihood(returns, factor_model):
    # Make causal
    returns = returns.shift(-1)

    m = np.zeros((returns.shape[1], 1))

    log_likelihoods = np.zeros(len(factor_model))

    times = []
    for i, item in enumerate(factor_model):
        r = returns.loc[item.time].values.reshape(-1, 1)
        Sigma = item.exposure.T @ item.factor_covariance @ item.exposure + np.diag(
            item.diagonal
        )

        times.append(item.time)

        log_likelihoods[i] = _single_log_likelihood(r, Sigma, m)

    return pd.Series(log_likelihoods, index=times)
