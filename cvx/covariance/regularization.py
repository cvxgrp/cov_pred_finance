# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
import scipy as sc

LowRank = namedtuple("LowRank", ["Loading", "Cov", "D", "Approximation"])


def _regularize_correlation(R, r):
    """
    param Rs: nxn numpy array of correlation matrices
    param r: float, rank of low rank component

    returns: low rank + diag approximation of R\
        R_hat = sum_i^r lambda_i q_i q_i' + E, where E is diagonal,
        defined so that R_hat has unit diagonal; lamda_i, q_i are eigenvalues
        and eigenvectors of R (the r first, in descending order)
    """
    # number of rows and columns in R
    n = R.shape[0]

    # ascending order of the largest r eigenvalues
    lamda, Q = sc.linalg.eigh(a=R, subset_by_index=[n - r, n - 1])

    # Sort eigenvalues in descending order
    lamda = lamda[::-1]

    # reshuffle the eigenvectors accordingly
    Q = Q[:, ::-1]

    # Get low rank component
    R_lo = Q @ np.diag(lamda) @ Q.T

    # Get diagonal component
    D = np.diag(np.diag(R - R_lo))

    # Create low rank approximation
    return LowRank(Loading=Q, Cov=lamda, D=D, Approximation=R_lo + D)
    # return R_lo + D


def regularize_covariance(sigmas, r):
    """
    param Sigmas: dictionary of covariance matrices
    param r: float, rank of low rank component

    returns: regularized covariance matrices according to "Factor form
    regularization." of Section 7.2 in the paper "A Simple Method
    for Predicting Covariance Matrices of Financial Returns"
    """
    for time, sigma in sigmas.items():
        vola = np.sqrt(np.diag(sigma))
        R = sigma / np.outer(vola, vola)
        R = _regularize_correlation(R, r)
        # todo: requires some further work
        cov = vola.reshape(-1, 1) * R.Approximation * vola.reshape(1, -1)
        yield time, pd.DataFrame(cov, index=sigma.columns, columns=sigma.columns)
