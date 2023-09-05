from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
import scipy as sc

# from tqdm import tqdm

LowRank = namedtuple("LowRank", ["Loading", "Lambda", "D", "Approximation"])
LowRankDiag = namedtuple("LowRankCovariance", ["F", "d"])


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
    return LowRank(Loading=Q, Lambda=lamda, D=D, Approximation=R_lo + D)
    # return R_lo + D


def regularize_covariance(sigmas, r, low_rank_format=False):
    """
    param Sigmas: dictionary of covariance matrices
    param r: float, rank of low rank component

    returns: regularized covariance matrices according to "Factor form
    regularization." of Section 7.2 in the paper "A Simple Method for Predicting Covariance Matrices of Financial Returns"
    """
    for time, sigma in sigmas.items():
        vola = np.sqrt(np.diag(sigma)).reshape(-1, 1)
        # R = sigma / np.outer(vola, vola)
        R = (1 / vola) * sigma * (1 / vola).T
        R = _regularize_correlation(R, r)
        # todo: requires some further work
        if not low_rank_format:
            cov = vola.reshape(-1, 1) * R.Approximation * vola.reshape(1, -1)
            yield time, pd.DataFrame(cov, index=sigma.columns, columns=sigma.columns)
        else:
            F = vola.reshape(-1, 1) * R.Loading * np.sqrt(R.Lambda)
            d = np.diag(vola.reshape(-1, 1) * R.D * vola.reshape(1, -1))
            yield time, LowRankDiag(
                F=pd.DataFrame(F, index=sigma.columns),
                d=pd.Series(d, index=sigma.columns),
            )


##### Expectation-Maximization algorithm proposed by Emmanuel Candes #####


def _e_step(Sigma, F, d):
    G = np.linalg.inv((F.T * (1 / d.values.reshape(1, -1))) @ F + np.eye(F.shape[1]))
    L = G @ F.T * (1 / d.values.reshape(1, -1))
    Cxx = Sigma
    Cxs = Sigma @ L.T
    Css = L @ (Sigma @ L.T) + G

    return Cxx, Cxs, Css


def _m_step(Cxx, Cxs, Css):
    F = Cxs @ np.linalg.inv(Css)
    # d = np.diag(Cxx - 2 * Cxs @ F.T + F @ Css @ F.T)
    d = np.diag(Cxx) - 2 * np.sum(Cxs * F, axis=1) + np.sum(F * (F @ Css), axis=1)
    return LowRankDiag(F=F, d=pd.Series(d, index=F.index))


def _em_low_rank_approximation(sigma, initial_sigma):
    """
    param sigma: one covariance matrices
    param rank: float, rank of low rank component

    returns: regularized covariance matrices
    """
    assets = sigma.columns
    F = initial_sigma.F
    d = initial_sigma.d

    for _ in range(5):
        Cxx, Cxs, Css = _e_step(sigma, F, d)
        F, d = _m_step(Cxx, Cxs, Css)

    return LowRankDiag(F=pd.DataFrame(F, index=assets), d=pd.Series(d, index=assets))


def em_regularize_covariance(sigmas, initial_sigmas):
    """
    param sigmas: dictionary of covariance matrices
    param  initial_sigmas: dictionary of initial low rank + diagonal approximations; these
    are namedtuples with fields F and d, with F nxk and d length n

    returns: regularized covariance matrices
    """

    # for time, sigma in tqdm(sigmas.items()):
    for time, sigma in sigmas.items():
        yield time, _em_low_rank_approximation(sigma, initial_sigmas[time])
