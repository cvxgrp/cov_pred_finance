from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
import scipy as sc
from numpy.linalg import slogdet as logdet

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


#######
#  ADMM implementation of convex-concave procedure for low-rank plus diagonal
#  approximation
#######


def KL_div(Sigma_hat, Sigma):
    """
    Compute KL divergence between two Gaussian distributions
    """
    n = Sigma.shape[0]
    # assert Sigma and Sigma_hat are positive definite

    assert np.all(np.linalg.eigvals(Sigma) > 0), "Sigma is not positive definite"
    assert np.all(
        np.linalg.eigvals(Sigma_hat) > 0
    ), "Sigma_hat is not positive definite"

    return 0.5 * (
        logdet(Sigma_hat)[1]
        - logdet(Sigma)[1]
        - n
        + np.trace(np.linalg.inv(Sigma_hat) @ Sigma)
    )


class CCPLowrank:
    """
    Approximately solves problem
    minimize KL(Sigma, Sigma_hat)

    using the convex-concave procedure
    """

    pass


def CCP(
    Sigma,
    e_init,
    G_init,
    rho=1,
    max_iter=1000,
    eps_abs=1e-6,
    eps_rel=1e-4,
    alpha=1,
    max_ccp_iter=10,
):
    state = _State(Sigma, e_init, G_init, rho, max_iter, eps_abs, eps_rel, alpha)

    KL_divergence = np.zeros(max_ccp_iter)

    for i in range(max_ccp_iter):
        e, G, _ = state.update()

        Sigma_hat = np.linalg.inv(np.diag(e) - G @ G.T)

        KL_divergence[i] = KL_div(Sigma_hat, Sigma)

    return e, G, KL_divergence


class _State:
    """
    The convex-concave procedure state
    """

    def __init__(
        self,
        Sigma,
        e_init,
        G_init,
        rho=1,
        max_iter=1000,
        eps_abs=1e-6,
        eps_rel=1e-4,
        alpha=1,
    ) -> None:
        """
        param Sigma: covariance matrix to fit
        param e_init: initial value of the diagonal of the E matrix
        param G_init: initial value of the G matrix
        param rho: ADMM step size
        param max_iter: maximum number of ADMM iterations
        param eps_abs: absolute tolerance
        param eps_rel: relative tolerance
        param alpha: over-relaxation parameter
        """
        self.Sigma = Sigma
        self.e = e_init
        self.G = G_init
        self.rho = rho
        self.max_iter = max_iter
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.alpha = alpha

    @property
    def n(self):
        return self.e.shape[0]

    @property
    def k(self):
        return self.G.shape[1]

    @property
    def X(self):
        return self._construct_X(self.e, self.G)

    def update(self):
        e, G, history = self._ADMM_ccp_iteration()
        self.e = e
        self.G = G

        return e, G, history

    def _ADMM_ccp_iteration(self):
        # Sigma, ek, Gk, rho=1, max_iter=1000, eps_abs=1e-6, eps_rel=1e-4, alpha=1):
        """
        Solves problem
        minimize -log det(X) + trace(Sigma*E) - 2*(Gk)^T * Sigma * G
        subject to X = [E G; G^T I]

        using ADMM
        """
        # n, k = Gk.shape # n number of features, k number of factors

        # Initialize variables
        # Xinit = _construct_X(ek, Gk)
        Sigma = self.Sigma
        rho = self.rho
        n, k = self.n, self.k
        _, Gk = self.e, self.G
        eps_abs, eps_rel = self.eps_abs, self.eps_rel

        Xinit = self.X

        X = Xinit
        Z = Xinit
        U = np.zeros_like(X)

        # History
        primal_res = np.zeros(self.max_iter)
        dual_res = np.zeros(self.max_iter)

        primal_eps = np.zeros(self.max_iter)
        dual_eps = np.zeros(self.max_iter)

        obj_vals_ests = np.zeros(self.max_iter)
        obj_vals_true = np.zeros(self.max_iter)

        KL_divergence = np.zeros(self.max_iter)

        history = {
            "primal_res": primal_res,
            "dual_res": dual_res,
            "primal_eps": primal_eps,
            "dual_eps": dual_eps,
            "obj_vals_ests": obj_vals_ests,
            "obj_vals_true": obj_vals_true,
            "KL_divergence": KL_divergence,
        }

        # ADMM loop
        for i in range(self.max_iter):
            # Update X
            X = self.update_X(Sigma, Gk, U, Z, rho)

            # over-relaxation
            Xhat = self.alpha * X + (1 - self.alpha) * Z

            # Update Z
            Z_old = Z
            Z = self.update_Z(Xhat, U, k)

            # Update U
            U = U + Xhat - Z

            # Compute residuals
            primal_res[i] = np.linalg.norm(X - Z)
            dual_res[i] = rho * np.linalg.norm(Z - Z_old)

            # Compute epsilons
            primal_eps[i] = eps_abs * np.sqrt(n * k) + eps_rel * max(
                np.linalg.norm(X), np.linalg.norm(Z)
            )
            dual_eps[i] = eps_abs * np.sqrt(n * k) + eps_rel * np.linalg.norm(rho * U)

            # Compute objective value
            obj_vals_ests[i] = self.objective_value_estimate(X, Sigma, Gk)
            obj_vals_true[i] = self.objective_value_true(X, Sigma, Gk)

            E = X[:n, :n]
            G = X[:n, n:]
            Sigma_hat = np.linalg.inv(E - G @ G.T)
            KL_divergence[i] = KL_div(Sigma_hat, Sigma)

            # Check convergence
            if primal_res[i] < primal_eps[i] and dual_res[i] < dual_eps[i]:
                break

        history = pd.DataFrame(history).iloc[: i + 1]

        e = np.diag(X[:n, :n])
        G = X[:n, n:]

        return e, G, history

    @staticmethod
    def objective_value_estimate(X, Sigma, Gk):
        n, _ = Gk.shape
        E = X[:n, :n]
        G = X[:n, n:]

        return -logdet(X)[1] + np.trace(Sigma @ E) - 2 * np.trace(Gk.T @ Sigma @ G)

    @staticmethod
    def objective_value_true(X, Sigma, Gk):
        n, k = Gk.shape
        E = X[:n, :n]
        G = X[:n, n:]

        X_true = np.vstack([np.hstack([E, G]), np.hstack([G.T, np.eye(k)])])

        return -logdet(X_true)[1] + np.trace(Sigma @ E) - 2 * np.trace(Gk.T @ Sigma @ G)

    @staticmethod
    def _construct_X(e, G):
        """
        Construct X matrix from E and G
        """
        n = e.shape[0]
        k = G.shape[1]

        X = np.zeros((n + k, n + k))
        X[:n, :n] = np.diag(e)
        X[:n, n:] = G
        X[n:, :n] = G.T
        X[n:, n:] = np.eye(k)

        return X

    @staticmethod
    def update_X(Sigma, Gk, U, Z, rho):
        """
        X-update ADMM step
        """

        n, k = Gk.shape

        Sigma_at_Gk = Sigma @ Gk

        V = np.zeros((n + k, n + k))
        W = np.zeros((n + k, n + k))
        V[:n, :n] = Sigma
        W[:n, n:] = Sigma_at_Gk
        W[n:, :n] = Sigma_at_Gk.T

        RHS = rho * (Z - U) + W - V

        eigvals, eigvecs = np.linalg.eigh(RHS)
        Xtilde_diag = ((eigvals + np.sqrt(eigvals**2 + 4 * rho)) / (2 * rho)).reshape(
            -1, 1
        )
        X = (eigvecs * Xtilde_diag.T) @ eigvecs.T

        return X

    @staticmethod
    def update_Z(X, U, k):
        """
        Z-update ADMM step
        """
        n = X.shape[0] - k
        matrix = X + U

        E = np.diag(np.diag(matrix[:n, :n]))
        Identity = np.eye(k)

        # Project onto the feasible set
        matrix[:n, :n] = E
        matrix[n:, n:] = Identity

        return matrix


# def cvxpy_ccp_iteration(Sigma, Gk, solver):
#     """
#     Solves problem
#     minimize -log det(X) + trace(Sigma*E) - 2*(Gk)^T * Sigma * G
#     subject to X = [E G; G^T I]

#     using cvxpy
#     """
#     n, k = Gk.shape # n number of features, k number of factors

#     # Define variables
#     G = cp.Variable((n,k))
#     e = cp.Variable(n)
#     E = cp.diag(e)

#     X = cp.vstack([cp.hstack([E, G]), cp.hstack([G.T, np.eye(k)])])

#     # Define objective
#     obj = -cp.log_det(X) + cp.trace(Sigma @ E) - 2 * cp.trace(Gk.T @ Sigma @ G)

#     # Define problem
#     prob = cp.Problem(cp.Minimize(obj))

#     # Solve problem
#     prob.solve(verbose=False, solver=solver)

#     E, G = X[:n,:n].value, X[:n,n:].value

#     return E, G, prob.value
