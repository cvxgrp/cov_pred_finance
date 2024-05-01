from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
import scipy as sc
from numpy.linalg import slogdet as logdet
from scipy.sparse import bmat, csr_matrix, diags, eye
from scipy.sparse.linalg import inv, norm
from tqdm import tqdm

# from tqdm import tqdm

LowRank = namedtuple("LowRank", ["Loading", "Lambda", "D", "Approximation"])
LowRankDiag = namedtuple("LowRankCovariance", ["F", "d"])
LowRankDiagInverse = namedtuple("LowRankCovarianceInverse", ["G", "e"])


def _from_cov_to_precision(LowRankDiag):
    """
    param LowRankDiag: namedtuple with fields F and d

    returns: LowRankDiagInverse namedtuple with fields G and e
    """
    _, k = LowRankDiag.F.shape
    asset_names = LowRankDiag.F.index
    factor_names = LowRankDiag.F.columns

    d = LowRankDiag.d.values
    F = LowRankDiag.F.values

    e = 1 / d

    # construct G
    d = d.reshape(-1, 1)
    d_inv = 1 / d
    matrix = np.linalg.inv(np.eye(k) + (F.T * d_inv.T) @ F)
    L = np.linalg.cholesky(matrix)
    G = d_inv * (F @ L)

    return LowRankDiagInverse(
        G=pd.DataFrame(G, index=asset_names, columns=factor_names),
        e=pd.Series(e, index=asset_names),
    )


def from_cov_to_precisions(LowRankDiags):
    """
    param LowRankDiags: dictionary of LowRankDiag namedtuples

    returns: dictionary of LowRankDiagInverse namedtuples
    """

    for time, LowRankDiag in LowRankDiags.items():
        yield time, _from_cov_to_precision(LowRankDiag)


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

    # lamda, Q = np.linalg.eigh(R) XXX actually faster to use eigh in numpy...

    # Sort eigenvalues in descending order and get r largest
    lamda = lamda[::-1]
    # lamda = lamda[:r]

    # reshuffle the eigenvectors accordingly
    Q = Q[:, ::-1]
    # Q = Q[:, :r]

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
    d = np.diag(Cxx) - 2 * np.sum(Cxs * F, axis=1) + np.sum(F * (F @ Css), axis=1)
    return LowRankDiag(F=F, d=pd.Series(d, index=F.index))


def _em_low_rank_approximation(sigma, initial_sigma, history=False):
    """
    param sigma: one covariance matrices
    param rank: float, rank of low rank component

    returns: regularized covariance matrices
    """
    assets = sigma.columns
    F = initial_sigma.F
    d = initial_sigma.d

    n_iters = 5

    KL_divergence = np.zeros(n_iters)

    for i in range(n_iters):
        Cxx, Cxs, Css = _e_step(sigma, F, d)
        F, d = _m_step(Cxx, Cxs, Css)
        if history:
            Sigma_hat = F @ F.T + np.diag(d)
            KL_divergence[i] = KL_div(Sigma_hat.values, sigma.values)

    if history:
        return LowRankDiag(F=F, d=d), KL_divergence
    else:
        return LowRankDiag(
            F=pd.DataFrame(F, index=assets), d=pd.Series(d, index=assets)
        )


def em_regularize_covariance(sigmas, initial_sigmas, history=False):
    """
    param sigmas: dictionary of covariance matrices
    param  initial_sigmas: dictionary of initial low rank + diagonal approximations; these
    are namedtuples with fields F and d, with F nxk and d length n

    returns: regularized covariance matrices
    """

    for time, sigma in sigmas.items():
        yield time, _em_low_rank_approximation(
            sigma, initial_sigmas[time], history=history
        )


#########
# ADMM implementation of the low rank + diagonal approximation CCP
# solution
#########


def KL_div(Sigma_hat, Sigma):
    """
    Compute KL divergence between two Gaussian distributions
    """
    n = Sigma.shape[0]
    return 0.5 * (
        logdet(Sigma_hat)[1]
        - logdet(Sigma)[1]
        - n
        + np.trace(np.linalg.inv(Sigma_hat) @ Sigma)
    )


def ccp_regularize_covariance(
    sigmas,
    initial_thetas,
    rho=1,
    max_iter=1000,
    eps_abs=1e-6,
    eps_rel=1e-4,
    alpha=1,
    max_iter_ccp=10,
    history=False,
):
    """
    param sigmas: dictionary of covariance matrices
    param  initial_thetas: dictionary of initial low rank + diagonal
    approximations of precision matrices; these
    are namedtuples with fields G and e, with G nxk and e length n; Theta =
    diag(e) - G G^T
    param rho: ADMM step size
    param max_iter: maximum number of ADMM iterations
    param eps_abs: absolute tolerance
    param eps_rel: relative tolerance
    param alpha: over-relaxation parameter
    param max_ccp_iter: maximum number of CCP iterations


    returns: regularized covariance matrices
    """

    for time, sigma in tqdm(sigmas.items()):
        # for time, sigma in sigmas.items():

        G = initial_thetas[time].G.values
        e = initial_thetas[time].e.values

        e, G, KL_divergence, history_last = CCP(
            sigma.values,
            e,
            G,
            rho,
            max_iter,
            eps_abs,
            eps_rel,
            alpha,
            max_iter_ccp,
            history=history,
        )

        if history:
            yield time, LowRankDiagInverse(
                G=pd.DataFrame(G, index=sigma.columns),
                e=pd.Series(e, index=sigma.columns),
            ), KL_divergence, history_last
        else:
            yield time, LowRankDiagInverse(
                G=pd.DataFrame(G, index=sigma.columns),
                e=pd.Series(e, index=sigma.columns),
            )

        # yield time, CCP(sigma.values, e, G, rho, max_iter, eps_abs, eps_rel, alpha, max_iter_ccp)


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
    history=False,
):
    state = _State(
        Sigma, e_init, G_init, rho, max_iter, eps_abs, eps_rel, alpha, history
    )

    if history:
        KL_divergence = np.zeros(max_ccp_iter)

    U = csr_matrix((state.n + state.k, state.n + state.k))
    for i in range(max_ccp_iter):
        if i == 0:
            _max_iter = max_iter * 1
        else:
            _max_iter = int(max(max_iter * 0.75**i, 10))

        e, G, history_last, U = state.update(U, _max_iter)

        if history:  ### XXX make sparse?
            Sigma_hat = np.linalg.inv(np.diag(e) - G @ G.T)
            KL_divergence[i] = KL_div(Sigma_hat, Sigma)

    if history:
        return e, G, KL_divergence, history_last  # XXX only final history
    else:
        return e, G, None, None


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
        history=False,
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
        param history: whether to store the history of the optimization
        """
        self.Sigma = Sigma
        self.e = e_init
        self.G = G_init
        self.rho = rho
        self.max_iter = max_iter
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.alpha = alpha
        self.history = history

    @property
    def n(self):
        return self.e.shape[0]

    @property
    def k(self):
        return self.G.shape[1]

    @property
    def X(self):
        return self._construct_X(self.e, self.G)

    def update(self, Uinit=None, max_iter=None):
        e, G, history, U = self._ADMM_ccp_iteration(Uinit, max_iter=max_iter)
        self.e = e
        self.G = G

        return e, G, history, U

    def _ADMM_ccp_iteration(self, Uinit=None, max_iter=None):
        # Sigma, ek, Gk, rho=1, max_iter=1000, eps_abs=1e-6, eps_rel=1e-4, alpha=1):
        """
        Solves problem
        minimize -log det(X) + trace(Sigma*E) - 2*(Gk)^T * Sigma * G
        subject to X = [E G; G^T I]

        using ADMM
        """
        # n, k = Gk.shape # n number of features, k number of factors

        # Initialize variables

        Sigma = self.Sigma
        rho = self.rho
        n, k = self.n, self.k
        _, Gk = self.e, self.G
        eps_abs, eps_rel = self.eps_abs, self.eps_rel

        Xinit = self._construct_X_sparse(self.e, self.G)

        X = Xinit
        Z = Xinit
        Uinit = csr_matrix((n + k, n + k)) if Uinit is None else Uinit
        U = Uinit

        primal_res = np.zeros(self.max_iter)
        dual_res = np.zeros(self.max_iter)

        primal_eps = np.zeros(self.max_iter)
        dual_eps = np.zeros(self.max_iter)

        # History
        if self.history:
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
        # for i in trange(self.max_iter):

        max_iter = max_iter or self.max_iter

        for i in range(max_iter):
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
            primal_res[i] = norm(X - Z)
            dual_res[i] = rho * norm(Z - Z_old)

            # Compute epsilons
            primal_eps[i] = eps_abs * np.sqrt(n * k) + eps_rel * max(norm(X), norm(Z))
            dual_eps[i] = eps_abs * np.sqrt(n * k) + eps_rel * norm(rho * U)

            if self.history:
                # Compute objective value
                obj_vals_ests[i] = self.objective_value_estimate(X.toarray(), Sigma, Gk)
                obj_vals_true[i] = self.objective_value_true(X.toarray(), Sigma, Gk)

                E = Z[:n, :n]
                G = Z[:n, n:]

                Sigma_hat = inv(E - G @ G.T)

                KL_divergence[i] = KL_div(Sigma_hat.toarray(), Sigma)

            # if self.history: ### XXX
            # Check convergence
            if primal_res[i] < primal_eps[i] and dual_res[i] < dual_eps[i]:
                break

        e = X.diagonal()[:n]
        G = X[:n, n:].toarray()

        if self.history:
            history = pd.DataFrame(history).iloc[: i + 1]

            return e, G, history, U
        else:
            return e, G, None, U

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
    def _construct_X_sparse(e, G):
        """
        Construct X matrix from e and G

        Parameters:
            e (array-like): The diagonal elements of the first block diagonal matrix.
            G (csr_matrix): The matrix G used in the second block and its transpose in the third.

        Returns:
            csr_matrix: Sparse matrix X = [diag(e), G; G^T, I]
        """
        k = G.shape[1]

        E = diags(e)
        if not isinstance(G, csr_matrix):
            G = csr_matrix(G)

        identity = eye(k)

        X = bmat([[E, G], [G.T, identity]], format="csr")

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
        # to sparse
        V = csr_matrix(V)
        W = csr_matrix(W)

        RHS = rho * (Z - U) + W - V

        eigvals, eigvecs = np.linalg.eigh(RHS.toarray())

        Xtilde_diag = ((eigvals + np.sqrt(eigvals**2 + 4 * rho)) / (2 * rho)).reshape(
            -1, 1
        )
        X = (eigvecs * Xtilde_diag.T) @ eigvecs.T

        return csr_matrix(X)

    @staticmethod
    def update_Z(X, U, k):
        """
        Z-update ADMM step
        """
        n = X.shape[0] - k
        matrix = X + U

        E = np.diag(np.diag(matrix.toarray()[:n, :n]))
        Identity = np.eye(k)

        # Project onto the feasible set
        matrix[:n, :n] = E
        matrix[n:, n:] = Identity

        return csr_matrix(matrix)


# import cvxpy as cp


# def cvxpy_ccp_iteration(Sigma, Gk, solver):
#     """
#     Solves problem
#     minimize -log det(X) + trace(Sigma*E) - 2*(Gk)^T * Sigma * G
#     subject to X = [E G; G^T I]

#     using cvxpy
#     """
#     n, k = Gk.shape  # n number of features, k number of factors

#     # Define variables
#     G = cp.Variable((n, k))
#     e = cp.Variable(n)
#     E = cp.diag(e)

#     X = cp.vstack([cp.hstack([E, G]), cp.hstack([G.T, np.eye(k)])])

#     # Define objective
#     obj = -cp.log_det(X) + cp.trace(Sigma @ E) - 2 * cp.trace(Gk.T @ Sigma @ G)

#     # Define problem
#     prob = cp.Problem(cp.Minimize(obj))

#     # Solve problem
#     prob.solve(verbose=False, solver=solver)

#     E, G = X[:n, :n].value, X[:n, n:].value

#     return E, G, prob.value
