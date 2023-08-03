# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np

slogdet = np.linalg.slogdet


class _State:
    def __init__(self, Sigma, F, d):
        self.rank = F.shape[1]

        # Construct E, G, s.t, E-GG^T = (D+FF^T)^{-1}
        d_inv = 1 / d.reshape(-1, 1)
        Lamda = np.linalg.cholesky(np.linalg.inv(np.eye(self.rank) + F.T @ (d_inv * F)))

        self.Gk = d_inv * (F @ Lamda)  # k=ccp iteration
        self.ek = d_inv.flatten()

        self.Sigma = Sigma
        self.n = Sigma.shape[0]

    def objective(self):
        I = np.eye(self.rank)

        matrix = np.vstack(
            [np.hstack([np.diag(self.ek), self.Gk]), np.hstack([self.Gk.T, I])]
        )

        obj = (
            -slogdet(matrix)[1]
            + np.trace(self.Sigma @ np.diag(self.ek))
            - np.linalg.norm(np.linalg.cholesky(self.Sigma).T @ self.Gk, ord="fro") ** 2
        )

        return obj

    def construct_problem(self):
        """
        Construct the ccp problem
        """
        self.e_var = cp.Variable((self.n, 1))
        self.G_var = cp.Variable((self.n, self.rank))
        self.Gk_par = cp.Parameter((self.n, self.rank))

        I = np.eye(self.rank)

        matrix = cp.vstack(
            [cp.hstack([cp.diag(self.e_var), self.G_var]), cp.hstack([self.G_var.T, I])]
        )

        obj = (
            -cp.log_det(matrix)
            + cp.trace(cp.multiply(self.Sigma, self.e_var.T))
            - 2 * cp.trace(self.Sigma @ self.Gk_par @ self.G_var.T)
        )

        self.prob = cp.Problem(cp.Minimize(obj))

    def iterate(self):
        """
        One iteration of the algorithm
        """

        self.Gk_par.value = self.Gk.copy()
        self.prob.solve(solver="MOSEK")
        print(self.objective())
        self.ek = self.e_var.value.flatten()
        self.Gk = self.G_var.value

    def build(self):
        """
        Return final F and d
        """
        e_inv = 1 / self.ek.reshape(-1, 1)
        Lamda = np.linalg.cholesky(
            np.linalg.inv((np.eye(self.rank) - self.Gk.T @ (e_inv * self.Gk)))
        )
        F = e_inv * (self.Gk @ Lamda)
        d = e_inv.flatten()

        return F, d


class _State2:
    def __init__(self, Sigma, F, d):
        self.rank = F.shape[1]

        # Construct E, G, s.t, E-GG^T = (D+FF^T)^{-1}
        d_inv = 1 / d.reshape(-1, 1)
        Lamda = np.linalg.cholesky(np.linalg.inv(np.eye(self.rank) + F.T @ (d_inv * F)))

        self.Gk = d_inv * (F @ Lamda)  # k=ccp iteration
        self.ek = d_inv.flatten()

        self.Sigma = Sigma
        self.n = Sigma.shape[0]

    def objective(self):
        I = np.eye(self.rank)

        matrix = np.vstack(
            [np.hstack([np.diag(self.ek), self.Gk]), np.hstack([self.Gk.T, I])]
        )

        print(slogdet(matrix)[0])
        print(slogdet(matrix)[1])
        obj = (
            -slogdet(matrix)[1]
            + np.trace(self.Sigma @ np.diag(self.ek))
            - np.linalg.norm(np.linalg.cholesky(self.Sigma).T @ self.Gk, ord="fro") ** 2
        )

        return obj

    def iterate(self):
        """
        One iteration of the algorithm
        """

        self.ek, self.Gk = _solve_proximal_gradient(
            self.ek, self.Gk, self.Gk, self.Sigma
        )
        # print(self.Gk)
        print(self.objective())

    def build(self):
        """
        Return final F and d
        """
        e_inv = 1 / self.ek.reshape(-1, 1)
        Lamda = np.linalg.cholesky(
            np.linalg.inv((np.eye(self.rank) - self.Gk.T @ (e_inv * self.Gk)))
        )
        F = e_inv * (self.Gk @ Lamda)
        d = e_inv.flatten()

        return F, d


def _common_factor(e, G):
    """
    The common factor in the gradient
    """
    e_inv = 1 / e.reshape(-1, 1)
    I = np.eye(G.shape[1])

    return e_inv * (G @ np.linalg.inv(I - G.T @ (e_inv * G)))


def _grad_G(common_factor):
    return 2 * common_factor


def _grad_e(e, G, common_factor):
    e_inv = (1 / e).reshape(-1, 1)

    return -(e_inv.flatten() + np.diag(common_factor @ (G.T * e_inv.T)))


def _solve_proximal_gradient(e_init, G_init, Gk, Sigma, maxiters=100):
    ej = e_init.copy()
    Gj = G_init.copy()

    Sigma_at_Gk = Sigma @ Gk  # For proximal gradient update
    for j in range(maxiters):
        common_factor = _common_factor(ej, Gj)
        G_grad = _grad_G(common_factor)
        e_grad = _grad_e(ej, Gj, common_factor)

        # TODO: hardcoded
        alpha = 1
        ej = (ej - (e_grad + np.diag(Sigma))) * alpha
        Gj = (Gj - (G_grad - 2 * Sigma_at_Gk)) * alpha

    return ej, Gj
