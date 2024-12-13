from __future__ import annotations

import warnings
from collections import namedtuple

import cvxpy as cvx
import numpy as np
import pandas as pd

from .ewma import iterated_ewma

# from tqdm import trange

# Mute specific warning
warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")


def _map_nested_dicts(ob, func):
    """
    Recursively applies a function to a nested dictionary
    """
    if isinstance(ob, dict):
        return {k: _map_nested_dicts(v, func) for k, v in ob.items()}

    return func(ob)


def _cholesky_precision(cov):
    """
    Computes Cholesky factor of the inverse of each covariance matrix in cov

    param cov: dictionary of covariance matrices {time: Sigma}
    """
    return {
        time: np.linalg.cholesky(np.linalg.inv(item.values))
        for time, item in cov.items()
    }


def _B_t_col(Ls, nus, returns):
    """
    Computes L.T @ return for each L factor in Ls and corresponding
    return in returns
    """
    return {time: L.T @ returns.loc[time].values - nus[time] for time, L in Ls.items()}


def _diag_part(cholesky):
    return {key: np.diag(L) for key, L in cholesky.items()}


def _A(diags_interval, K):
    """
    param diags_interval: NxK pandas DataFrame where each entry is diagonal
    of an expert Cholesky factor matrix
    param K: number of expert predictors

    returns: nNxK matrix where each column is a vector of stacked diagonals
    """
    return np.column_stack(
        [np.vstack(diags_interval.iloc[:, i]).flatten() for i in range(K)]
    )


def _nu(Ls, means):
    """
    Computes L.T @ mu for each L factor in Ls and corresponding
    mu in means
    """
    return {time: L.T @ means[time] for time, L in Ls.items()}


# Declaring namedtuple()
Result = namedtuple("Result", ["time", "mean", "covariance", "weights"])


class _CombinationProblem:
    def __init__(self, keys, n, window, smoother, gamma):
        self.keys = keys
        K = len(keys)
        self._weight = cvx.Variable(len(keys), name="weights")
        self.A_param = cvx.Parameter((n * window, K))
        self.P_chol_param = cvx.Parameter((K, K))

        if smoother:
            self._weight_prev = cvx.Parameter(len(keys), name="weights_prev")
            if smoother == "l2":
                self.smooth_penalty = cvx.norm(self._weight - self._weight_prev, 2)
            elif smoother == "l1":
                self.smooth_penalty = cvx.norm(self._weight - self._weight_prev, 1)
            elif smoother == "sum_squares":
                self.smooth_penalty = cvx.sum_squares(self._weight - self._weight_prev)
            else:
                raise ValueError("smoother must be None, 'l2', 'l1' or 'sum_squares'")

        self.smoother = smoother
        self.gamma = gamma

    @property
    def _constraints(self):
        return [cvx.sum(self._weight) == 1, self._weight >= 0]

    @property
    def _objective(self):
        if self.smoother:
            return (
                cvx.sum(cvx.log(self.A_param @ self._weight))
                - 0.5 * cvx.sum_squares(self.P_chol_param.T @ self._weight)
                - self.gamma * self.smooth_penalty
            )
        else:
            return cvx.sum(
                cvx.log(self.A_param @ self._weight)
            ) - 0.5 * cvx.sum_squares(self.P_chol_param.T @ self._weight)

    def _construct_problem(self):
        self.prob = cvx.Problem(cvx.Maximize(self._objective), self._constraints)

    def solve(self, **kwargs):
        return self.prob.solve(**kwargs)

    @property
    def weights(self):
        return pd.Series(index=self.keys, data=self._weight.value)

    @property
    def status(self):
        return self.prob.status


def from_ewmas(
    returns, pairs, min_periods_vola=20, min_periods_cov=20, clip_at=None, mean=False
):
    """
    Estimate a series of covariance matrices using the iterated EWMA method

    param returns: Frame of returns
    param pairs: list of pairs of EWMA half lives, e.g. [(20, 20), (10, 50), (20, 60)],
                pair[0] is the half life for volatility estimation
                pair[1] is the half life for covariance estimation
    param min_periods_vola: minimum number of observations to start EWMA for volatility estimation (optional)
    param min_periods_cov: minimum number of observations to start EWMA for covariance estimation (optional)
    param clip_at: clip volatility adjusted returns at +- clip_at (optional)
    param mean: subtract EWMA mean from returns and volatility adjusted returns (optional)

    return: Yields tuples with time, mean, covariance matrix, weights
    """
    # compute the covariance matrices, one time series for each pair
    results = {
        f"{pair[0]}-{pair[1]}": {
            result.time: result
            for result in iterated_ewma(
                returns=returns,
                vola_halflife=pair[0],
                cov_halflife=pair[1],
                min_periods_vola=min_periods_vola,
                min_periods_cov=min_periods_cov,
                clip_at=clip_at,
                mean=mean,
            )
        }
        for pair in pairs
    }

    sigmas = _map_nested_dicts(results, lambda result: result.covariance)
    means = _map_nested_dicts(results, lambda result: result.mean)

    # combination of covariance matrix valued time series
    return _CovarianceCombination(sigmas=sigmas, returns=returns, means=means)


def from_sigmas(sigmas, returns, means=None):
    return _CovarianceCombination(sigmas=sigmas, returns=returns, means=means)


class _CovarianceCombination:
    def __init__(self, sigmas, returns, means=None):
        """
        Computes the covariance combination of a set of covariance matrices

        param sigmas: dictionary of covariance matrices {key: {time: sigma}}
        param returns: pandas DataFrame of returns
        param means: dictionary of means {key: {time: mu}}, optional

        Note: sigmas[key][time] is a covariance matrix prediction for time+1
        """
        # Assert Sigmas and means have same keys if means not None
        n = returns.shape[1]
        if means is not None:
            for key, sigma in sigmas.items():
                # Assert sigmas and means have same keys
                assert (
                    sigma.keys() == means[key].keys()
                ), "sigmas and means must have same keys"
        else:
            # Set means to zero if not provided
            means = {
                k: {time: np.zeros(n) for time in sigma.keys()}
                for k, sigma in sigmas.items()
            }

        self.__means = means
        self.__sigmas = sigmas
        self.__returns = returns

        # all those quantities don't depend on the window size
        self.__Ls = pd.DataFrame(
            {k: _cholesky_precision(sigma) for k, sigma in self.sigmas.items()}
        )
        self.__Ls_shifted = self.__Ls.shift(1).dropna()
        self.__nus = pd.DataFrame(
            {key: _nu(Ls, self.means[key]) for key, Ls in self.__Ls.items()}
        )
        self.__nus_shifted = self.__nus.shift(1).dropna()

    @property
    def sigmas(self):
        return self.__sigmas

    @property
    def means(self):
        return self.__means

    @property
    def returns(self):
        return self.__returns

    @property
    def K(self):
        """
        Returns the number of expert predictors
        """
        return len(self.sigmas)

    @property
    def assets(self):
        """
        Returns the assets in the covariance combination problem
        """
        return self.returns.columns

    def solve(
        self,
        window=None,
        times=None,
        smoother=None,
        gamma=None,
        weight_init=None,
        **kwargs,
    ):
        """
        The size of the window is crucial to specify the size of the parameters
        for the cvxpy problem. Hence those computations are not in the __init__ method

        Solves the covariance combination problem at a time steps given in times

        param window: number of previous time steps to use in the covariance
        combination problem
        param times: list of time steps to solve the problem at; if None, solve
        at all available time steps
        param smoother: smoothing parameter for the covariance combination
        problem; None, 'l2', 'l1', or 'sum_squares'. 'l2' yields piecewise
        constant weights; 'l1' yields sparese weight updates; 'sum_squares'
        yields smooth weight updates
        param gamma: regularization parameter for the covariance combination;
        only applicable if smoother is not None; penalty becomes gamma *
        ||w||_2,  gamma * ||w||_1, or gamma * ||w||_2^2 depending on the value
        of the 'smoother' parameter
        param weight_init: initial weights for the covariance combination
        problem; if None, use the default uniform initialization; only
        applicable if smoother is not None
        """
        # If window is None, use all available data; cap window at length of data
        window = window or len(self.__Ls_shifted)
        window = min(window, len(self.__Ls_shifted))

        # Compute P matrix and its Cholesky factor
        Lts_at_r = pd.DataFrame(
            {
                key: _B_t_col(Ls, self.__nus_shifted[key], self.returns)
                for key, Ls in self.__Ls_shifted.items()
            }
        )

        # time steps to solve the problem at
        all_available_times = Lts_at_r.index

        if times is None:
            times = Lts_at_r.index
        else:
            times = list(times)

            # assert times are consecutive in all_available_times
            zero_index = all_available_times.get_loc(times[0])
            assert list(
                all_available_times[zero_index : zero_index + len(times)]
            ) == list(times)
            "times must be consecutive"

            # add window - 1 previous time steps to times
            times = (
                list(all_available_times[zero_index - window + 1 : zero_index]) + times
            )

        Bs = {time: np.column_stack(Lts_at_r.loc[time]) for time in times}
        prod_Bs = pd.Series({time: B.T @ B for time, B in Bs.items()})
        # times = prod_Bs.index
        P = {
            times[i]: sum(prod_Bs.loc[times[i - window + 1] : times[i]])
            for i in range(window - 1, len(times))
        }

        P_chol = {time: np.linalg.cholesky(matrix) for time, matrix in P.items()}

        # Compute A matrix
        Ls_diag = pd.DataFrame({k: _diag_part(L) for k, L in self.__Ls_shifted.items()})

        A = {
            times[i]: _A(
                Ls_diag.truncate(before=times[i - window + 1], after=times[i]), self.K
            )
            for i in range(window - 1, len(times))
        }

        problem = _CombinationProblem(
            keys=self.sigmas.keys(),
            n=len(self.assets),
            window=window,
            smoother=smoother,
            gamma=gamma,
        )

        problem._construct_problem()
        if smoother and not weight_init:
            problem._weight_prev.value = np.ones(self.K) / self.K

        for time, AA in A.items():
            problem.A_param.value = AA
            problem.P_chol_param.value = P_chol[time]

            try:
                yield self._solve(time=time, problem=problem, **kwargs)
                if smoother:
                    problem._weight_prev.value = problem._weight.value
            except cvx.SolverError:
                print(f"Solver did not converge at time {time}")
                yield None

    def _solve(self, time, problem, **kwargs):
        """
        Solves the covariance combination problem at a given time t
        """
        problem.solve(**kwargs)

        weights = problem.weights

        # Get non-shifted L
        L = sum(self.__Ls.loc[time] * weights.values)  # prediction for time+1
        nu = sum(self.__nus.loc[time] * weights.values)  # prediction for time+1

        mean = pd.Series(index=self.assets, data=np.linalg.inv(L.T) @ nu)
        sigma = pd.DataFrame(
            index=self.assets, columns=self.assets, data=np.linalg.inv(L @ L.T)
        )

        mean = pd.Series(index=self.assets, data=np.linalg.inv(L.T) @ nu)
        sigma = pd.DataFrame(
            index=self.assets, columns=self.assets, data=np.linalg.inv(L @ L.T)
        )

        return Result(time=time, mean=mean, covariance=sigma, weights=weights)
