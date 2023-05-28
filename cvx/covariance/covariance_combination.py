from collections import namedtuple
import cvxpy as cvx
import numpy as np
import pandas as pd
import warnings

# Mute specific warning
warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")


def _cholesky_precision(cov):
    """
    Computes Cholesky factor of the inverse of each covariance matrix in cov

    param cov: dictionary of covariance matrices {time: Sigma}
    """
    return {time: np.linalg.cholesky(np.linalg.inv(item.values)) for time, item in cov.items()}


def _B_t_col(Ls, nus, returns):
    """
    Computes L.T @ return for each L factor in Ls and corresponding
    return in returns
    """
    return {time:  L.T @ returns.loc[time].values - nus[time] for time, L in Ls.items()}


def _diag_part(cholesky):
    return {key: np.diag(L) for key, L in cholesky.items()}


def _A(diags_interval, K):
    """
    param diags_interval: NxK pandas DataFrame where each entry is diagonal
    of an expert Cholesky factor matrix
    param K: number of expert predictors

    returns: nNxK matrix where each column is a vector of stacked diagonals
    """
    return np.column_stack([np.vstack(diags_interval.iloc[:,i]).flatten() for i in range(K)])


def _nu(Ls, means):
    """
    Computes L.T @ mu for each L factor in Ls and corresponding
    mu in means
    """
    return {time:  L.T @ means[time] for time, L in Ls.items()}  

# Declaring namedtuple()
Result = namedtuple('Result', ['time', 'window', 'mean', 'covariance', 'weights'])


class _CombinationProblem:
    def __init__(self, keys, n, window):
        self.keys = keys
        self.K = len(keys)
        self._weight = cvx.Variable(self.K, name="weights")
        self.A_param = cvx.Parameter((n * window, self.K))
        self.P_chol_param = cvx.Parameter((self.K, self.K))

    @property
    def constraints(self):
        return [cvx.sum(self._weight) == 1, self._weight >= 0]

    @property
    def objective(self):
        return cvx.sum(cvx.log(self.A_param @ self._weight)) - 0.5 * cvx.sum_squares(self.P_chol_param.T @ self._weight)

    @property
    def problem(self):
        return cvx.Problem(cvx.Maximize(self.objective), self.constraints)

    def solve(self, **kwargs):
        self.problem.solve(**kwargs)
        return self.weights

    @property
    def weights(self):
        return pd.Series(index=self.keys, data=self._weight.value)


class CovarianceCombination:
    def __init__(self, sigmas, returns, means=None, window=None):
        """
        Computes the covariance combination of a set of covariance matrices

        param sigmas: dictionary of covariance matrices {key: {time: sigma}}
        param returns: pandas DataFrame of returns
        param means: dictionary of means {key: {time: mu}}, optional
        param window: number of periods to use in covariance estimation, optional
        """
        # Assert Sigmas and means have same keys if means not None
        n = returns.shape[1]
        if means is not None:
            for key, sigma in sigmas.items():
                # Assert sigmas and means have same keys
                assert sigma.keys() == means[key].keys(), "sigmas and means must have same keys"
        else:
            # Set means to zero if not provided
            means = {k: {time: np.zeros(n) for time in sigma.keys()} for k, sigma in sigmas.items()}
                
        self.__means = means
        self.__sigmas = sigmas
        self.__returns = returns

        self.__Ls = pd.DataFrame({k: _cholesky_precision(sigma) for k, sigma in sigmas.items()})
        self.__Ls_shifted = self.__Ls.shift(1).dropna()
        self.__nus = pd.DataFrame({key: _nu(Ls, means[key]) for key, Ls in self.__Ls.items()})
        nus_shifted = self.__nus.shift(1).dropna()

        # If window is None, use all available data; cap window at length of data
        window = window or len(self.__Ls_shifted)
        window = min(window, len(self.__Ls_shifted))
        self.window = window

        # Compute P matrix and its Cholesky factor
        Lts_at_r = pd.DataFrame({key: _B_t_col(Ls, nus_shifted[key], returns) for key, Ls in self.__Ls_shifted.items()})

        Bs = {time: np.column_stack(Lts_at_r.loc[time]) for time in Lts_at_r.index}
        prod_Bs = pd.Series({time: B.T @ B for time, B in Bs.items()})
        times = prod_Bs.index
        P = {times[i]: sum(prod_Bs.loc[times[i - window + 1]:times[i]]) for i in range(window - 1, len(times))}

        self.__P_chol = {time: np.linalg.cholesky(matrix) for time, matrix in P.items()}

        # Compute A matrix
        Ls_diag = pd.DataFrame({k: _diag_part(L) for k, L in self.__Ls_shifted.items()})

        self.__A = {times[i]: _A(Ls_diag.truncate(before=times[i - window + 1], after=times[i]), self.K) for i in
             range(window - 1, len(times))}

        self.__problem = _CombinationProblem(keys=self.__sigmas.keys(), n=len(self.assets), window=window)

    @property
    def K(self):
        """
        Returns the number of expert predictors
        """
        return len(self.__sigmas)

    @property
    def index(self):
        """
        Returns the index of the covariance combination problem, e.g. the timestamps
        """
        return self.__A.keys()

    @property
    def assets(self):
        """
        Returns the assets in the covariance combination problem
        """
        return self.__returns.columns

    def solve(self, **kwargs):
        """
        Solves the covariance combination problem at a given time, i.e.,
        finds the prediction for the covariance matrix at 'time+1'
        """
        for time in self.index:
            yield self._solve(time=time, **kwargs)


    def _solve(self, time, **kwargs):
        # Update parameters and solve problem
        self.__problem.A_param.value = self.__A[time]
        self.__problem.P_chol_param.value = self.__P_chol[time]
        weights = self.__problem.solve(**kwargs)

        # Get non-shifted L
        L = sum(self.__Ls.loc[time] * weights.values)    # prediction for time+1
        nu = sum(self.__nus.loc[time] * weights.values)  # prediction for time+1

        mean = pd.Series(index=self.assets, data=np.linalg.inv(L.T) @ nu)
        sigma = pd.DataFrame(index=self.assets, columns=self.assets, data=np.linalg.inv(L @ L.T))
        return Result(time=time, window=self.window, mean=mean, covariance=sigma, weights=weights)

    #def solve_window(self, **kwargs):
    #    """
    #    Solves the covariance combination problem for all time steps
    #    """
    #    for time in self.index:
    #        yield self.solve(time, **kwargs)
