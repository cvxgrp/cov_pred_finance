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
Result = namedtuple('Result', ['time', 'mean', 'covariance', 'weights'])


class _CombinationProblem:
    def __init__(self, keys, n, window):
        self.keys = keys
        K = len(keys)
        self._weight = cvx.Variable(len(keys), name="weights")
        self.A_param = cvx.Parameter((n * window, K))
        self.P_chol_param = cvx.Parameter((K, K))

    @property
    def _constraints(self):
        return [cvx.sum(self._weight) == 1, self._weight >= 0]

    @property
    def _objective(self):
        return cvx.sum(cvx.log(self.A_param @ self._weight)) - 0.5 * cvx.sum_squares(self.P_chol_param.T @ self._weight)

    @property
    def _problem(self):
        return cvx.Problem(cvx.Maximize(self._objective), self._constraints)

    def solve(self, **kwargs):
        self._problem.solve(**kwargs)
        return self.weights

    @property
    def weights(self):
        return pd.Series(index=self.keys, data=self._weight.value)


class CovarianceCombination:
    def __init__(self, sigmas, returns, means=None):
        """
        Computes the covariance combination of a set of covariance matrices

        param sigmas: dictionary of covariance matrices {key: {time: sigma}}
        param returns: pandas DataFrame of returns
        param means: dictionary of means {key: {time: mu}}, optional
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

        # all those quantities don't depend on the window size
        self.__Ls = pd.DataFrame({k: _cholesky_precision(sigma) for k, sigma in sigmas.items()})
        self.__Ls_shifted = self.__Ls.shift(1).dropna()
        self.__nus = pd.DataFrame({key: _nu(Ls, means[key]) for key, Ls in self.__Ls.items()})
        self.__nus_shifted = self.__nus.shift(1).dropna()

    @property
    def K(self):
        """
        Returns the number of expert predictors
        """
        return len(self.__sigmas)

    @property
    def assets(self):
        """
        Returns the assets in the covariance combination problem
        """
        return self.__returns.columns

    def solve(self, window=None, **kwargs):
        """
        The size of the window is crucial to specify the size of the parameters
        for the cvxpy problem. Hence those computations are not in the __init__ method

        Solves the covariance combination problem at a given time, i.e.,
        finds the prediction for the covariance matrix at 'time+1'

        param window: number of previous time steps to use in the covariance combination
        """
        # If window is None, use all available data; cap window at length of data
        window = window or len(self.__Ls_shifted)
        window = min(window, len(self.__Ls_shifted))

        # Compute P matrix and its Cholesky factor
        Lts_at_r = pd.DataFrame({key: _B_t_col(Ls, self.__nus_shifted[key], self.__returns) for key, Ls in self.__Ls_shifted.items()})

        Bs = {time: np.column_stack(Lts_at_r.loc[time]) for time in Lts_at_r.index}
        prod_Bs = pd.Series({time: B.T @ B for time, B in Bs.items()})
        times = prod_Bs.index
        P = {times[i]: sum(prod_Bs.loc[times[i - window + 1]:times[i]]) for i in range(window - 1, len(times))}

        P_chol = {time: np.linalg.cholesky(matrix) for time, matrix in P.items()}

        # Compute A matrix
        Ls_diag = pd.DataFrame({k: _diag_part(L) for k, L in self.__Ls_shifted.items()})

        A = {times[i]: _A(Ls_diag.truncate(before=times[i - window + 1], after=times[i]), self.K) for i in
            range(window - 1, len(times))}

        problem = _CombinationProblem(keys=self.__sigmas.keys(), n=len(self.assets), window=window)

        for time in A.keys():
            problem.A_param.value = A[time]
            problem.P_chol_param.value = P_chol[time]

            yield self._solve(time=time, problem=problem, **kwargs)


    def _solve(self, time, problem, **kwargs):
        """
        Solves the covariance combination problem at a given time t
        """
        # solve problem
        weights = problem.solve(**kwargs)

        # Get non-shifted L
        L = sum(self.__Ls.loc[time] * weights.values)    # prediction for time+1
        nu = sum(self.__nus.loc[time] * weights.values)  # prediction for time+1

        mean = pd.Series(index=self.assets, data=np.linalg.inv(L.T) @ nu)
        sigma = pd.DataFrame(index=self.assets, columns=self.assets, data=np.linalg.inv(L @ L.T))
        return Result(time=time, mean=mean, covariance=sigma, weights=weights)
