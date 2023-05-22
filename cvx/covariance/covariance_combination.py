from collections import namedtuple
import cvxpy as cvx
import numpy as np
import pandas as pd
from tqdm import tqdm
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

def _zero_means(times, n):
    return  {time: np.zeros(n) for time in times}

def _nu(Ls, means):
    """
    Computes L.T @ mu for each L factor in Ls and corresponding
    mu in means
    """
    return {time:  L.T @ means[time] for time, L in Ls.items()}  

# Declaring namedtuple()
Result = namedtuple('Result', ['time', 'window', 'mean', 'covariance', 'weights'])


class CovarianceCombination:
    def __init__(self, Sigmas, returns, means=None):

        ### Assert Sigmas and means have same keys if means not None
        n = returns.shape[1]
        if means is not None:
            for key, Sigma in Sigmas.items():
                assert Sigma.keys() == means[key].keys(), "Sigmas and means must have same keys"
        else:
            means = {k: _zero_means(Sigma.keys(), n) for k, Sigma in Sigmas.items()}
    
        # ### Fix conditioning
        # # Multiply entries in means and returns by 100
        # means = {k: {time: 100*item for time, item in means[k].items()} for k in means.keys()}
        # returns = 100*returns

        # # Multiply entries in Sigmas by 10000
        # Sigmas = {k: {time: 10000*item for time, item in Sigma.items()} for k, Sigma in Sigmas.items()}
                
        self.__means = means
        self.__Sigmas = Sigmas
        self.__returns = returns
        self.__K = len(self.__Sigmas)
        self.__weight = cvx.Variable(self.__K, name="weights")
        self.__n = n
        self._initilized = False

    @property
    def K(self):
        return self.__K

    @property
    def index(self):
        return self.__A.index

    @property
    def assets(self):
        return self.__returns.columns

    def _combine_experts(self, experts):
        return np.sum([expert * w_k for expert, w_k in zip(experts, self.__weight)])

    def _precompute(self, window=10):
        """
        Precomputes parameter values for CVXPY problem
        """
        returns = self.__returns
        Sigmas = self.__Sigmas
        means = self.__means

        self.Ls = pd.DataFrame({k: _cholesky_precision(Sigma) for k, Sigma in Sigmas.items()})
        self.Ls_shifted = self.Ls.shift(1).dropna()
        self.nus = pd.DataFrame({key: _nu(Ls, means[key]) for key, Ls in self.Ls.items()})
        nus_shifted = self.nus.shift(1).dropna()
        
        # If window is None, use all available data; cap window at length of data
        window = window or len(self.Ls_shifted)
        window = min(window, len(self.Ls_shifted))

        ### Compute the parameter values for CVXPY problem
        # Compute P matrix and its Cholesky factor
        Lts_at_r = pd.DataFrame({key: _B_t_col(Ls, nus_shifted[key], returns) for key, Ls in self.Ls_shifted.items()}) 
        Bs = {time: np.column_stack(Lts_at_r.loc[time]) for time in Lts_at_r.index}
        prod_Bs = pd.Series({time: B.T@B for time, B in Bs.items()})
        times = prod_Bs.index
        P = pd.Series({times[i]: sum(prod_Bs.loc[times[i-window+1]:times[i]]) for i in range(window-1, len(times))})
        self.__P_chol = pd.Series({time: np.linalg.cholesky(P.loc[time]) for time in P.index})

        # Compute A matrix
        Ls_diag = pd.DataFrame({k: _diag_part(L) for k, L in self.Ls_shifted.items()})
        self.__A = pd.Series({times[i]: _A(Ls_diag.truncate(before=times[i-window+1], after=times[i]), self.K) for i in range(window-1, len(times))})

        ### Generate CVPXY problem
        self.__A_param = cvx.Parameter((self.__n*window, self.__K))
        self.__P_chol_param = cvx.Parameter((self.__K, self.__K))
        self.__obj = cvx.sum(cvx.log(self.__A_param@self.__weight)) - 0.5*cvx.sum_squares(self.__P_chol_param.T@self.__weight)
        self.__constraints = [self.__weight >= 0, cvx.sum(self.__weight) == 1]
        self.__prob = cvx.Problem(cvx.Maximize(self.__obj), self.__constraints)

    def solve(self, time, window=10, **kwargs):
        """
        Solves the covariance combination problem at a given time, i.e.,
        finds the prediction for the covariance matrix at 'time+1'
        """
        

        # Precompute parameter values for CVXPY problem (only do once)
        if not self._initilized:
            self._precompute(window)
            self._initilized = True

        # Update parameters and solve problem
        self.__A_param.value = self.__A.loc[time]
        self.__P_chol_param.value = self.__P_chol.loc[time]
        self.__prob.solve(**kwargs)

        # Get non-shifted L
        L = sum(self.Ls.loc[time]*self.__weight.value) # prediction for time+1
        nu = sum(self.nus.loc[time]*self.__weight.value) # prediction for time+1

        ### Conditioning
        mean = np.linalg.inv(L.T)@nu # / 100
        mean = pd.Series(index=self.assets, data=mean)
        Sigma = np.linalg.inv(L @ L.T) # / 10000
        Sigma = pd.DataFrame(index=self.assets, columns=self.assets, data=Sigma)
        weights = pd.Series(index=self.__Sigmas.keys(), data=self.__weight.value)
        return Result(time=time, window=window, mean=mean, covariance=Sigma, weights=weights)

    def solve_window(self, window=10, verbose=True, **kwargs):
        """
        Solves the covariance combination problem for all timesteps
        """

        self._precompute(window)
        self._initilized = True

        if verbose:
            for time in tqdm(self.index):
                yield self.solve(time, window=window, **kwargs)
        else:
            for time in self.index:
                yield self.solve(time, window=window, **kwargs)


