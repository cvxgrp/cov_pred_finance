import numpy as np
import pandas as pd
import pytest
import cvxpy as cp

from cvx.covariance.ewma import ewma_cov
from cvx.covariance.predictors import AlternatingWeightedPredictor, \
    CovariancePredictor

@pytest.fixture()
def prices(prices):
    return prices[["GOOG", "AAPL", "FB"]].head(100)

@pytest.fixture()
def returns(prices):
    return prices.pct_change().dropna(axis=0, how="all")

def log_likelihood(L, r):
    return cp.sum(cp.log(cp.diag(L))) - 0.5 * cp.sum_squares(L.T @ r)

def cholesky_precision(cov):
    return {key: np.linalg.cholesky(np.linalg.inv(item.values)) for key, item in cov.items()}


def predictors(returns):
    # K=3 ewma predictors (shift(1) to make them causal)
    # returns = returns.fillna(0.0)
    min_periods = 20
    Sigmas = dict()
    for k, cov in enumerate([10, 21, 63]):
        Sigmas[k] = pd.Series(dict(ewma_cov(returns, halflife=cov, min_periods=min_periods))).shift(1).dropna()

    return Sigmas


def get_LLL(Sigmas):

    # Cholesky factors
    L = pd.DataFrame({k: cholesky_precision(Sigma) for k, Sigma in Sigmas.items()})

    return L

def benchmark_weights(N, returns, Sigmas):
    """
    param N: number of days to look back
    param LLL: TxK dataframe of cholesky factors of inverse covariance matrices\
        T=number of days, K=number of predictors
    param returns: Txn dataframe of returns, n=number of assets
    """
    LLL = get_LLL(Sigmas)

    #n = len(LLL)
    #xxx = pd.DataFrame(LLL)
    #print(xxx)
    #assert False

    T_not_na, n = LLL.shape

    w = cp.Variable(n, name="weights")

    L_hat = dict()
    for time, experts in LLL.iterrows():
        L_hat[time] = np.sum([expert * w_k for expert, w_k in zip(experts, w)])

    # don't shift here, as shifting has taken place before for the individual experts
    L_hat = pd.Series(L_hat)

    ws = np.ones_like(LLL)*np.nan

    # get number of nan rows in LLL
    #nans = LLL.shape[0] - LLL_dropna.shape[0]

    for t in range(N, T_not_na):
        obj = cp.sum([log_likelihood(L_hat[time], r=returns.loc[time].values)\
            for time in [*L_hat.index][t-N:t]])

        prob = cp.Problem(cp.Maximize(obj), [cp.sum(w) == 1, w >= 0])
        prob.solve()
        ws[t] = w.value

    return pd.DataFrame(ws, index=LLL.index, columns=[0,1,2])

def my_weights(N, returns, Sigmas):

    #Sigma_10, Sigma_21, Sigma_63 = Sigmas[0], Sigmas[1], Sigmas[2]

    # Weighted iterated EWMA predictor
    const_L = "off"
    ignore_dpp = True

    Sigma_10_new = CovariancePredictor()
    Sigma_10_new.Sigma_hats = {t: mat.values for t, mat in Sigmas[0].items()}


    Sigma_21_new = CovariancePredictor()
    Sigma_21_new.Sigma_hats = {t: mat.values for t, mat in Sigmas[1].items()}

    Sigma_63_new = CovariancePredictor()
    Sigma_63_new.Sigma_hats = {t: mat.values for t, mat in Sigmas[2].items()}


    experts = [Sigma_10_new, Sigma_21_new, Sigma_63_new]
    returns_new = returns.loc[[t for t in Sigma_10_new.Sigma_hats.keys()]]

    cm_iewma = AlternatingWeightedPredictor(None, returns_new, likelihood_memory=N, init_cutoff=0, const_L=const_L,\
        ignore_dpp=ignore_dpp,\
        experts=experts)
    cm_iewma.name = "CM-IEWMA"

    # Just to make dataframes consistent in comparison
    cm_iewma.weights.columns = [0,1,2]
    cm_iewma.weights.index.name = None

    return cm_iewma.weights



def test_cm_predictor(returns):
    N = 10

    Sigmas = predictors(returns)

    # print(Sigmas)

    ws_benchmark = benchmark_weights(N, returns, Sigmas)
    for t, row in ws_benchmark.iterrows():
        # print(t, row)
        pass
    #assert False

    ws_mine = my_weights(N, returns, Sigmas)

    # Solutions are not very precise, hence atol=1e-3
    ws_benchmark_dropna = ws_benchmark

    print("col names", ws_mine.columns)
    print("col names", ws_benchmark_dropna.columns)

    pd.testing.assert_frame_equal(ws_benchmark.dropna(axis=0, how="all"), ws_mine, check_dtype=False,atol=1e-3)


    

   