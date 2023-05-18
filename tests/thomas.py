import cvxpy
import pandas as pd

from cvx.covariance.ewma import iterated_ewma
from cvx.covariance.covariance_combination import CovarianceCombination


def f(returns, pairs, lower=None, upper=None, window=10, **kwargs):
    # compute the covariance matrices, one time series for each pair
    Sigmas = {f"{pair[0]}-{pair[1]}": iterated_ewma(returns, vola_halflife=pair[0], cov_halflife=pair[1], lower=lower, upper=upper) for pair in pairs}

    # combination of covariance matrix valued time series
    combination = CovarianceCombination(sigmas=Sigmas, returns=returns)

    for result in combination.solve_window(window=window, **kwargs):
        yield result

if __name__ == '__main__':
    prices = pd.read_csv("resources/stock_prices.csv", index_col=0, header=0, parse_dates=True).ffill()
    prices = prices[["GOOG", "AAPL", "FB"]].head(100)

    # Frame of returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    # compute the covariance matrices, one time series for each pair
    Sigmas = {pair: iterated_ewma(returns, vola_halflife=pair[0], cov_halflife=pair[1], lower=-4.2, upper=4.2) for pair in pairs}

    # combination of covariance matrix valued time series
    combination = CovarianceCombination(sigmas=Sigmas, returns=returns)

    # Here we just combine all available matrices in one optimization problem
    result = combination.solve()
    print(result.t_start)
    print(result.t_end)
    print(result.Sigma)
    print(result.weights)

    # Here we iterate through time in windows of constant length
    for result in combination.solve_window(window=10, solver=cvxpy.ECOS):
        print(result.t_start)
        print(result.t_end)
        print(result.Sigma)
        print(result.weights)

    # one function rule them all...
    for result in f(returns, pairs, lower=-4.2, upper=4.2, window=10, solver=cvxpy.ECOS):
        print(result.weights)
