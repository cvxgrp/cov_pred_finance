import pandas as pd

from cvx.covariance.ewma import iterated_ewma, cov
from cvx.covariance.covariance_combination import CovarianceCombination


def f(returns, pairs, clip_at=None, window=10, **kwargs):
    # compute the covariance matrices, one time series for each pair
    Sigmas = {f"{pair[0]}-{pair[1]}": cov(returns=returns, vola_halflife=pair[0], cov_halflife=pair[1], clip_at=clip_at) for pair in pairs}

    # combination of covariance matrix valued time series
    combination = CovarianceCombination(sigmas=Sigmas, returns=returns, window=window)

    for result in combination.solve(**kwargs):
        yield result

if __name__ == '__main__':
    prices = pd.read_csv("resources/stock_prices.csv", index_col=0, header=0, parse_dates=True).ffill()
    prices = prices[["GOOG", "AAPL", "FB"]].head(100)

    # Frame of returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    # compute the covariance matrices, one time series for each pair
    sigmas = {f"{pair[0]}-{pair[1]}": cov(returns=returns, vola_halflife=pair[0], cov_halflife=pair[1], clip_at=4.2) for pair in pairs}

    # combination of covariance matrix valued time series
    combination = CovarianceCombination(sigmas=sigmas, returns=returns, window=10)

    # Here we just combine all available matrices in one optimization problem
    for result in combination.solve():
        print(result)
        print(dir(result))
        print(result.time)
        print(result.covariance)
        print(result.weights)


    # one function rule them all...
    for result in f(returns, pairs, window=10):
        print(result.weights)
