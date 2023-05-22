import pandas as pd
from cvx.covariance.covariance_combination import CovarianceCombination
from cvx.covariance.ewma import iterated_ewma


### TODO: is there a better way to import data???
returns = pd.read_csv("../tests/resources/stock_prices.csv", index_col=0, header=0, parse_dates=True).ffill()[["GOOG", "AAPL", "FB"]].pct_change().dropna(axis=0, how="all")


### Iterated EWWMA covariance prediction
iewma = iterated_ewma(returns, vola_halflife=10, cov_halflife=21, min_periods_vola=20, min_periods_cov=20, clip_at=4.2)
Sigma = iewma[returns.index[-1]]
Sigma.to_csv("../tests/resources/Sigma_iewma.csv")

### Covariance combination
pairs = [(10, 10), (21, 21), (21, 63)]
Sigmas = {f"{pair[0]}-{pair[1]}": iterated_ewma(returns, vola_halflife=pair[0], cov_halflife=pair[1], clip_at=4.2) for pair in pairs}

combinator = CovarianceCombination(Sigmas=Sigmas, returns=returns)
results = combinator.solve(time=returns.index[-1], window=10000)
weights = results.weights
print(weights)
Sigma = results.covariance
weights.to_csv("../tests/resources/weights_combinator.csv", header=False)
Sigma.to_csv("../tests/resources/Sigma_combinator.csv")