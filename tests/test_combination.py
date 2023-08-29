from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def frame():
    return pd.DataFrame(data=np.eye(2))


# def test_wrong_data(prices):
#     # Frame of returns
#     returns = prices.pct_change().dropna(axis=0, how="all")

#     # Pairs of halflife volatility vs halflife covariances
#     pairs = [(10, 10), (21, 21), (21, 63)]

#     combinator = from_ewmas(returns, pairs, clip_at=4.2)
#     print(type(combinator))

#     x = cp.Variable(shape=(1), name="x")

#     problem = cp.Problem(cp.Minimize(x[0]), [0 >= x, x >= 1])
#     problem.weights = pd.Series(1)
#     # problem.solve()
#     # print(problem.status)
#     # assert False
#     a = combinator._solve(time=returns.index[100], problem=problem)
#     print(a)
#     assert False
# with pytest.raises(cp.SolverError):
#     combinator._solve(time=returns.index[100], problem=problem)

# assert False
