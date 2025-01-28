from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cvx.covariance.ewma import iterated_ewma

# import test data
resource_dir = Path(__file__).parent / "resources"
returns = (
    pd.read_csv(resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True)
    .ffill()
    .pct_change()
    .iloc[1:]
)

np.random.seed(0)
mask = np.random.choice([False, True], size=returns.shape, p=[0.9, 0.1])
returns[mask] = np.nan


# Iterated EWWMA covariance prediction
iewma_pair = (63, 125)
iewma = list(
    iterated_ewma(
        returns,
        vola_halflife=iewma_pair[0],
        cov_halflife=iewma_pair[1],
        min_periods_vola=0,
        min_periods_cov=0,
    )
)
iewma = {iterate.time: iterate.covariance for iterate in iewma}

covariance = iewma[returns.index[-1]]
covariance.to_csv(resource_dir / "cov_iewma_nan.csv")
