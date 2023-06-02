# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest

from cvx.covariance.regularization import regularize_covariance


@pytest.fixture
def returns(prices):
    return 100 * prices.pct_change().fillna(0.0)


@pytest.fixture
def matrices(returns):
    covs = returns.ewm(com=40, min_periods=40).cov()
    covs = covs.dropna(how="all", axis=0)
    return covs


def test_regularization(matrices):
    timestamps = matrices.index.get_level_values(0)
    aaa = {t: matrices.loc[t] for t in timestamps}

    regularized = dict(regularize_covariance(sigmas=aaa, r=1))
    print(regularized)
