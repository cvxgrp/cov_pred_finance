# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from cvx.covariance.regularization import em_regularize_covariance
from cvx.covariance.regularization import regularize_covariance


def test_em():
    np.random.seed(0)
    A = np.random.randn(10, 10)

    Sigma = {"time": pd.DataFrame(A @ A.T)}
    Sigma_reg = dict(regularize_covariance(Sigma, 2, low_rank_format=True))
    Sigma_em = dict(em_regularize_covariance(Sigma, Sigma_reg))

    F_test = pd.DataFrame(
        np.array(
            [
                [-3.03970054, 0.57577317],
                [-0.6231586, 2.26692157],
                [0.3302449, 0.51237691],
                [1.78743535, 1.05064365],
                [0.51603338, -2.3063646],
                [1.73142215, 0.51311818],
                [2.30309145, -1.26758546],
                [0.16607721, -0.26199752],
                [1.66306842, 1.96699766],
                [-0.1759997, 0.7431212],
            ]
        )
    )

    d_test = pd.Series(
        np.array(
            [
                5.36390008,
                0.61685613,
                19.68099245,
                3.92494831,
                9.52083963,
                0.17725664,
                1.30694537,
                5.10017934,
                7.55662054,
                6.06427555,
            ]
        )
    )

    pd.testing.assert_frame_equal(Sigma_em["time"].F, F_test)
    pd.testing.assert_series_equal(Sigma_em["time"].d, d_test)
