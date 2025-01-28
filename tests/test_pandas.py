from __future__ import annotations

import numpy as np
import pandas as pd


def test_division():
    x = pd.Series(index=[1, 2, 3, 4], data=[10, 11, 12, 14])
    y = pd.Series(index=[3, 4], data=[6, 7])
    a = x / y
    pd.testing.assert_series_equal(pd.Series(index=[1, 2, 3, 4], data=[np.nan, np.nan, 2.0, 2.0]), a)


def test_frame_all_nan():
    x = pd.DataFrame(index=["A", "B"], columns=["A", "B"], data=np.nan)
    assert np.isnan(x.values).all()


def test_series_all_nan():
    x = pd.Series(index=[1, 2], data=np.nan)
    assert np.isnan(x.values).all()


def test_shift():
    x = pd.Series(index=[1, 2], data=[10, 20])
    pd.testing.assert_series_equal(x.shift(1), pd.Series(index=[1, 2], data=[np.nan, 10]))
