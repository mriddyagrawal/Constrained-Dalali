import pytest
import pandas as pd
import numpy as np

from utils import compute_rsi
def test_compute_rsi_constant_increase():
    # If the price strictly increases, loss is 0.
    # The current implementation uses loss.replace(0, np.nan), so rs becomes NaN,
    # and therefore the RSI becomes NaN.
    series = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    rsi = compute_rsi(series, window=14)
    # The first 14 values will be NaN due to the rolling window of 14,
    # where the first value requires 14 diffs, meaning 15 original values.
    # Actually diff makes first element NaN.
    assert np.isnan(rsi.iloc[-1])

def test_compute_rsi_constant_decrease():
    # If the price strictly decreases, gain is 0, loss is positive.
    # rs = 0 / positive = 0
    # RSI = 100 - (100 / (1 + 0)) = 0
    series = pd.Series([25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
    rsi = compute_rsi(series, window=14)
    assert rsi.iloc[-1] == 0.0

def test_compute_rsi_alternating():
    # Alternating prices
    series = pd.Series([10, 15, 10, 15, 10, 15, 10, 15, 10, 15, 10, 15, 10, 15, 10, 15])
    rsi = compute_rsi(series, window=14)
    # The average gain and average loss will be identical
    # rs = 1, RSI = 100 - (100 / 2) = 50
    assert rsi.iloc[-1] == 50.0

def test_compute_rsi_shorter_than_window():
    series = pd.Series([10, 11, 12])
    rsi = compute_rsi(series, window=14)
    assert rsi.isna().all()

def test_compute_rsi_all_same():
    series = pd.Series([10] * 20)
    rsi = compute_rsi(series, window=14)
    assert rsi.isna().all()

def test_compute_rsi_custom_window():
    series = pd.Series([10, 12, 11, 13, 12, 14, 13, 15])
    # window = 3
    # diffs: [NaN, 2, -1, 2, -1, 2, -1, 2]
    # For last element, window 3 diffs: [2, -1, 2]
    # gain: [2, 0, 2] -> mean = 4/3
    # loss: [0, 1, 0] -> mean = 1/3
    # rs = 4
    # RSI = 100 - (100 / (1 + 4)) = 100 - 20 = 80
    rsi = compute_rsi(series, window=3)
    assert np.isclose(rsi.iloc[-1], 80.0)
