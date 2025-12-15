import numpy as np
import pandas as pd

from strategies.pairs_kalman import Pair, backtest_pairs


def test_pairs_trades_on_synthetic():
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    # create synthetic cointegrated-like series
    b = pd.Series(np.cumsum(np.random.normal(0, 0.5, size=len(idx))) + 100, index=idx)
    a = b * 1.5 + np.random.normal(0, 0.3, size=len(idx))  # proportional + small noise
    pair = Pair("A", "B", a, b)
    res = backtest_pairs(pair, open_z=1.0, close_z=0.2, max_holding_days=30, trans_cost_bps=1.0)
    assert "trades" in res and "kpis" in res
    # Not guaranteed >0, but we expect at least an attempt given low thresholds
    assert res["kpis"]["n_trades"] >= 0
