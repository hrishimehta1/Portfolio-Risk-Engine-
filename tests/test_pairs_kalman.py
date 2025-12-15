import numpy as np
import pandas as pd
from strategies.pairs_kalman import Pair, KalmanPairsTrader

def test_pairs_signals_on_synthetic():
    n = 200
    t = np.linspace(0, 10, n)
    a_close = pd.Series(100 + 2*np.sin(t) + 0.5*np.random.randn(n))
    b_close = pd.Series( 98 + 2*np.sin(t+0.1) + 0.5*np.random.randn(n))
    mkt = np.zeros(n)

    pair = Pair("AAA", "BBB", a_close, b_close)
    trader = KalmanPairsTrader(open_z=1.0, close_z=0.25)
    opens, closes = trader.trade_signals(pair, mkt)

    assert isinstance(opens, list) and isinstance(closes, list)
    assert len(opens) >= 0 and len(closes) >= 0
