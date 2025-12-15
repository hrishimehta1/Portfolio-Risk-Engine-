import pandas as pd
from core.backtest import run_backtest


def _toy_data():
    # 6 days, 2 tickers; simple for deterministic tests
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01","2024-01-02","2024-01-03",
            "2024-01-01","2024-01-02","2024-01-03"
        ]),
        "ticker": ["AAA","AAA","AAA","BBB","BBB","BBB"],
        "adj_close": [100,101,102,50,49,51]
    })
    return df


def test_backtest_kpis_present():
    df = _toy_data()
    init_w = {"AAA": 0.6, "BBB": 0.4}
    tgt_w  = {"AAA": 0.6, "BBB": 0.4}
    kpis = run_backtest(df, init_w, tgt_w, window=2, step=1, rebalance_mode="threshold", drift_tol=0.02, trans_cost_bps=1.0)
    expected = {"total_return","ann_vol","hist_VaR","hist_CVaR","norm_VaR","sharpe","sortino","max_drawdown","calmar","turnover_L1","cost_estimate","n_days"}
    assert expected.issubset(set(kpis))
