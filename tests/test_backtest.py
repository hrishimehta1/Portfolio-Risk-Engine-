from core.backtest import run_backtest
import pandas as pd

def test_backtest_runs_minimal():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01","2024-01-02","2024-01-03","2024-01-01","2024-01-02","2024-01-03"]),
        "ticker": ["A","A","A","B","B","B"],
        "adj_close": [100,101,102,50,49,51]
    })
    init_w = {"A": 0.6, "B": 0.4}
    tgt_w  = {"A": 0.6, "B": 0.4}
    kpis = run_backtest(df, init_w, tgt_w, window=2, step=1)
    assert set(kpis).issuperset({"total_return","ann_vol","hist_VaR_5","hist_CVaR_5"})
