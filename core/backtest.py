from typing import Iterator, Dict
import pandas as pd
from .entities import Asset, Portfolio
from .model import compute_returns, var_cvar, rebalance_policy

def rolling_windows(df: pd.DataFrame, window: int, step: int) -> Iterator[tuple[int, int]]:
    """Generator: yield (start, end) index pairs for rolling windows."""
    i = 0
    while i + window <= len(df):
        yield (i, i + window)
        i += step

def run_backtest(prices_long: pd.DataFrame,
                 init_weights: Dict[str, float],
                 rebalance_target: Dict[str, float],
                 window: int = 60, step: int = 20) -> Dict[str, float]:
    """Walk-forward backtest using rolling windows and simple rebalancing."""
    wide = prices_long.pivot(index="date", columns="ticker", values="adj_close").dropna(how="any")
    rets = compute_returns(wide)
    tickers = list(wide.columns)

    # build assets (composition)
    assets = {t: Asset(t, wide[t], rets[t]) for t in tickers}
    port = Portfolio(assets, init_weights.copy())

    port_curve = []
    curr_w = port.normalized_weights()
    for (s, e) in rolling_windows(rets, window, step):
        rw = rets.iloc[s:e]
        # rebalance check
        curr_w = rebalance_policy(curr_w, rebalance_target, tol=0.02)
        # daily portfolio returns (weighted sum)
        w_vec = [curr_w.get(t, 0.0) for t in tickers]
        p_rets = (rw * w_vec).sum(axis=1)
        port_curve.extend(list(p_rets.values))

    port_series = pd.Series(port_curve, index=rets.index[:len(port_curve)])
    total_ret = float((1 + port_series).prod() - 1)
    ann_vol = float(port_series.std() * (252 ** 0.5))
    v, cv = var_cvar(port_series, alpha=0.05)
    return {"total_return": total_ret, "ann_vol": ann_vol, "hist_VaR_5": v, "hist_CVaR_5": cv}
