from __future__ import annotations
from typing import Iterator, Dict, Optional, Tuple, Literal
import numpy as np
import pandas as pd

from .entities import Asset, Portfolio
from .model import compute_returns, portfolio_metrics, rebalance_policy_threshold


def rolling_windows(n_rows: int, window: int, step: int) -> Iterator[tuple[int, int]]:
    """
    Generator yielding (start, end) index pairs.
    Ensures windows fit within [0, n_rows].
    """
    i = 0
    while i + window <= n_rows:
        yield (i, i + window)
        i += step


def run_backtest(
    prices_long: pd.DataFrame,
    init_weights: Dict[str, float],
    rebalance_target: Dict[str, float],
    window: int = 60,
    step: int = 20,
    return_kind: Literal["simple", "log"] = "simple",
    rebalance_mode: Literal["threshold", "periodic"] = "threshold",
    drift_tol: float = 0.02,
    trans_cost_bps: float = 1.0,   # round-trip cost in basis points applied on weight changes
) -> Dict[str, float]:
    """
    Walk-forward backtest:
      1) pivot prices wide
      2) compute returns
      3) roll windows, rebalance according to mode
      4) apply transaction costs on turnover
      5) aggregate KPIs

    Returns a dictionary of KPIs (see model.portfolio_metrics) plus turnover and cost impact.
    """
    # 1) prices wide & returns
    wide = prices_long.pivot(index="date", columns="ticker", values="adj_close").dropna(how="any")
    rets = compute_returns(wide, kind=return_kind)
    tickers = list(wide.columns)

    # 2) build assets (composition)
    assets = {t: Asset(t, wide.loc[rets.index, t], rets[t]) for t in tickers}
    port = Portfolio(assets, init_weights.copy())

    # 3) simulate portfolio daily P&L with rebalancing
    curr_w = port.normalized_weights()
    port_ret_list = []
    turnover = 0.0
    total_cost = 0.0

    # vectorized daily returns matrix aligned to same index
    R = rets.values  # (T, N)
    dates = rets.index
    t_len, n_assets = R.shape
    weights_hist = []  

    # periodic bookkeeping
    period_counter = 0
    period_len = max(step, 1)

    for t in range(t_len):
        # rebalancing decision
        if rebalance_mode == "threshold":
            new_w = rebalance_policy_threshold(curr_weights=curr_w, target=rebalance_target, tol=drift_tol)
            did_rebalance = (new_w is not curr_w)
        else:  # "periodic"
            did_rebalance = (period_counter % period_len == 0 and t != 0)
            new_w = rebalance_target if did_rebalance else curr_w

        if did_rebalance:
            # approximate transaction cost as L1 change in weights * notional * bps
            l1 = sum(abs(new_w.get(k, 0.0) - curr_w.get(k, 0.0)) for k in tickers)
            turnover += l1
            # cost in returns space: subtract bps/10000 times l1 once
            cost_today = (trans_cost_bps / 1e4) * l1
            total_cost += cost_today
            curr_w = new_w

        # daily portfolio return (matrix mult)
        w_vec = np.array([curr_w.get(tic, 0.0) for tic in tickers], dtype=float)
        p_ret = float(np.nansum(R[t, :] * w_vec)) - (cost_today if did_rebalance else 0.0)
        port_ret_list.append(p_ret)
        weights_hist.append(curr_w.copy())
        period_counter += 1

    port_series = pd.Series(port_ret_list, index=dates, name="portfolio_rets")

    # 4) KPIs
    kpis = portfolio_metrics(port_series, alpha=0.05)
    kpis.update({
        "turnover_L1": float(turnover),
        "cost_estimate": float(total_cost),
        "n_days": int(t_len),
    })
    return kpis
