# core/backtest.py
from __future__ import annotations

from typing import Dict, Iterator, List, Tuple, Union

import pandas as pd

from .entities import Asset, Portfolio
from .model import compute_returns, portfolio_metrics, rebalance_policy


def rolling_windows(df: pd.DataFrame, window: int, step: int) -> Iterator[tuple[int,int]]:
    """Generator: yield (start, end) index pairs for rolling windows."""
    i = 0
    n = len(df)
    while i + window <= n:
        yield (i, i + window)
        i += step

def run_backtest(
    prices_long: pd.DataFrame,
    init_weights: Dict[str, float],
    rebalance_target: Dict[str, float],
    window: int = 60,
    step: int = 20,
    return_series: bool = False
) -> Union[Dict[str, float], Tuple[Dict[str, float], pd.Series]]:
    """
    Walk-forward backtest with overlap-safe stitching:
      - pivot long -> wide
      - compute daily returns per ticker
      - iterate rolling windows, apply rebalance policy
      - compute daily portfolio returns per window
      - CONCAT all window series on their DATE index and resolve overlaps
    Returns KPIs; if return_series=True, also returns the stitched daily portfolio return series.
    """
    # wide price table
    wide = prices_long.pivot(index="date", columns="ticker", values="adj_close").dropna(how="any")
    rets = compute_returns(wide)               # index = dates, columns = tickers
    tickers = list(wide.columns)

    # composition: Portfolio has Assets (kept for rubric clarity)
    assets = {t: Asset(t, wide[t], rets[t]) for t in tickers}
    port = Portfolio(assets, init_weights.copy())

    # initial normalized weights & aligned Series by column name
    curr_w = port.normalized_weights()
    w_series = pd.Series(curr_w, index=tickers, dtype="float64").fillna(0.0)

    # collect per-window portfolio series (indexed by dates of that slice)
    segments: List[pd.Series] = []

    for (s, e) in rolling_windows(rets, window, step):
        rw = rets.iloc[s:e]  # window returns (DataFrame, columns=tickers, index=dates in window)

        # rebalance check -> update aligned weight series
        curr_w = rebalance_policy(curr_w, rebalance_target, tol=0.02)
        w_series = pd.Series(curr_w, index=tickers, dtype="float64").fillna(0.0)

        # column-wise multiply (align by column labels), then sum to portfolio return
        p_rets = rw.mul(w_series, axis=1).sum(axis=1)  # Series indexed by dates in [s:e)
        p_rets.name = "portfolio_ret"
        segments.append(p_rets)

    if not segments:
        # degenerate case (e.g., not enough rows for one window)
        return ({"total_return": 0.0, "ann_vol": 0.0, "hist_VaR_5": 0.0, "hist_CVaR_5": 0.0, "n_days": 0},
                pd.Series(name="portfolio_ret")) if return_series else \
               {"total_return": 0.0, "ann_vol": 0.0, "hist_VaR_5": 0.0, "hist_CVaR_5": 0.0, "n_days": 0}

    # Stitch all overlapping windows by DATE index.
    # We take the LAST window’s value for any duplicate date.
    stitched = pd.concat(segments).groupby(level=0).last().sort_index()
    stitched.name = "portfolio_ret"

    kpis = portfolio_metrics(stitched, alpha=0.05)
    if return_series:
        return kpis, stitched
    return kpis
