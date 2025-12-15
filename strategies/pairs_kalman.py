from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd

try:
    from pykalman import KalmanFilter
except Exception:
    KalmanFilter = None  # graceful degrade if not installed for tests


@dataclass
class Pair:
    """
    Represents a trading pair (A,B) with price and residual spread series.
    """
    a: str
    b: str
    px_a: pd.Series
    px_b: pd.Series

    def ratio(self) -> pd.Series:
        """Price ratio A/B."""
        return (self.px_a / self.px_b).dropna()

    def log_spread(self) -> pd.Series:
        """Log spread ln(A) - beta*ln(B); beta via rolling OLS (simple)."""
        # quick beta estimate using rolling correlation/std or a static OLS.
        # static for simplicity:
        ln_a, ln_b = np.log(self.px_a), np.log(self.px_b)
        idx = ln_a.index.intersection(ln_b.index)
        X = ln_b.loc[idx].values.reshape(-1, 1)
        y = ln_a.loc[idx].values
        # OLS beta = (X'X)^(-1) X'y
        beta = float(np.linalg.lstsq(np.c_[X, np.ones_like(X)], y, rcond=None)[0][0])
        spread = ln_a.loc[idx] - beta * ln_b.loc[idx]
        return spread.dropna()


def kalman_smooth(y: pd.Series, process_var: float = 1e-3, obs_var: float = 1e-2) -> pd.Series:
    """Smooth a 1D series with a local level Kalman filter."""
    if KalmanFilter is None:
        # simple fallback: rolling mean as a stand-in
        return y.rolling(10, min_periods=3).mean()
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=y.iloc[0],
        initial_state_covariance=1.0,
        transition_covariance=process_var,
        observation_covariance=obs_var,
    )
    state_means, _ = kf.smooth(y.values)
    return pd.Series(state_means.ravel(), index=y.index, name="smooth")


def zscore(x: pd.Series, window: int = 60) -> pd.Series:
    mu = x.rolling(window, min_periods=max(10, window // 5)).mean()
    sig = x.rolling(window, min_periods=max(10, window // 5)).std(ddof=1)
    z = (x - mu) / sig.replace(0.0, np.nan)
    return z.fillna(0.0)


def backtest_pairs(
    pair: Pair,
    open_z: float = 2.0,
    close_z: float = 0.5,
    max_holding_days: int = 15,
    trans_cost_bps: float = 1.0,
) -> Dict[str, object]:
    """
    Simple mean-reversion pairs backtest:
      - Build log spread, smooth with Kalman
      - Trade when |z| > open_z, close when |z| < close_z or timeout
      - 1: long A/short B when z < -open_z (expect revert up)
         - and the opposite when z > open_z.
      - PnL in spread space, costs in bps each side.

    Returns
    -------
    dict with keys: 'trades' (DataFrame), 'kpis' (dict)
    """
    spread = pair.log_spread().dropna()
    smooth = kalman_smooth(spread)
    zs = zscore(smooth, window=60)

    pos = 0  # -1 short spread (A↓ vs B↑), +1 long spread (A↑ vs B↓)
    entry_idx = None
    trades = []

    for i, (dt, z) in enumerate(zs.items()):
        if pos == 0:
            if z > open_z:
                pos = -1
                entry_idx = dt
                entry_val = smooth.loc[dt]
            elif z < -open_z:
                pos = +1
                entry_idx = dt
                entry_val = smooth.loc[dt]
        else:
            # close condition
            days_held = (dt - entry_idx).days if entry_idx is not None else 0
            if abs(z) < close_z or days_held >= max_holding_days:
                exit_val = smooth.loc[dt]
                # spread PnL: long spread = exit - entry; short spread = entry - exit
                pnl = (exit_val - entry_val) * pos
                # transaction costs (both legs both sides ~ 4 legs): approximate
                costs = (trans_cost_bps / 1e4) * 4.0
                trades.append({
                    "entry": entry_idx, "exit": dt, "side": pos,
                    "entry_val": float(entry_val), "exit_val": float(exit_val),
                    "pnl": float(pnl - costs),
                    "days": days_held
                })
                pos = 0
                entry_idx = None

    trades_df = pd.DataFrame(trades)
    kpis = {}
    if not trades_df.empty:
        tot_pnl = float(trades_df["pnl"].sum())
        win_rate = float((trades_df["pnl"] > 0).mean())
        avg_days = float(trades_df["days"].mean())
        kpis = {"n_trades": int(len(trades_df)), "total_pnl_spread": tot_pnl, "win_rate": win_rate, "avg_days": avg_days}
    else:
        kpis = {"n_trades": 0, "total_pnl_spread": 0.0, "win_rate": 0.0, "avg_days": 0.0}

    return {"trades": trades_df, "kpis": kpis}
