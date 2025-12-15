from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
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
        """
        Log spread ln(A) - beta*ln(B); beta via static OLS on overlapping dates.
        """
        ln_a, ln_b = np.log(self.px_a), np.log(self.px_b)
        idx = ln_a.index.intersection(ln_b.index)
        X = ln_b.loc[idx].values.reshape(-1, 1)
        y = ln_a.loc[idx].values
        beta, intercept = np.linalg.lstsq(np.c_[X, np.ones_like(X)], y, rcond=None)[0]
        spread = ln_a.loc[idx] - beta * ln_b.loc[idx]
        return spread.dropna()


def kalman_smooth(y: pd.Series, process_var: float = 1e-3, obs_var: float = 1e-2) -> pd.Series:
    """Smooth a 1D series with a local level Kalman filter (fallback = rolling mean)."""
    if KalmanFilter is None:
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
      - +1 long spread (A↑ vs B↓) when z < -open_z; -1 when z > open_z
      - PnL computed in spread space; costs approximate 4 legs in/out.

    Returns
    -------
    dict with keys: 'trades' (DataFrame), 'kpis' (dict)
    """
    spread = pair.log_spread().dropna()
    if len(spread) < 30:
        return {"trades": pd.DataFrame(), "kpis": {"n_trades": 0, "total_pnl_spread": 0.0, "win_rate": 0.0, "avg_days": 0.0}}

    smooth = kalman_smooth(spread)
    zs = zscore(smooth, window=60)

    pos = 0  # -1 short spread, +1 long spread
    entry_dt = None
    entry_val = None
    trades = []

    for dt, z in zs.items():
        if pos == 0:
            if z > open_z:
                pos, entry_dt, entry_val = -1, dt, float(smooth.loc[dt])
            elif z < -open_z:
                pos, entry_dt, entry_val = +1, dt, float(smooth.loc[dt])
        else:
            days_held = int((dt - entry_dt).days) if entry_dt is not None else 0
            if abs(z) < close_z or days_held >= max_holding_days:
                exit_val = float(smooth.loc[dt])
                pnl = (exit_val - entry_val) * pos
                costs = (trans_cost_bps / 1e4) * 4.0
                trades.append({
                    "entry": entry_dt, "exit": dt, "side": pos,
                    "entry_val": entry_val, "exit_val": exit_val,
                    "pnl": float(pnl - costs),
                    "days": days_held
                })
                pos, entry_dt, entry_val = 0, None, None

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        kpis = {"n_trades": 0, "total_pnl_spread": 0.0, "win_rate": 0.0, "avg_days": 0.0}
    else:
        kpis = {
            "n_trades": int(len(trades_df)),
            "total_pnl_spread": float(trades_df["pnl"].sum()),
            "win_rate": float((trades_df["pnl"] > 0).mean()),
            "avg_days": float(trades_df["days"].mean()),
        }
    return {"trades": trades_df, "kpis": kpis}


def plot_spread_and_z(spread: pd.Series, window: int = 60):
    import matplotlib.pyplot as plt
    smooth = kalman_smooth(spread)
    zs = zscore(smooth, window=window)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(smooth.index, smooth, label="Kalman-smoothed spread")
    ax1.axhline(0, linestyle="--", linewidth=1)
    ax1.set_title("Spread (smoothed) and Z-score")
    ax2 = ax1.twinx()
    ax2.plot(zs.index, zs, alpha=0.5, label="z-score")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    return fig
