# core/model.py
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------- Returns & transforms ----------

def compute_returns(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple daily returns per column (ticker).
    Drops the first NA row from pct_change; fills residual NaNs with 0 for stability.
    """
    rets = prices_wide.pct_change().dropna(how="all")
    return rets.fillna(0.0)

# ---------- Risk metrics (historical always available; parametric optional) ----------

def var_historical(series: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR at level alpha (reported as positive loss magnitude)."""
    q = series.quantile(alpha)
    return float(-q)

def cvar_historical(series: pd.Series, alpha: float = 0.05) -> float:
    """Historical CVaR/Expected Shortfall at level alpha (positive loss magnitude)."""
    q = series.quantile(alpha)
    tail = series[series <= q]
    if len(tail) == 0:
        return 0.0
    return float(-tail.mean())

def var_parametric_normal(series: pd.Series, alpha: float = 0.05) -> Optional[float]:
    """
    Parametric Normal VaR at level alpha. Returns None if SciPy is unavailable.
    VaR = -(mu + z_alpha * sigma), reported as positive loss magnitude.
    """
    mu = float(series.mean())
    sigma = float(series.std(ddof=1))
    if sigma == 0:
        return 0.0
    try:
        from scipy.stats import norm  # optional dependency
        z = float(norm.ppf(alpha))
        var = -(mu + z * sigma)
        return float(max(var, 0.0))
    except Exception:
        return None

def portfolio_metrics(portfolio_rets: pd.Series, alpha: float = 0.05) -> Dict[str, float]:
    """
    Build a KPI dictionary from daily portfolio returns.
    Always includes: total_return, ann_vol, hist_VaR_5, hist_CVaR_5, sharpe, sortino, max_drawdown, calmar.
    If SciPy is installed, also includes norm_VaR_5.
    """
    # Growth & volatility
    total_ret = float((1.0 + portfolio_rets).prod() - 1.0)
    ann_vol = float(portfolio_rets.std(ddof=1) * np.sqrt(252.0))

    # Historical risk
    hist_var = var_historical(portfolio_rets, alpha=alpha)
    hist_cvar = cvar_historical(portfolio_rets, alpha=alpha)

    # Optional parametric risk
    norm_var_opt = var_parametric_normal(portfolio_rets, alpha=alpha)

    # Sharpe/Sortino (risk-free ~0, daily -> annualized)
    mean_daily = float(portfolio_rets.mean())
    std_daily = float(portfolio_rets.std(ddof=1))
    downside = float(portfolio_rets[portfolio_rets < 0].std(ddof=1)) if (portfolio_rets < 0).any() else 0.0
    sharpe = (mean_daily / std_daily * np.sqrt(252.0)) if std_daily > 0 else 0.0
    sortino = (mean_daily / downside * np.sqrt(252.0)) if downside > 0 else 0.0

    # Max drawdown & Calmar (use cumulative curve)
    eq = (1.0 + portfolio_rets).cumprod()
    peak = eq.cummax()
    drawdown = eq / peak - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    calmar = (total_ret / abs(max_dd)) if max_dd < 0 else 0.0

    kpis: Dict[str, float] = {
        "total_return": total_ret,
        "ann_vol": ann_vol,
        "hist_VaR_5": hist_var,
        "hist_CVaR_5": hist_cvar,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "n_days": int(len(portfolio_rets)),
    }
    if norm_var_opt is not None:
        kpis["norm_VaR_5"] = norm_var_opt
    return kpis

# ---------- Rebalancing policy (used by backtest) ----------

def rebalance_policy(curr_weights: Dict[str, float],
                     target: Dict[str, float],
                     tol: float = 0.02) -> Dict[str, float]:
    """
    If any weight deviates from target by more than tol, reset to target; else hold.
    Demonstrates dict comprehension + map/lambda (Part 2 features).
    """
    dev = {k: abs(curr_weights.get(k, 0.0) - target.get(k, 0.0)) for k in target}
    need_rebalance = any(map(lambda d: d > tol, dev.values()))
    return (target if need_rebalance else curr_weights)
