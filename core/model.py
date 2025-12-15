from __future__ import annotations
from typing import Dict, Tuple, Literal, Optional
import numpy as np
import pandas as pd


def compute_returns(prices_wide: pd.DataFrame,kind: Literal["simple", "log"] = "simple",) -> pd.DataFrame:
    """
    Compute returns per column (ticker).

    Parameters
    ----------
    prices_wide : pd.DataFrame
        Wide dataframe of adj_close (index = date, columns = tickers).
    kind : {'simple','log'}
        Return type.

    Returns
    -------
    pd.DataFrame
        Return matrix aligned to `prices_wide` index.
    """
    px = prices_wide.ffill().dropna(how="any")
    if kind == "log":
        rets = np.log(px / px.shift(1))
    else:
        rets = px.pct_change()
    rets = rets.dropna(how="all").fillna(0.0)
    return rets


def var_cvar_historical(portfolio_rets: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    """Historical VaR/CVaR (positive numbers = losses)."""
    q = portfolio_rets.quantile(alpha)
    cvar = portfolio_rets[portfolio_rets <= q].mean()
    return float(-q), float(-cvar)


def var_parametric_normal(portfolio_rets: pd.Series, alpha: float = 0.05) -> float:
    """Parametric VaR under Normal(μ, σ)."""
    mu = float(portfolio_rets.mean())
    sigma = float(portfolio_rets.std(ddof=1))
    if sigma == 0.0:
        return 0.0
    from math import sqrt
    # inverse CDF for normal at alpha:
    from scipy.stats import norm  
    z = float(norm.ppf(alpha))
    var = -(mu + z * sigma)
    return var


def sharpe_ratio(portfolio_rets: pd.Series, rf_daily: float = 0.0) -> float:
    ex = portfolio_rets - rf_daily
    denom = ex.std(ddof=1)
    return float(0.0 if denom == 0 else (ex.mean() / denom) * np.sqrt(252.0))


def sortino_ratio(portfolio_rets: pd.Series, rf_daily: float = 0.0) -> float:
    ex = portfolio_rets - rf_daily
    downside = ex[ex < 0.0].std(ddof=1)
    return float(0.0 if downside == 0 else (ex.mean() / downside) * np.sqrt(252.0))


def max_drawdown(portfolio_rets: pd.Series) -> float:
    curve = (1.0 + portfolio_rets).cumprod()
    peak = curve.cummax()
    dd = (curve / peak) - 1.0
    return float(dd.min())


def calmar_ratio(portfolio_rets: pd.Series) -> float:
    ann_return = float((1.0 + portfolio_rets).prod() ** (252.0 / len(portfolio_rets)) - 1.0)
    mdd = abs(max_drawdown(portfolio_rets))
    return float(0.0 if mdd == 0 else ann_return / mdd)


def rebalance_policy_threshold(curr_weights: Dict[str, float], target: Dict[str, float], tol: float = 0.02) -> Dict[str, float]:
    """
    If *any* absolute deviation > tol, snap back to target. Else hold.
    Uses dict comprehension + map/lambda for Part 2.
    """
    dev = {k: abs(curr_weights.get(k, 0.0) - target.get(k, 0.0)) for k in target}
    need = any(map(lambda d: d > tol, dev.values()))
    return (target if need else curr_weights)


def portfolio_metrics(port_series: pd.Series, alpha: float = 0.05) -> Dict[str, float]:
    """Compute a rich KPI set from daily portfolio returns."""
    total_return = float((1.0 + port_series).prod() - 1.0)
    ann_vol = float(port_series.std(ddof=1) * np.sqrt(252.0))
    var_hist, cvar_hist = var_cvar_historical(port_series, alpha=alpha)
    var_norm = var_parametric_normal(port_series, alpha=alpha)
    sr = sharpe_ratio(port_series)
    sor = sortino_ratio(port_series)
    mdd = max_drawdown(port_series)
    calmar = calmar_ratio(port_series)
    return {
        "total_return": total_return,
        "ann_vol": ann_vol,
        "hist_VaR": var_hist,
        "hist_CVaR": cvar_hist,
        "norm_VaR": var_norm,
        "sharpe": sr,
        "sortino": sor,
        "max_drawdown": mdd,
        "calmar": calmar,
    }
