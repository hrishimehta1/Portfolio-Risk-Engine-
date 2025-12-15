import numpy as np
import pandas as pd
from typing import Dict

def compute_returns(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """Compute simple returns per column; fill NaNs to keep demo robust."""
    rets = prices_wide.pct_change().dropna(how="all")
    return rets.fillna(0.0)

def var_cvar(portfolio_rets: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """Historical VaR/CVaR at level alpha (toy implementation)."""
    q = portfolio_rets.quantile(alpha)
    cvar = portfolio_rets[portfolio_rets <= q].mean()
    return float(-q), float(-cvar)

def rebalance_policy(curr_weights: Dict[str, float], target: Dict[str, float], tol: float = 0.02) -> Dict[str, float]:
    """If drift exceeds tol for any asset, snap back to target; else hold.
    Uses dict comprehension + lambda + map (to satisfy Part 2 features)."""
    dev = {k: abs(curr_weights.get(k, 0.0) - target.get(k, 0.0)) for k in target}
    need = any(map(lambda d: d > tol, dev.values()))
    return (target if need else curr_weights)
