from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

@dataclass
class Pair:
    """A stock pair with aligned close prices."""
    a: str
    b: str
    a_close: pd.Series
    b_close: pd.Series

    def ratio(self) -> pd.Series:
        """Price ratio A/B."""
        return self.a_close / self.b_close

    def spread(self, market_excess: np.ndarray, gamma: float) -> np.ndarray:
        """Residual spread: (A - B) - gamma * market_excess, length-aligned."""
        n = min(len(self.a_close), len(self.b_close), len(market_excess))
        return (self.a_close.values[:n] - self.b_close.values[:n]) - gamma * market_excess[:n]

    def __str__(self) -> str:
        return f"Pair({self.a},{self.b}, n={len(self.a_close)})"

@dataclass
class KalmanPairsTrader:
    """Kalman/Vasicek-smoothed residual spread with threshold entry/exit."""
    theta: float = 0.0
    kappa: float = 0.1
    sigma: float = 0.1
    gamma: float = 0.5
    delta_t: float = 1/252
    open_z: float = 1.5
    close_z: float = 0.4

    def _kalman_smooth(self, y: np.ndarray) -> np.ndarray:
        F = np.array([[np.exp(-self.kappa * self.delta_t)]])
        H = np.array([[1.0]])
        Q = np.array([[self.sigma**2 * (1 - np.exp(-2 * self.kappa * self.delta_t)) / (2 * self.kappa)]])
        R = np.array([[1.0]])
        kf = KalmanFilter(
            transition_matrices=F, observation_matrices=H,
            transition_covariance=Q, observation_covariance=R,
            initial_state_mean=self.theta,
            initial_state_covariance=np.array([[self.sigma**2/(2*self.kappa)]])
        )
        means, _ = kf.smooth(y)
        return means.ravel()

    def trade_signals(self, pair: Pair, market_excess: np.ndarray) -> tuple[list[int], list[int]]:
        res = pair.spread(market_excess, self.gamma)
        smoothed = self._kalman_smooth(res)
        opens: list[int] = []; closes: list[int] = []; position = False
        for i, m in enumerate(smoothed):
            z = abs(m - self.theta)
            if (not position) and z > self.open_z:
                opens.append(i); position = True
            elif position and z < self.close_z:
                closes.append(i); position = False
        return opens, closes
