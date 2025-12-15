from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import pandas as pd


@dataclass
class TimeSeries:
    """
    Thin wrapper around a pandas Series to illustrate operator overloading.

    Parameters
    ----------
    values : pd.Series
        The time series data indexed by datetime.
    """
    values: pd.Series

    def __add__(self, other: "TimeSeries") -> "TimeSeries":
        return TimeSeries(self.values.add(other.values, fill_value=0.0))

    def __mul__(self, scalar: float) -> "TimeSeries":
        return TimeSeries(self.values * float(scalar))

    def __str__(self) -> str:
        return f"TimeSeries(len={len(self.values)}, start={self.values.index.min()}, end={self.values.index.max()})"


@dataclass
class Asset:
    """
    Represents a single asset.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    prices : pd.Series
        Adjusted close price series (datetime index).
    returns : pd.Series
        Return series aligned to `prices` index.
    """
    ticker: str
    prices: pd.Series
    returns: pd.Series

    def __post_init__(self) -> None:
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise TypeError("Asset.prices must be indexed by DatetimeIndex.")
        if not self.prices.index.equals(self.returns.index):
            # align to be safe
            idx = self.prices.index.intersection(self.returns.index)
            self.prices = self.prices.loc[idx]
            self.returns = self.returns.loc[idx]

    def __str__(self) -> str:
        return f"Asset({self.ticker}, n={len(self.prices)})"


@dataclass
class Portfolio:
    """
    Composition: Portfolio 'has' many Asset and weights for them.

    Parameters
    ----------
    assets : Dict[str, Asset]
        Mapping from ticker to Asset.
    weights : Dict[str, float]
        Raw weights that will be normalized to sum to 1.
    """
    assets: Dict[str, Asset]
    weights: Dict[str, float]

    def normalized_weights(self) -> Dict[str, float]:
        s = float(sum(self.weights.values()))
        if s <= 0:
            raise ValueError("Sum of weights must be positive.")
        return {k: v / s for k, v in self.weights.items()}

    def __str__(self) -> str:
        return f"Portfolio(n_assets={len(self.assets)}, sum_weights={sum(self.weights.values()):.3f})"
