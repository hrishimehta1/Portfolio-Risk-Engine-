from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class TimeSeries:
    """Thin wrapper on a pandas Series with operator overloading for demos."""
    values: pd.Series

    def __add__(self, other: "TimeSeries") -> "TimeSeries":
        return TimeSeries(self.values.add(other.values, fill_value=0))

    def __mul__(self, scalar: float) -> "TimeSeries":
        return TimeSeries(self.values * scalar)

    def __str__(self) -> str:
        return f"TimeSeries(len={len(self.values)})"

@dataclass
class Asset:
    """Represents a single asset's price and return series."""
    ticker: str
    prices: pd.Series
    returns: pd.Series

    def __str__(self) -> str:
        return f"Asset({self.ticker}, n={len(self.prices)})"

@dataclass
class Portfolio:
    """Composition: Portfolio 'has' many Assets, and weights for them."""
    assets: dict[str, Asset]
    weights: dict[str, float]

    def normalized_weights(self) -> dict[str, float]:
        """Return weights that sum to 1, raising if invalid."""
        s = float(sum(self.weights.values()))
        if s <= 0.0:
            raise ValueError("Sum of weights must be positive.")
        return {k: v / s for k, v in self.weights.items()}
