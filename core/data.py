from __future__ import annotations
from pathlib import Path
import pandas as pd

class SchemaError(Exception):
    """Raised when an input CSV lacks required columns."""

def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """
    Load long-format prices with columns: date, ticker, adj_close.
    Enforces schema and sorts for stable downstream behavior.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing prices file: {p.resolve()}")
    df = pd.read_csv(p)
    required = {"date", "ticker", "adj_close"}
    missing = required - set(df.columns)
    if missing:
        raise SchemaError(f"Prices CSV missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)

def save_kpis(path: str | Path, kpis: dict) -> None:
    """Write KPIs to JSON (creates parent folders if needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(kpis, dtype="float64").to_json(p)
