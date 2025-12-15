from pathlib import Path
import pandas as pd

class SchemaError(Exception):
    """Raised when a dataset is missing required columns."""
    pass

def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load a long-form prices CSV with columns: date,ticker,adj_close."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing prices file: {p.resolve()}")
    df = pd.read_csv(p)
    required = {"date", "ticker", "adj_close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise SchemaError(f"Prices CSV missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)

def save_kpis(path: str | Path, kpis: dict) -> None:
    """Save KPI dictionary as a JSON file (float cast for stability)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.Series({k: float(v) for k, v in kpis.items()}).to_json(p)
