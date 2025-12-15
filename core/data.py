from pathlib import Path
import pandas as pd
import yfinance as yf

class SchemaError(Exception):
    """Raised when a fetched dataset is missing required columns."""
    pass

def fetch_or_cache_yf(tickers: list[str], period: str = "1y", interval: str = "1d", cache_dir: str = "data/cache") -> pd.DataFrame:
    """Fetch price data via yfinance and cache per-ticker CSVs. Returns long DataFrame."""
    out: list[pd.DataFrame] = []
    cdir = Path(cache_dir); cdir.mkdir(parents=True, exist_ok=True)

    for t in tickers:
        cache = cdir / f"{t.upper()}_{period}_{interval}.csv"
        try:
            if cache.exists():
                df = pd.read_csv(cache, parse_dates=["Datetime"])
            else:
                hist = yf.Ticker(t).history(period=period, interval=interval, actions=False)
                if hist.empty or "Close" not in hist.columns:
                    raise SchemaError(f"No 'Close' prices returned for {t}.")
                df = hist.reset_index()[["Datetime", "Close"]]
                df.to_csv(cache, index=False)
            df["ticker"] = t.upper()
            df = df.rename(columns={"Datetime": "date", "Close": "adj_close"})
            out.append(df[["date", "ticker", "adj_close"]])
        except Exception as e:
            print(f"Warning: failed to fetch {t}: {e}")
    if not out:
        raise FileNotFoundError("No ticker data available after fetching/caching.")
    return pd.concat(out).sort_values(["ticker", "date"]).reset_index(drop=True)
