from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import hashlib
import time

import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


class SchemaError(Exception):
    """Raised when input data has missing or malformed columns."""


@dataclass
class FetchConfig:
    period: str = "2y"          # e.g. '1y', '2y', '5y', '30d' (for 60m)
    interval: str = "1d"        # '1d', '1h', '60m', '1wk'
    cache_dir: Path = Path("data/cache")
    force_refresh: bool = False
    max_retries: int = 3
    retry_sleep: float = 1.25   # seconds


def _cache_key(tickers: Iterable[str], cfg: FetchConfig) -> Path:
    key_src = "|".join(sorted([t.upper() for t in tickers])) + f"|{cfg.period}|{cfg.interval}"
    h = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg.cache_dir / f"yf_{h}.csv"


def _normalize_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yahoo sometimes returns Date or Datetime index/column.
    Normalize to a 'date' datetime64[ns] column.
    """
    out = df.copy()
    if "Date" in out.columns:
        out = out.rename(columns={"Date": "date"})
    if "Datetime" in out.columns:
        out = out.rename(columns={"Datetime": "date"})
    if not isinstance(out.index, pd.DatetimeIndex) and "date" not in out.columns:
        # sometimes date is the index
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "date"})
    if "date" not in out.columns:
        # final fallback: promote index to date
        out = out.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"])
    return out


def fetch_or_cache_yf(
    tickers: list[str],
    cfg: Optional[FetchConfig] = None,
) -> pd.DataFrame:
    """
    Fetch Adjusted Close for multiple tickers from Yahoo Finance, with caching.

    Returns
    -------
    pd.DataFrame (long)
        Columns = ['date', 'ticker', 'adj_close']
    """
    if cfg is None:
        cfg = FetchConfig()
    cache_path = _cache_key(tickers, cfg)

    if cache_path.exists() and not cfg.force_refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        _validate_prices_df(df)
        print(f"[data] Loaded from cache: {cache_path}")
        return df

    if yf is None:
        raise ImportError("yfinance is not installed. pip install yfinance")

    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            data = yf.download(
                tickers=" ".join(tickers),
                period=cfg.period,
                interval=cfg.interval,
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            break
        except Exception as e:
            last_err = e
            print(f"[data] Attempt {attempt} failed: {e}")
            time.sleep(cfg.retry_sleep)
    else:
        raise RuntimeError(f"Failed to fetch from Yahoo after {cfg.max_retries} attempts: {last_err}")

    if isinstance(data.columns, pd.MultiIndex):
        long = []
        for t in tickers:
            col = (t, "Adj Close")
            if col not in data.columns:
                raise SchemaError(f"Missing Adj Close for {t}")
            s = data[col].dropna()
            frame = _normalize_index_column(s.to_frame(name="adj_close"))
            frame["ticker"] = t.upper()
            long.append(frame[["date", "ticker", "adj_close"]])
        df = pd.concat(long, ignore_index=True)
    else:
        # Single ticker
        if "Adj Close" not in data.columns:
            raise SchemaError("Missing 'Adj Close' in Yahoo output.")
        frame = _normalize_index_column(data[["Adj Close"]].rename(columns={"Adj Close": "adj_close"}))
        frame["ticker"] = tickers[0].upper()
        df = frame[["date", "ticker", "adj_close"]]

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    print(f"[data] Fetched and cached: {cache_path}")
    return df


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing prices file: {p.resolve()}")
    df = pd.read_csv(p, parse_dates=["date"])
    _validate_prices_df(df)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def _validate_prices_df(df: pd.DataFrame) -> None:
    required = {"date", "ticker", "adj_close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise SchemaError(f"Prices CSV missing columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise SchemaError("date must be datetime-like")
    if df["adj_close"].isna().any():
        raise SchemaError("adj_close contains NaNs; clean your input file.")


def save_kpis(path: str | Path, kpis: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.Series(kpis, dtype="float64").to_json(path, indent=2)
