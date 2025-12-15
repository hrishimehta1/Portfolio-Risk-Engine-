from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import itertools
import pandas as pd

from core.data import fetch_or_cache_yf, FetchConfig
from strategies.pairs_kalman import Pair, backtest_pairs


# ----- sector universes (uppercased) -----
XLY = ['AMZN', 'TSLA', 'MCD', 'HD', 'LOW', 'NKE', 'SBUX', 'TJX', 'TGT', 'BKNG']
XLP = ['PG', 'PEP', 'KO', 'COST', 'WMT', 'PM', 'MO', 'CL', 'ADM']
XLF = ['JPM', 'BAC', 'WFC', 'SCHW', 'GS', 'MS', 'SPGI', 'BLK', 'CB']
XLV = ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'BMY', 'DHR']
XLI = ['RTX', 'HON', 'UNP', 'UPS', 'LMT', 'CAT', 'DE', 'GE', 'NOC', 'BA']
XLB = ['LIN', 'APD', 'SHW', 'CTVA', 'FCX', 'ECL', 'NUE', 'NEM', 'DOW', 'ALB']
XLK = ['AAPL', 'MSFT', 'V', 'NVDA', 'MA', 'AVGO', 'CSCO', 'ACN', 'CRM', 'ADBE', 'QCOM', 'IBM', 'AMD', 'AMAT', 'INTC']
QQQ = ['AAPL', 'AMZN', 'MSFT', 'GOOG', 'META', 'NVDA', 'TSLA', 'PYPL', 'ADBE', 'NFLX']
XLC = ['META', 'GOOGL', 'NFLX', 'CHTR', 'CMCSA', 'TMUS', 'DIS', 'T', 'VZ']
SMH = ['TSM', 'NVDA', 'ASML', 'AVGO', 'TXN', 'ADI', 'KLAC', 'LRCX', 'QCOM', 'INTC', 'AMAT', 'MU', 'AMD']


@dataclass
class ScanConfig:
    period: str = "2y"             # '30d' works for intraday like '60m'
    interval: str = "1d"           # '1d', '60m', etc.
    open_z: float = 2.0
    close_z: float = 0.5
    max_hold_days: int = 15
    cost_bps: float = 1.0
    out_dir: Path = Path("data/scans")
    save_plots: bool = False


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rank_and_save(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    if df.empty:
        (out_dir / f"{name}_empty.csv").write_text("no candidates\n")
        return
    # rank: win_rate then total pnl
    rank = df.sort_values(["win_rate", "total_pnl_spread"], ascending=[False, False])
    rank.to_csv(out_dir / f"{name}.csv", index=False)


def scan_universe(universe: List[str], cfg: ScanConfig) -> pd.DataFrame:
    """
    Pulls data once for the universe, runs all ordered pairs A!=B,
    returns a summary dataframe with at least:
    ['a','b','n_trades','win_rate','total_pnl_spread','avg_days']
    """
    _ensure_dir(cfg.out_dir)
    yf_cfg = FetchConfig(period=cfg.period, interval=cfg.interval, force_refresh=False)
    prices_long = fetch_or_cache_yf(universe, cfg=yf_cfg)
    wide = prices_long.pivot(index="date", columns="ticker", values="adj_close").dropna()

    rows = []
    for a, b in itertools.permutations(universe, 2):
        if a not in wide.columns or b not in wide.columns:
            continue
        pair = Pair(a, b, wide[a], wide[b])
        res = backtest_pairs(pair,
                             open_z=cfg.open_z,
                             close_z=cfg.close_z,
                             max_holding_days=cfg.max_hold_days,
                             trans_cost_bps=cfg.cost_bps)
        k = res["kpis"]
        rows.append({
            "a": a, "b": b,
            "n_trades": k["n_trades"],
            "win_rate": k["win_rate"],
            "total_pnl_spread": k["total_pnl_spread"],
            "avg_days": k["avg_days"]
        })

   

    return pd.DataFrame(rows)


def scan_all(cfg: ScanConfig) -> None:
    universes = {
        "XLY": XLY, "XLP": XLP, "XLF": XLF, "XLV": XLV, "XLI": XLI,
        "XLB": XLB, "XLK": XLK, "QQQ": QQQ, "XLC": XLC, "SMH": SMH
    }
    _ensure_dir(cfg.out_dir)
    for name, uni in universes.items():
        print(f"[scan] {name} ({len(uni)} tickers) …")
        df = scan_universe(uni, cfg)
        _rank_and_save(df, name, cfg.out_dir)
        print(f"[scan] saved {name} -> {cfg.out_dir / (name + '.csv')}")


if __name__ == "__main__":
    # quick manual run example:
    cfg = ScanConfig(period="2y", interval="1d", open_z=2.0, close_z=0.5, max_hold_days=20, cost_bps=1.5, save_plots=False)
    scan_all(cfg)
