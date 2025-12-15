from __future__ import annotations
import argparse
from pathlib import Path

from core.data import load_prices_csv, save_kpis, SchemaError
from core.backtest import run_backtest

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Portfolio Risk Engine — CSV-only")
    p.add_argument("--csv", type=str, default="data/prices_small.csv",
                   help="Path to long CSV with columns: date,ticker,adj_close")
    p.add_argument("--window", type=int, default=60, help="Rolling window length (trading days)")
    p.add_argument("--step", type=int, default=20, help="Step size between windows")
    p.add_argument("--kpis-out", type=str, default="data/kpis.json",
                   help="Where to write KPIs JSON")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    prices = load_prices_csv(Path(args.csv))
    tickers = list(prices["ticker"].unique())
    if not tickers:
        raise ValueError("No tickers found in CSV.")
    # equal weights over tickers present in the CSV
    init_w = {t: 1.0 / len(tickers) for t in tickers}
    tgt_w = init_w.copy()

    kpis = run_backtest(prices, init_w, tgt_w, window=args.window, step=args.step)
    print("KPIs:", kpis)
    save_kpis(args.kpis_out, kpis)

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print("Missing file:", e)
    except SchemaError as e:
        print("Schema error:", e)
    except ValueError as e:
        print("Bad inputs:", e)
