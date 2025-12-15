# main.py  (CSV-only + optional plotting)
from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from core.backtest import run_backtest
from core.data import SchemaError, load_prices_csv, save_kpis
from core.plot import plot_cumulative, plot_drawdown, plot_return_hist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Portfolio Risk Engine — CSV-only")
    p.add_argument("--csv", type=str, default="data/prices_small.csv",
                   help="Path to long CSV with columns: date,ticker,adj_close")
    p.add_argument("--window", type=int, default=60, help="Rolling window (days)")
    p.add_argument("--step", type=int, default=20, help="Step between windows")
    p.add_argument("--kpis-out", type=str, default="data/kpis.json",
                   help="Where to write KPIs JSON")
    p.add_argument("--plot", action="store_true",
                   help="Save plots (cumulative, drawdown, return hist) to --out-dir")
    p.add_argument("--out-dir", type=str, default="reports",
                   help="Directory to save plots and summaries when --plot is used")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    prices = load_prices_csv(Path(args.csv))
    tickers = list(prices["ticker"].unique())
    if not tickers:
        raise ValueError("No tickers found in CSV.")
    init_w = {t: 1.0 / len(tickers) for t in tickers}
    tgt_w  = init_w.copy()

    if args.plot:
        kpis, series = run_backtest(prices, init_w, tgt_w, window=args.window, step=args.step, return_series=True)
    else:
        kpis = run_backtest(prices, init_w, tgt_w, window=args.window, step=args.step)

    print("KPIs:")
    pprint(kpis)
    save_kpis(args.kpis_out, kpis)
    print(f"Saved KPIs JSON -> {args.kpis_out}")

    if args.plot:
        outdir = Path(args.out_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        p1 = plot_cumulative(series, outdir)
        p2 = plot_drawdown(series, outdir)
        p3 = plot_return_hist(series, outdir)
        # Simple text summary alongside images
        summary = outdir / "SUMMARY.txt"
        summary.write_text(
            "Portfolio Risk Engine — Report\n"
            f"Source CSV: {args.csv}\n"
            f"Window={args.window}, Step={args.step}\n\n"
            f"KPIs:\n{kpis}\n\n"
            f"Figures:\n- {p1.name}\n- {p2.name}\n- {p3.name}\n"
        )
        print(f"Saved plots & summary -> {outdir}")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print("Missing file:", e)
    except SchemaError as e:
        print("Schema error:", e)
    except ValueError as e:
        print("Bad inputs:", e)
