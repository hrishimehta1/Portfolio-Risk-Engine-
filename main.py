from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from core.data import load_prices_csv, fetch_or_cache_yf, FetchConfig, save_kpis, SchemaError
from core.backtest import run_backtest
from strategies.pairs_kalman import Pair, backtest_pairs


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Portfolio Risk & Pairs Trading Engine")
    ap.add_argument("--data", default=None, help="Path to long CSV with columns [date,ticker,adj_close]. If omitted, use --tickers to fetch.")
    ap.add_argument("--tickers", default="AAPL,MSFT,GOOGL", help="Comma-separated tickers for Yahoo fetch.")
    ap.add_argument("--period", default="2y", help="Yahoo period (e.g., 1y, 2y, 5y)")
    ap.add_argument("--interval", default="1d", help="Yahoo interval (e.g., 1d, 1h, 1wk)")
    ap.add_argument("--force-refresh", action="store_true", help="Ignore cache and refetch Yahoo data.")
    ap.add_argument("--weights", default=None, help="JSON path with {\"init\": {...}, \"target\": {...}}.")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--rebalance", default="threshold", choices=["threshold", "periodic"])
    ap.add_argument("--drift-tol", type=float, default=0.02)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--pairs", default=None, help="Comma pair 'AAPL,MSFT' to run pairs demo.")
    ap.add_argument("--out", default="data/kpis.json", help="Path to save KPI JSON.")
    return ap.parse_args()


def _load_weights(path: str | None, tickers: list[str]):
    init_w = {t: 1.0 / len(tickers) for t in tickers}
    tgt_w = init_w.copy()
    if path:
        with open(path) as f:
            w = json.load(f)
        init_w = w.get("init", init_w)
        tgt_w = w.get("target", tgt_w)
    return init_w, tgt_w


def main() -> None:
    args = _parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    # Load long prices data (CSV or Yahoo)
    if args.data:
        prices_long = load_prices_csv(args.data)
    else:
        cfg = FetchConfig(period=args.period, interval=args.interval, force_refresh=args.force_refresh)
        prices_long = fetch_or_cache_yf(tickers, cfg=cfg)

    # Portfolio backtest
    init_w, tgt_w = _load_weights(args.weights, tickers)
    print(f"[cfg] tickers={tickers} window={args.window} step={args.step} rebalance={args.rebalance} tol={args.drift_tol} cost_bps={args.cost_bps}")
    kpis = run_backtest(
        prices_long=prices_long,
        init_weights=init_w,
        rebalance_target=tgt_w,
        window=args.window,
        step=args.step,
        rebalance_mode=args.rebalance,
        drift_tol=args.drift_tol,
        trans_cost_bps=args.cost_bps,
    )
    print("[portfolio] KPIs:", kpis)
    save_kpis(args.out, kpis)

    #pairs demo 
    if args.pairs:
        a, b = [x.strip().upper() for x in args.pairs.split(",")]
        px = prices_long.pivot(index="date", columns="ticker", values="adj_close").dropna()
        if a not in px.columns or b not in px.columns:
            print(f"[pairs] Skipping: {a},{b} not both in data.")
            return
        pair = Pair(a, b, px[a], px[b])
        res = backtest_pairs(pair, open_z=2.0, close_z=0.5, max_holding_days=15, trans_cost_bps=args.cost_bps)
        trades = res["trades"]
        trades_path = Path("data/trades.csv")
        if not trades.empty:
            trades.to_csv(trades_path, index=False)
            print(f"[pairs] trades saved: {trades_path}")
        print("[pairs] kpis:", res["kpis"])


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print("Missing file:", e)
    except SchemaError as e:
        print("Schema error:", e)
    except ValueError as e:
        print("Value error:", e)
    except Exception as e:
        print("Unhandled error:", repr(e))
