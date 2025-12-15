from pathlib import Path
from core.io import load_prices_csv, save_kpis, SchemaError
from core.backtest import run_backtest

def main():
    data = Path("data") / "prices_small.csv"
    prices = load_prices_csv(data)
    init_w = {"AAA": 0.5, "BBB": 0.5}
    tgt_w  = {"AAA": 0.5, "BBB": 0.5}
    kpis = run_backtest(prices, init_w, tgt_w, window=2, step=1)
    print("KPIs:", kpis)
    save_kpis("data/kpis.json", kpis)

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print("Missing file:", e)
    except SchemaError as e:
        print("Schema error:", e)
    except ValueError as e:
        print("Bad weights:", e)
