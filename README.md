# Portfolio Risk & Pairs Trading Engine

**Goal:** Backtest a simple long-only portfolio with periodic drift checks and measure return, volatility, and historical VaR/CVaR. Includes a Kalman-based pairs signal module.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Run (script)
```bash
python main.py
```

## Notebook
Open `notebooks/main.ipynb` and run the cells to:
- Load `data/prices_small.csv` and run a walk-forward backtest
- Fetch/cache Yahoo Finance data (AAPL, MSFT, SPY) and plot pairs residual spread + signals

## Requirements (rubric mapping)
- Classes with composition: `Portfolio` has `Asset`; `KalmanPairsTrader` works with `Pair`
- Functions: `compute_returns`, `run_backtest`, etc.
- Advanced libs: pandas, numpy, matplotlib, yfinance, pykalman
- Exceptions: FileNotFoundError, SchemaError, ValueError
- Data I/O: CSV load/save + cache; KPIs → JSON
- Control flow: for/while/if patterns
- Part 2: map/zip/lambda, comprehensions, pathlib, operator overloading (TimeSeries), generator (`rolling_windows()`), `__name__`, `__str__`

## Notes
- Pin to Python **3.12** for library compatibility.
- A tiny CSV is included so tests run offline.
