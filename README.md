
# Portfolio Risk & Pairs Trading Engine

**Goal:** Backtest a simple long‑only portfolio (returns, volatility, historical VaR/CVaR) and optionally evaluate a mean‑reversion **pairs** strategy using a Kalman‑smoothed spread and z‑score triggers.

## Features
- **Clean module layout** (`core/`, `strategies/`, `tests/`, `notebooks/`).
- **Real data** via Yahoo Finance with caching (daily or intraday intervals).
- **Walk‑forward backtest** with simple drift‑check rebalancing.
- **Risk metrics:** total return, annualized volatility, historical VaR/CVaR.
- **Pairs strategy:** log‑spread, Kalman smoothing fallback, z‑score entries/exits.
- **Pytest** smoke tests and schema checks.
- **Notebook** for visuals and exploratory runs.

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

### Run
Minimal run with CSV sample:
```bash
python main.py
```

Real data (daily):
```bash
python main.py --tickers AAPL,MSFT,GOOGL,NVDA --period 2y --interval 1d
```

Intraday (60m, last 30 days — Yahoo limitation):
```bash
python main.py --tickers AAPL,MSFT --period 30d --interval 60m
```

Optional single‑pair demo from CLI:
```bash
python main.py --tickers AAPL,MSFT,GOOGL --pairs AAPL,MSFT
```

### Sector‑wide Pairs Scan
```bash
python strategies/pairs_scan.py
# outputs CSVs ranked by win_rate and total_pnl_spread under data/scans/
```

## Repository Layout
```
portfolio-risk-engine/
├─ notebooks/
│  └─ main.ipynb
├─ core/
│  ├─ __init__.py
│  ├─ data.py          # Yahoo fetch/cache + CSV load/save + schema checks
│  ├─ entities.py      # TimeSeries (op overload), Asset, Portfolio (composition)
│  ├─ model.py         # compute_returns(), var_cvar(), rebalance_policy()
│  └─ backtest.py      # rolling generator + run_backtest()
├─ strategies/
│  ├─ pairs_kalman.py  # Pair, Kalman/zscore backtest, plotting helper
│  └─ pairs_scan.py    # scan universes, write ranked CSVs
├─ tests/
│  ├─ test_io.py       # schema + file load
│  └─ test_backtest.py # minimal backtest run
├─ data/
│  ├─ prices_small.csv
│  └─ (cache/, scans/ created on demand)
├─ main.py
├─ README.md
├─ requirements.txt
└─ pyproject.toml
```

## Notebook Tips
If you see `ModuleNotFoundError: No module named 'core'` inside the notebook, add the repo root to `sys.path`. The provided `notebooks/main.ipynb` already includes a cell that does:

```python
import sys, os, pathlib
repo_root = pathlib.Path.cwd().parent
sys.path.insert(0, str(repo_root))
```

## Tests
```bash
pytest -q
```
- `tests/test_io.py`: CSV load and schema errors.
- `tests/test_backtest.py`: minimal backtest KPI presence.

## Requirements
See `requirements.txt` (pandas, numpy, matplotlib, yfinance, pytest, pykalman (optional for pairs)).

## License
For course use only.
