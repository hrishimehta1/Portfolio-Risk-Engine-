# Portfolio Risk & Pairs Trading Engine

**Course:** AAI/CPE/EE 551  
**Semester:** 2025 Fall  


---

## Project Title
**Portfolio Risk & Pairs Trading Engine (Backtesting, Risk Metrics, Optional Pairs Module)**

---

## Student Names & Emails
- **Hrishi Mehta** — hrishimehta009@gmail.com  
- **Mohan Dichpally** — mdichpal@stevens.edu

---

## Problem Description
Investors need a repeatable, offline way to evaluate portfolio risk and performance from historical prices without relying on external services. This project implements a modular engine that:

- Loads multi-asset price data from CSV,  
- Computes returns and runs a walk-forward backtest with a simple drift-based rebalancing policy,  
- Calculates risk KPIs (total return, annualized volatility, historical VaR/CVaR, parametric normal VaR, Sharpe, Sortino, max drawdown, Calmar, turnover, simple cost estimate),   
- Exports plots and a text summary so graders can quickly verify results.

The **main program and visuals** live in a **Jupyter Notebook**, while all **logic** is factored into **.py modules**.

---

## How to Use

### 1) Environment (Python 3.12 or 3.13)
```bash
python3.12 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run the CLI
```bash
# Minimal (uses included synthetic data)
python main.py --csv data/prices_small.csv

# With plots and saved artifacts
python main.py --csv data/prices_small.csv --plot --out-dir reports
```

**Outputs**
- Console prints KPI dictionary  
- `data/kpis.json` (machine-readable KPIs)  
- If `--plot`:
  - `reports/01_cumulative_returns.png`
  - `reports/02_drawdown.png`
  - `reports/03_return_hist.png`
  - `reports/SUMMARY.txt` (KPIs + config)

### 3) Notebook (`main.ipynb`)
Open **`notebooks/main.ipynb`** to run the full pipeline, reproduce plots, and experiment with parameters (`window`, `step`, `weights`).

### 4) Tests (PyTest)
```bash
pytest -q
```
- `tests/test_io.py` – schema & I/O exception tests  
- `tests/test_backtest.py` – KPI smoke test

---

## Data Source Policy
Repo ships **synthetic CSVs** to avoid IP issues and to guarantee offline execution:
- `data/prices_small.csv` (small example, 3–5 tickers)
- `data/prices_medium.csv` (longer sample)

If swapped for public data later, keep the same schema:
```
date,ticker,adj_close
2024-01-02,AAPL,187.4
...
```

---

## Project Structure
```
project-root/
├─ core/
│  ├─ __init__.py
│  ├─ entities.py       # Classes: TimeSeries (op overload), Asset, Portfolio (composition)
│  ├─ data.py           # load_prices_csv(), save_kpis(), schema checks
│  ├─ model.py          # compute_returns(), portfolio_metrics(), VaR/CVaR, drawdown, sharpe/sortino
│  ├─ backtest.py       # rolling_windows() generator + run_backtest()
│  └─ plot.py           # plot_cumulative(), plot_drawdown(), plot_return_hist()
├─ data/
│  ├─ prices_small.csv
│  └─ prices_medium.csv
├─ notebooks/
│  └─ main.ipynb
├─ tests/
│  ├─ test_io.py
│  └─ test_backtest.py
├─ reports/             # created at runtime if --plot is used
├─ main.py              # CLI entrypoint with __name__ guard
├─ requirements.txt
└─ README.md
```

---

## Canvas Part 1 Mapping

**Two classes + relationship:**  
- `Asset` and `Portfolio` in `core/entities.py` (composition: Portfolio “has many” Assets).  
- `TimeSeries` wrapper demonstrates operator overloading and readable `__str__`.

**Two+ meaningful functions:**  
- `compute_returns()` and `run_backtest()` (plus `portfolio_metrics()`, `max_drawdown()`, etc.).

**Advanced libraries (critical use):**  
- `pandas` (data transforms, pivots, quantiles), `numpy` (vector ops), `matplotlib` (plots).

**Exceptions (≥2) & tests (PyTest):**  
- `FileNotFoundError` / `SchemaError` in `load_prices_csv()`; `ValueError` in weight validation;  
- tests in `tests/test_io.py` and `tests/test_backtest.py`.

**Meaningful data I/O:**  
- CSV read (`data/*.csv`), JSON write (`data/kpis.json`), optional plots to `reports/`.

**Control flow:**  
- `while` loop in `rolling_windows()`, `for` loops over windows/dates, multiple `if` guards.

**Docstrings & comments:**  
- Each class/function includes concise docstrings and explanatory comments.

**README present:**  
- This file provides setup and usage guide.

---

## Canvas Part 2 Mapping (≥4 items met)
- **Special functions:** `map`, `zip`, `lambda` (rebalance/metric helpers).  
- **Comprehensions:** list/dict comprehensions throughout.  
- **Built-in modules:** `pathlib`, `argparse` (and `itertools` if used).  
- **Mutable & immutable:** uses `dict`, `list` (mutable) and `tuple`, `str` (immutable).  
- **Operator overloading:** `TimeSeries.__add__`, `__mul__`.  
- **Generator:** `rolling_windows()` yields `(start, end)` windows.  
- **`__name__`:** standard guard in `main.py`.  
- **`__str__`:** implemented for core classes to improve logs.  

*We exceed the minimum—all eight are present.*

---

## Main Contributions *(edit to reflect your commits)*
**Hrishi Mehta**  
- Backtesting pipeline & metrics (`portfolio_metrics()`), CLI plotting & artifacts  
- Notebook authoring and repo bootstrap; README initial draft

**Mohan Dichpally**  
- Data module & schema checks, tests, pairs strategy module  
- README finalization; dataset preparation and plots review

*Both members made ≥5 meaningful commits (code, tests, docs, data, and plots).*

---

## Rubric Guide for Graders
```bash
python main.py --csv data/prices_small.csv --plot --out-dir reports
```
- KPIs appear in console and `data/kpis.json`.  
- Plots saved in `reports/` (cumulative, drawdown, histogram) + `SUMMARY.txt`.  
- Notebook `notebooks/main.ipynb` reproduces visuals.  
- Tests: `pytest -q`.  
- Code: classes in `core/entities.py`, functions in `core/model.py` & `core/backtest.py`.

---

## Troubleshooting
- **ModuleNotFoundError: pandas** → `pip install -r requirements.txt` (inside the venv).  
- **Notebook path issues** → open the notebook from within the repo so `sys.path` bootstrap finds `core/`.  
- **Plots not showing in some IDEs** → add `%matplotlib inline` in the first cell.

---

## License / Notes
Repo contents are for AAI/CPE/EE 551 coursework.

