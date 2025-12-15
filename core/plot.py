# core/plot.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def cum_curve(series: pd.Series) -> pd.Series:
    return (1.0 + series).cumprod()

def drawdown(series: pd.Series) -> pd.Series:
    equity = cum_curve(series)
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd

def plot_cumulative(series: pd.Series, outdir: Path) -> Path:
    _ensure_dir(outdir)
    path = outdir / "01_cumulative_return.png"
    plt.figure(figsize=(9,4))
    (cum_curve(series)).plot()
    plt.title("Portfolio Cumulative Return")
    plt.xlabel("Date"); plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path

def plot_drawdown(series: pd.Series, outdir: Path) -> Path:
    _ensure_dir(outdir)
    path = outdir / "02_drawdown.png"
    dd = drawdown(series)
    plt.figure(figsize=(9,3.5))
    dd.plot()
    plt.title("Portfolio Drawdown")
    plt.xlabel("Date"); plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path

def plot_return_hist(series: pd.Series, outdir: Path) -> Path:
    _ensure_dir(outdir)
    path = outdir / "03_return_hist.png"
    plt.figure(figsize=(7,4))
    series.dropna().hist(bins=40)
    plt.title("Distribution of Daily Returns")
    plt.xlabel("Daily Return"); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path
