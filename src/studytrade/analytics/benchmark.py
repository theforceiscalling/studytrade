"""
StudyTrade — Benchmark and Random-Entry Baseline Comparison
=============================================================

Benchmarks strategy performance against:
  1. Underlying Buy & Hold Index (NIFTY).
  2. Random Entry Baseline: A null-hypothesis model generating random coin-flip
     entries with identical exit rules and transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from studytrade.analytics.metrics import compute_drawdowns
from studytrade.constants import TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class BenchmarkComparison:
    """Comparison between Strategy and Benchmark Index."""

    strategy_cagr_pct: float
    benchmark_cagr_pct: float
    strategy_sharpe: float
    benchmark_sharpe: float
    strategy_max_dd_pct: float
    benchmark_max_dd_pct: float
    outperformance_alpha_pct: float


def compare_against_buy_and_hold(
    strategy_equity_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    initial_capital: float = 500000.0,
    risk_free_rate: float = 0.07,
) -> BenchmarkComparison:
    """
    Compare strategy equity time-series against underlying spot buy & hold.
    """
    if strategy_equity_df.empty or spot_df.empty:
        return BenchmarkComparison(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Strategy metrics
    strat_eq = strategy_equity_df["equity"].values
    n_days = max(1, len(strat_eq) - 1)
    strat_ret = np.diff(strat_eq) / strat_eq[:-1]
    strat_total_ret = (strat_eq[-1] - initial_capital) / initial_capital
    strat_cagr = ((1.0 + strat_total_ret) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0) * 100.0

    std_s = float(np.std(strat_ret, ddof=1)) if n_days > 1 else 1e-6
    strat_sharpe = float(
        (np.mean(strat_ret) - risk_free_rate / TRADING_DAYS_PER_YEAR)
        / std_s
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    _, strat_mdd, _ = compute_drawdowns(strat_eq)

    # Benchmark metrics
    spot_prices = spot_df["close"].values
    bench_ret = np.diff(spot_prices) / spot_prices[:-1]
    bench_total_ret = (spot_prices[-1] - spot_prices[0]) / spot_prices[0]
    bench_cagr = ((1.0 + bench_total_ret) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0) * 100.0

    std_b = float(np.std(bench_ret, ddof=1)) if n_days > 1 else 1e-6
    bench_sharpe = float(
        (np.mean(bench_ret) - risk_free_rate / TRADING_DAYS_PER_YEAR)
        / std_b
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    _, bench_mdd, _ = compute_drawdowns(spot_prices)

    alpha = strat_cagr - bench_cagr

    return BenchmarkComparison(
        strategy_cagr_pct=strat_cagr,
        benchmark_cagr_pct=bench_cagr,
        strategy_sharpe=strat_sharpe,
        benchmark_sharpe=bench_sharpe,
        strategy_max_dd_pct=strat_mdd,
        benchmark_max_dd_pct=bench_mdd,
        outperformance_alpha_pct=alpha,
    )
