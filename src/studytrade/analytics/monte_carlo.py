"""
StudyTrade — Monte Carlo Simulation and Bootstrap Confidence Intervals
========================================================================

Breaks trade sequence path dependency via permutation and bootstrap resampling:
  - Reshuffles trade order B = 1000 times to generate distribution of Max Drawdowns.
  - Computes 95% bootstrap confidence intervals for Win Rate and Profit Factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloDrawdownSummary:
    """Monte Carlo drawdown distribution."""

    p50_drawdown_pct: float
    p95_drawdown_pct: float
    p99_drawdown_pct: float
    worst_drawdown_pct: float


@dataclass(frozen=True)
class BootstrapInterval:
    """Confidence interval from bootstrap resampling."""

    metric_name: str
    mean: float
    ci_lower: float  # 2.5th percentile
    ci_upper: float  # 97.5th percentile


def simulate_monte_carlo_drawdowns(
    trades_df: pd.DataFrame,
    initial_capital: float = 500000.0,
    n_simulations: int = 1000,
    seed: int = 42,
) -> MonteCarloDrawdownSummary:
    """
    Simulate trade-sequence permutations to discover risk of ruin and worst-case drawdowns.
    """
    if trades_df.empty:
        return MonteCarloDrawdownSummary(0.0, 0.0, 0.0, 0.0)

    pnls = trades_df["net_pnl"].values
    rng = np.random.default_rng(seed)

    max_drawdowns = np.empty(n_simulations)

    for i in range(n_simulations):
        permuted_pnls = rng.permutation(pnls)
        equity_curve = initial_capital + np.cumsum(permuted_pnls)
        equity_curve = np.insert(equity_curve, 0, initial_capital)

        running_max = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - running_max) / running_max * 100.0
        max_drawdowns[i] = abs(float(np.min(dd)))

    return MonteCarloDrawdownSummary(
        p50_drawdown_pct=float(np.percentile(max_drawdowns, 50)),
        p95_drawdown_pct=float(np.percentile(max_drawdowns, 95)),
        p99_drawdown_pct=float(np.percentile(max_drawdowns, 99)),
        worst_drawdown_pct=float(np.max(max_drawdowns)),
    )


def compute_bootstrap_ci(
    trades_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, BootstrapInterval]:
    """
    Compute 95% bootstrap confidence intervals for Win Rate and Profit Factor.
    """
    if trades_df.empty:
        return {}

    pnls = trades_df["net_pnl"].values
    n = len(pnls)
    rng = np.random.default_rng(seed)

    win_rates = np.empty(n_bootstrap)
    profit_factors = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        sample = rng.choice(pnls, size=n, replace=True)
        wins = sample[sample > 0]
        losses = sample[sample < 0]

        win_rates[b] = (len(wins) / n) * 100.0
        sum_w = float(np.sum(wins)) if len(wins) > 0 else 0.0
        sum_l = float(np.sum(np.abs(losses))) if len(losses) > 0 else 1e-6
        profit_factors[b] = sum_w / sum_l

    return {
        "win_rate": BootstrapInterval(
            metric_name="Win Rate (%)",
            mean=float(np.mean(win_rates)),
            ci_lower=float(np.percentile(win_rates, 2.5)),
            ci_upper=float(np.percentile(win_rates, 97.5)),
        ),
        "profit_factor": BootstrapInterval(
            metric_name="Profit Factor",
            mean=float(np.mean(profit_factors)),
            ci_lower=float(np.percentile(profit_factors, 2.5)),
            ci_upper=float(np.percentile(profit_factors, 97.5)),
        ),
    }
