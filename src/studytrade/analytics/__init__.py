"""
StudyTrade — Performance Analytics and Overfitting Verification
================================================================
"""

from studytrade.analytics.benchmark import (
    BenchmarkComparison,
    compare_against_buy_and_hold,
)
from studytrade.analytics.metrics import (
    PerformanceSummary,
    compute_drawdowns,
    compute_performance_metrics,
)
from studytrade.analytics.monte_carlo import (
    BootstrapInterval,
    MonteCarloDrawdownSummary,
    compute_bootstrap_ci,
    simulate_monte_carlo_drawdowns,
)
from studytrade.analytics.overfitting import compute_deflated_sharpe_ratio

__all__ = [
    "BenchmarkComparison",
    "BootstrapInterval",
    "MonteCarloDrawdownSummary",
    "PerformanceSummary",
    "compare_against_buy_and_hold",
    "compute_bootstrap_ci",
    "compute_deflated_sharpe_ratio",
    "compute_drawdowns",
    "compute_performance_metrics",
    "simulate_monte_carlo_drawdowns",
]
