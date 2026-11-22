"""
Tests for studytrade.analytics — Performance metrics, Monte Carlo, and Deflated Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from studytrade.analytics.benchmark import compare_against_buy_and_hold
from studytrade.analytics.metrics import compute_drawdowns, compute_performance_metrics
from studytrade.analytics.monte_carlo import compute_bootstrap_ci, simulate_monte_carlo_drawdowns
from studytrade.analytics.overfitting import compute_deflated_sharpe_ratio


class TestPerformanceMetrics:
    """Validate performance calculations."""

    @pytest.fixture
    def sample_trades(self):
        return pd.DataFrame(
            {
                "net_pnl": [1000.0, -500.0, 1500.0, -400.0, 2000.0],
                "holding_bars": [2, 1, 3, 2, 4],
            }
        )

    @pytest.fixture
    def sample_equity(self):
        eq = [500000.0, 501000.0, 500500.0, 502000.0, 501600.0, 503600.0]
        return pd.DataFrame({"equity": eq})

    def test_performance_summary(self, sample_trades, sample_equity):
        summary = compute_performance_metrics(
            sample_trades, sample_equity, initial_capital=500000.0
        )
        assert summary.total_trades == 5
        assert summary.winning_trades == 3
        assert summary.losing_trades == 2
        assert summary.win_rate_pct == 60.0
        # Gains: 1000 + 1500 + 2000 = 4500. Losses: 500 + 400 = 900. PF = 5.0
        assert summary.profit_factor == 5.0
        assert summary.expectancy_inr > 0.0

    def test_compute_drawdowns(self):
        equity = pd.Series([100.0, 120.0, 110.0, 90.0, 130.0])
        # Peak was 120, low was 90 -> Drawdown = (90 - 120) / 120 = -25%
        dd_series, max_dd, max_duration = compute_drawdowns(equity)
        assert abs(max_dd - 25.0) < 1e-4


class TestMonteCarloAndOverfitting:
    """Validate Monte Carlo permutations and Deflated Sharpe."""

    def test_monte_carlo_drawdowns(self):
        trades = pd.DataFrame({"net_pnl": [500, -200, 300, -400, 100, -100] * 5})
        mc = simulate_monte_carlo_drawdowns(trades, initial_capital=100000.0, n_simulations=100)
        assert mc.worst_drawdown_pct >= mc.p95_drawdown_pct >= mc.p50_drawdown_pct

    def test_bootstrap_confidence_intervals(self):
        trades = pd.DataFrame({"net_pnl": [500, -200, 800, -100, 300]})
        ci = compute_bootstrap_ci(trades, n_bootstrap=100)
        assert "win_rate" in ci
        assert ci["win_rate"].ci_lower <= ci["win_rate"].mean <= ci["win_rate"].ci_upper

    def test_deflated_sharpe_ratio(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.015, 200)
        dsr = compute_deflated_sharpe_ratio(
            observed_sharpe=1.2, returns=returns, num_trials_tested=50
        )
        assert 0.0 <= dsr <= 1.0


class TestBenchmarkComparison:
    """Validate buy & hold benchmark comparisons."""

    def test_compare_against_buy_and_hold(self):
        eq_df = pd.DataFrame({"equity": np.linspace(500000, 550000, 50)})
        spot_df = pd.DataFrame({"close": np.linspace(20000, 21000, 50)})

        comp = compare_against_buy_and_hold(eq_df, spot_df)
        assert isinstance(comp.strategy_cagr_pct, float)
        assert isinstance(comp.benchmark_cagr_pct, float)
