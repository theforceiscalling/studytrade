"""
Tests for studytrade.risk — Position sizing, loss limits, and emergency kill switch.
"""

from __future__ import annotations

from studytrade.risk.limits import RiskManager
from studytrade.risk.sizing import PositionSizer


class TestPositionSizing:
    """Validate position sizing formulas."""

    def test_fixed_fractional_sizing(self):
        # Capital: 500,000. Risk: 1.5% -> Max risk = 7,500.
        # Option premium: 100. SL: 50% -> Risk/unit = 50.
        # Lot size = 25 -> Risk/lot = 1,250.
        # Expected lots = floor(7500 / 1250) = 6 lots.
        lots = PositionSizer.fixed_fractional(
            capital=500000.0,
            risk_pct=1.5,
            option_premium=100.0,
            stop_loss_pct=50.0,
            lot_size=25,
            max_lots=10,
        )
        assert lots == 6

    def test_volatility_adjusted_reduces_size(self):
        # When current NATR is double baseline NATR, position size should decrease
        normal_lots = PositionSizer.volatility_adjusted(
            capital=500000.0,
            base_risk_pct=1.5,
            current_natr=1.0,
            baseline_natr=1.0,
            option_premium=100.0,
            stop_loss_pct=50.0,
            lot_size=25,
        )
        high_vol_lots = PositionSizer.volatility_adjusted(
            capital=500000.0,
            base_risk_pct=1.5,
            current_natr=2.0,  # Elevated volatility
            baseline_natr=1.0,
            option_premium=100.0,
            stop_loss_pct=50.0,
            lot_size=25,
        )
        assert high_vol_lots < normal_lots

    def test_half_kelly_zero_on_negative_edge(self):
        # If win rate is 30% and payoff is 1.0 (negative expectancy), lots must be 0
        lots = PositionSizer.half_kelly(
            win_rate=0.30,
            payoff_ratio=1.0,
            capital=500000.0,
            option_premium=100.0,
            stop_loss_pct=50.0,
            lot_size=25,
        )
        assert lots == 0


class TestRiskLimits:
    """Validate circuit breakers and risk limits."""

    def test_concurrent_position_blocking(self):
        rm = RiskManager(initial_capital=500000.0, max_concurrent_positions=2)
        # 1 position open -> Allowed
        res1 = rm.check_trade_allowed(current_open_count=1, proposed_premium_cost=5000.0)
        assert res1.allowed

        # 2 positions open -> Blocked
        res2 = rm.check_trade_allowed(current_open_count=2, proposed_premium_cost=5000.0)
        assert not res2.allowed
        assert "Max concurrent positions reached" in res2.reason

    def test_daily_loss_limit(self):
        rm = RiskManager(initial_capital=500000.0, max_daily_loss_pct=5.0)
        # Lose 30,000 (6% loss)
        rm.update_capital(470000.0)

        res = rm.check_trade_allowed(current_open_count=0, proposed_premium_cost=5000.0)
        assert not res.allowed
        assert "Daily loss limit hit" in res.reason

    def test_kill_switch_lockout(self):
        rm = RiskManager(initial_capital=500000.0, kill_switch_loss_pct=15.0)
        # Lose 80,000 (16% disaster drawdown)
        rm.update_capital(420000.0)
        assert rm.is_kill_switch_triggered

        res = rm.check_trade_allowed(current_open_count=0, proposed_premium_cost=1000.0)
        assert not res.allowed
        assert "KILL SWITCH ACTIVATED" in res.reason
