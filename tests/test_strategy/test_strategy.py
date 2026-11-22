"""
Tests for studytrade.strategy — Strategy engine, selector, exit rules, and signal plugins.
"""

from __future__ import annotations

import pytest

from studytrade.constants import OptionType
from studytrade.data.fetchers.synthetic import SyntheticDataGenerator
from studytrade.strategy.rules import ExitReason, ExitRuleEvaluator
from studytrade.strategy.selector import ContractSelector
from studytrade.strategy.signals.breakout import BreakoutStrategy
from studytrade.strategy.signals.debit_spread import DebitSpreadStrategy
from studytrade.strategy.signals.momentum import MomentumStrategy
from studytrade.strategy.signals.vol_expansion import VolatilityExpansionStrategy


class TestContractSelector:
    """Test expiry and strike selection."""

    @pytest.fixture
    def sample_chain(self):
        gen = SyntheticDataGenerator(seed=42)
        chain = gen.get_options("NIFTY", "2024-01-01", "2024-01-02")
        return chain

    def test_select_expiry_dte_range(self, sample_chain):
        cur_ts = sample_chain["timestamp"].iloc[0]
        selected_exp = ContractSelector.select_expiry(sample_chain, cur_ts, min_dte=2, max_dte=20)
        assert selected_exp is not None
        dte = (selected_exp.date() - cur_ts.date()).days
        assert 2 <= dte <= 20

    def test_select_strike_by_delta(self, sample_chain):
        exp = sample_chain["expiry"].iloc[0]
        contract = ContractSelector.select_strike_by_delta(
            sample_chain,
            spot_price=21500.0,
            expiry=exp,
            option_type=OptionType.CALL,
            target_delta=0.35,
        )
        assert contract is not None
        assert contract.option_type == OptionType.CALL
        assert abs(contract.delta - 0.35) < 0.25
        assert contract.bid <= contract.ask

    def test_select_debit_spread_legs(self, sample_chain):
        exp = sample_chain["expiry"].iloc[0]
        spread = ContractSelector.select_debit_spread(
            sample_chain,
            spot_price=21500.0,
            expiry=exp,
            option_type=OptionType.CALL,
            long_delta=0.50,
            short_delta=0.25,
        )
        if spread is not None:
            # Long strike must be lower than short strike for bull call spread
            assert spread.long_leg.strike < spread.short_leg.strike
            assert spread.net_debit > 0


class TestExitRules:
    """Test systematic position exit triggers."""

    @pytest.fixture
    def evaluator(self):
        return ExitRuleEvaluator(
            stop_loss_pct=40.0,
            profit_target_pct=100.0,
            max_holding_bars=5,
            min_dte_cutoff=2,
            trail_trigger_pct=50.0,
            trail_giveback_pct=25.0,
        )

    def test_stop_loss_triggered(self, evaluator):
        # Entry at 100, bid drops to 55 (-45% loss)
        decision = evaluator.evaluate(
            entry_price=100.0,
            current_bid=55.0,
            peak_bid=100.0,
            bars_held=2,
            dte_days=10,
        )
        assert decision.should_exit
        assert decision.reason == ExitReason.STOP_LOSS

    def test_profit_target_triggered(self, evaluator):
        # Entry at 100, bid surges to 205 (+105% gain)
        decision = evaluator.evaluate(
            entry_price=100.0,
            current_bid=205.0,
            peak_bid=205.0,
            bars_held=3,
            dte_days=8,
        )
        assert decision.should_exit
        assert decision.reason == ExitReason.PROFIT_TARGET

    def test_theta_decay_cutoff(self, evaluator):
        # DTE drops to 1 day -> mandatory cutoff
        decision = evaluator.evaluate(
            entry_price=100.0,
            current_bid=95.0,
            peak_bid=105.0,
            bars_held=2,
            dte_days=1,
        )
        assert decision.should_exit
        assert decision.reason == ExitReason.THETA_DECAY

    def test_time_stop_triggered(self, evaluator):
        # Held 6 bars with negligible gain (5%) -> cut stale trade
        decision = evaluator.evaluate(
            entry_price=100.0,
            current_bid=105.0,
            peak_bid=110.0,
            bars_held=6,
            dte_days=12,
        )
        assert decision.should_exit
        assert decision.reason == ExitReason.TIME_STOP


class TestStrategySignals:
    """Test strategy signal generators."""

    @pytest.fixture
    def sample_spot(self):
        gen = SyntheticDataGenerator(seed=42)
        return gen.generate_spot("NIFTY", n_days=60)

    def test_breakout_strategy(self, sample_spot):
        strat = BreakoutStrategy(lookback=10)
        signals = strat.generate_signals(sample_spot)
        assert isinstance(signals, list)

    def test_momentum_strategy(self, sample_spot):
        strat = MomentumStrategy(fast_ema=5, slow_ema=15)
        signals = strat.generate_signals(sample_spot)
        assert isinstance(signals, list)

    def test_vol_expansion_strategy(self, sample_spot):
        strat = VolatilityExpansionStrategy(bb_period=10, squeeze_lookback=5)
        signals = strat.generate_signals(sample_spot)
        assert isinstance(signals, list)

    def test_debit_spread_wrapper(self, sample_spot):
        base_strat = BreakoutStrategy(lookback=10)
        spread_strat = DebitSpreadStrategy(base_strat)
        # Even if 0 signals on this slice, check interface
        signals = spread_strat.generate_signals(sample_spot)
        assert isinstance(signals, list)
