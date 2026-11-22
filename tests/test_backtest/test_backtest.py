"""
Tests for studytrade.backtest — Backtesting engine, fill simulator, and costs.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from studytrade.backtest.costs import compute_nse_option_costs
from studytrade.backtest.engine import BacktestEngine
from studytrade.backtest.fills import FillSimulator
from studytrade.backtest.portfolio import Portfolio
from studytrade.constants import OptionType
from studytrade.data.fetchers.synthetic import SyntheticDataGenerator
from studytrade.strategy.signals.breakout import BreakoutStrategy


class TestCostsAndFills:
    """Validate fee models and fill simulator."""

    def test_nse_costs_breakdown(self):
        costs = compute_nse_option_costs(
            buy_price=100.0,
            sell_price=150.0,
            lot_size=25,
            n_lots=2,
            slippage_ticks=1.0,
            tick_size=0.05,
        )
        # Units = 50
        # Buy turnover = 5000, Sell turnover = 7500, Total = 12500
        assert costs.turnover_buy == 5000.0
        assert costs.turnover_sell == 7500.0
        assert costs.brokerage == 40.0  # 2 orders * 20
        assert costs.stt > 0
        assert costs.exchange_txn > 0
        assert costs.gst > 0
        assert costs.total_statutory_fees > 45.0

    def test_fill_simulator_spread_crossing(self):
        sim = FillSimulator(slippage_ticks=2.0, tick_size=0.05)
        now = datetime.now()

        # Buy crosses ask: 100.0 + 2*0.05 = 100.10
        buy_fill = sim.execute_buy(now, "NIFTY24JAN21500CE", ask_price=100.0, lot_size=25)
        assert buy_fill.executed_price == 100.10

        # Sell crosses bid: 150.0 - 2*0.05 = 149.90
        sell_fill = sim.execute_sell(now, "NIFTY24JAN21500CE", bid_price=150.0, lot_size=25)
        assert sell_fill.executed_price == 149.90

    def test_expiry_settlement(self):
        # Call with strike 21500, spot settles at 21650 -> Payoff = 150
        payoff_call = FillSimulator.settle_at_expiry(21500.0, 21650.0, OptionType.CALL)
        assert payoff_call == 150.0

        # Call with strike 21500, spot settles at 21400 -> Payoff = 0
        payoff_otm = FillSimulator.settle_at_expiry(21500.0, 21400.0, OptionType.CALL)
        assert payoff_otm == 0.0


class TestPortfolio:
    """Validate position and cash management."""

    def test_portfolio_open_and_close(self):
        port = Portfolio(initial_capital=200000.0)
        now = datetime.now()

        # Buy 1 lot (25 units) at 100 -> premium = 2500
        opened = port.open_position(
            timestamp=now,
            symbol="TEST_OPT",
            underlying_symbol="NIFTY",
            strike=21500.0,
            expiry=now,
            option_type=OptionType.CALL,
            entry_price=100.0,
            lot_size=25,
            n_lots=1,
        )
        assert opened
        assert port.cash == 200000.0 - 2500.0
        assert len(port.open_positions) == 1

        # Sell at 150 -> gross = 3750
        record = port.close_position(now, "TEST_OPT", exit_price=150.0, exit_reason="PROFIT_TARGET")
        assert record is not None
        assert record.gross_pnl == 1250.0  # (150 - 100) * 25
        assert record.net_pnl < record.gross_pnl  # Fees deducted
        assert port.cash > 200000.0  # Net profit gained


class TestBacktestEngine:
    """Validate full backtesting execution on synthetic data."""

    def test_end_to_end_backtest_run(self):
        gen = SyntheticDataGenerator(seed=42)
        spot_df = gen.generate_spot("NIFTY", n_days=40)
        options_df = gen.get_options("NIFTY", "2024-01-01", "2024-01-20")

        strat = BreakoutStrategy(lookback=5)
        engine = BacktestEngine(strategy=strat, initial_capital=300000.0)
        res = engine.run(spot_df, options_df, symbol="NIFTY")

        assert isinstance(res.trades_df, pd.DataFrame)
        assert isinstance(res.equity_df, pd.DataFrame)
        assert len(res.equity_df) == len(spot_df)
        assert res.final_equity > 0
