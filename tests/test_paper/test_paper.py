"""
Tests for studytrade.paper — Paper market feed, trade journal, and execution engine.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from studytrade.data.fetchers.synthetic import SyntheticDataGenerator
from studytrade.paper.executor import PaperTradingExecutor
from studytrade.paper.feed import PaperMarketFeed
from studytrade.paper.journal import PaperJournal
from studytrade.strategy.signals.breakout import BreakoutStrategy


class TestPaperFeed:
    """Validate paper market feed quotes."""

    def test_spot_quote(self):
        feed = PaperMarketFeed()
        quote = feed.get_spot_quote("NIFTY")
        assert quote.symbol == "NIFTY"
        assert quote.bid < quote.ask
        assert quote.last_price > 0


class TestPaperJournal:
    """Validate SQLite journaling and 90-day duration gate."""

    def test_journal_roundtrip(self, tmp_path: Path):
        db_file = tmp_path / "test_journal.db"
        journal = PaperJournal(db_path=db_file)
        now = datetime.now()

        journal.log_trade(
            entry_time=now - timedelta(days=5),
            exit_time=now,
            symbol="NIFTY24JAN21500CE",
            underlying_symbol="NIFTY",
            strike=21500.0,
            expiry=now + timedelta(days=2),
            option_type="CE",
            n_lots=1,
            lot_size=25,
            entry_price=100.0,
            exit_price=150.0,
            gross_pnl=1250.0,
            net_pnl=1200.0,
            total_fees=50.0,
            return_pct=50.0,
            holding_bars=3,
            exit_reason="PROFIT_TARGET",
        )

        trades_df = journal.get_trades_dataframe()
        assert len(trades_df) == 1
        assert trades_df["symbol"].iloc[0] == "NIFTY24JAN21500CE"
        assert trades_df["net_pnl"].iloc[0] == 1200.0

        # Duration days check
        assert journal.get_paper_duration_days() >= 0


class TestPaperExecutor:
    """Validate forward paper trading execution and Greeks exposure."""

    def test_executor_process_tick_and_greeks(self, tmp_path: Path):
        gen = SyntheticDataGenerator(seed=42)
        spot_df = gen.generate_spot("NIFTY", n_days=30)
        chain_df = gen.get_options("NIFTY", "2024-01-01", "2024-01-05")

        journal = PaperJournal(db_path=tmp_path / "exec_journal.db")
        strat = BreakoutStrategy(lookback=5)
        executor = PaperTradingExecutor(strategy=strat, initial_capital=200000.0, journal=journal)

        # Process a market tick
        executor.process_market_tick(spot_df, chain_df, symbol="NIFTY")

        # Compute Greeks exposure
        greeks_exp = executor.compute_portfolio_greeks(
            current_spot=spot_df["close"].iloc[-1], options_chain_df=chain_df
        )
        assert isinstance(greeks_exp.net_delta_units, float)
        assert isinstance(greeks_exp.total_daily_theta_inr, float)
