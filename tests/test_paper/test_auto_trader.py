"""
Unit tests for Autonomous Auto-Trader and Rules-Based Square-Off Engine.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import pytest

from studytrade.paper.auto_trader import AutoTrader
from studytrade.paper.journal import PaperJournal


def test_auto_trader_config_and_toggle(tmp_path):
    trader = AutoTrader(data_dir=tmp_path)
    assert trader.config["enabled"] is True
    assert trader.config["profit_target_pct"] == 50.0
    assert trader.config["stop_loss_pct"] == 30.0

    # Test parameter update
    updated = trader.save_config({"profit_target_pct": 60.0, "stop_loss_pct": 25.0})
    assert updated["profit_target_pct"] == 60.0
    assert updated["stop_loss_pct"] == 25.0

    # Test master toggle
    is_active = trader.toggle_master()
    assert is_active is False
    assert trader.config["enabled"] is False

    is_active_again = trader.toggle_master()
    assert is_active_again is True


def test_auto_trader_profit_target_square_off(tmp_path):
    trader = AutoTrader(data_dir=tmp_path)
    trader.config["profit_target_pct"] = 50.0
    trader.config["stop_loss_pct"] = 30.0

    # Add open position with entry 100, target 150, SL 70
    pos_id = trader.journal.add_open_position(
        symbol="NIFTY_TEST_CALL",
        underlying_symbol="NIFTY",
        strike=25000.0,
        expiry=(datetime.now() + timedelta(days=5)).isoformat(),
        option_type="CE",
        n_lots=1,
        lot_size=25,
        entry_price=100.0,
        current_price=100.0,
        target_price=150.0,
        stop_loss_price=70.0,
    )
    assert pos_id > 0
    assert len(trader.journal.get_open_positions()) == 1

    # Simulate price reaching 155 (target hit)
    trader.tick_cache.update_tick(
        "NIFTY_TEST_CALL",
        {"symbol": "NIFTY_TEST_CALL", "last_price": 155.0, "bid": 155.0},
    )

    closed = trader.evaluate_and_auto_square_off()
    assert len(closed) == 1
    assert "PROFIT_TARGET_HIT" in closed[0]["exit_reason"]
    assert closed[0]["exit_price"] == 155.0
    assert closed[0]["gross_pnl"] == (155.0 - 100.0) * 25
    assert closed[0]["net_pnl"] > 0

    # Confirm open positions is now empty and recorded in trades ledger
    assert len(trader.journal.get_open_positions()) == 0
    trades_df = trader.journal.get_trades_dataframe()
    assert len(trades_df) == 1
    assert trades_df["symbol"].iloc[0] == "NIFTY_TEST_CALL"


def test_auto_trader_stop_loss_square_off(tmp_path):
    trader = AutoTrader(data_dir=tmp_path)
    trader.config["profit_target_pct"] = 50.0
    trader.config["stop_loss_pct"] = 30.0

    # Add open position with entry 100, target 150, SL 70
    pos_id = trader.journal.add_open_position(
        symbol="BANKNIFTY_TEST_PUT",
        underlying_symbol="BANKNIFTY",
        strike=53000.0,
        expiry=(datetime.now() + timedelta(days=5)).isoformat(),
        option_type="PE",
        n_lots=1,
        lot_size=15,
        entry_price=200.0,
        current_price=200.0,
        target_price=300.0,
        stop_loss_price=140.0,
    )
    assert pos_id > 0

    # Simulate price dropping to 135 (SL hit)
    trader.tick_cache.update_tick(
        "BANKNIFTY_TEST_PUT",
        {"symbol": "BANKNIFTY_TEST_PUT", "last_price": 135.0, "bid": 135.0},
    )

    closed = trader.evaluate_and_auto_square_off()
    assert len(closed) == 1
    assert "STOP_LOSS_HIT" in closed[0]["exit_reason"]
    assert closed[0]["exit_price"] == 135.0
    assert len(trader.journal.get_open_positions()) == 0
