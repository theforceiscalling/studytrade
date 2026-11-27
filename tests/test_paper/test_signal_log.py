from __future__ import annotations

import sys
import types
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from studytrade.paper.auto_trader import AutoTrader
from studytrade.paper.signal_log import SignalLog
from studytrade.strategy.base import SignalDirection, TradeSignal


class _FakeStore:
    def __init__(self, *_a, **_k):
        pass

    def load_spot(self, sym):
        return pd.DataFrame({"close": [25000.0, 25100.0]})

    def load_options(self, sym):
        return pd.DataFrame({
            "timestamp": [pd.Timestamp("2026-11-26", tz="Asia/Kolkata")],
            "expiry": [pd.Timestamp("2026-12-29", tz="Asia/Kolkata")],
        })


def _rec(contract: str, ask: float, lot: int = 65):
    def fn(chain, direction, spot, **kw):
        tp, sl = kw["profit_target_pct"], kw["stop_loss_pct"]
        return {
            "contract_symbol": contract, "strike": 25100.0, "ask": ask, "premium": ask, "lot_size": lot,
            "capital_per_lot": ask * lot, "target_price": ask * (1 + tp / 100), "stop_loss_price": ask * (1 - sl / 100),
        }
    return fn


def _cycle(trader, rec_fn):
    sig = TradeSignal(datetime.now(), "NIFTY", SignalDirection.BUY_CALL, 0.7, "test")
    fake_views = types.SimpleNamespace(_get_manual_trade_recommendation=rec_fn)
    with patch("studytrade.data.store.DataStore", _FakeStore), \
         patch("studytrade.strategy.signals.gamma_squeeze.GammaSqueezeStrategy.generate_signals", return_value=[sig]), \
         patch.dict(sys.modules, {"dashboard": types.ModuleType("dashboard"), "dashboard.views": fake_views}):
        return trader.evaluate_and_auto_buy()


def test_new_signal_is_logged_traded_once_and_pnl_tracked(tmp_path):
    trader = AutoTrader(data_dir=tmp_path)
    trader.save_config({"max_capital_per_trade": 50000.0, "total_capital_limit": 100000.0})

    opened = _cycle(trader, _rec("NIFTY26DEC25100CE", 100.0))
    assert len(opened) >= 1

    # Same signal on the next cycle must not be re-recorded or re-bought
    again = _cycle(trader, _rec("NIFTY26DEC25100CE", 100.0))
    assert again == []
    log = SignalLog(trader.journal.db_path).list_with_pnl()
    nifty = [s for s in log["signals"] if s["underlying"] == "NIFTY"]
    assert len(nifty) == 1 and nifty[0]["action"] == "EXECUTED" and nifty[0]["status"] == "OPEN"

    # Mark-to-market profit, then close: realized P&L (net of fees) appears against the signal
    trader.journal.update_current_price("NIFTY26DEC25100CE", 120.0)
    assert SignalLog(trader.journal.db_path).list_with_pnl()["summary"]["unrealized_pnl"] == 20.0 * 65 * len(opened)
    trader.journal.close_position(symbol="NIFTY26DEC25100CE", exit_price=130.0)
    log = SignalLog(trader.journal.db_path).list_with_pnl()
    closed = next(s for s in log["signals"] if s["underlying"] == "NIFTY")
    assert closed["status"] == "CLOSED" and 0 < closed["pnl"] < 30.0 * 65
    assert log["summary"]["realized_pnl"] == closed["pnl"]

    # Closed position is not re-opened by the old signal
    assert _cycle(trader, _rec("NIFTY26DEC25100CE", 100.0)) == []


def test_signal_skipped_for_capital_is_still_recorded(tmp_path):
    trader = AutoTrader(data_dir=tmp_path)  # default per-trade cap ₹5,000
    assert _cycle(trader, _rec("NIFTY26DEC25100CE", 105.8)) == []
    sig = SignalLog(trader.journal.db_path).list_with_pnl()["signals"][0]
    assert sig["action"] == "SKIPPED_CAPITAL_LIMIT" and sig["pnl"] is None
