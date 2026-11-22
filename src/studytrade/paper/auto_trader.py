"""
StudyTrade — Autonomous Auto-Execution and Auto-Square-Off Engine
================================================================

Autonomous execution system for forward paper-trading:
- Auto-buys high-conviction Gamma Squeeze & Dealer Cascade signals within ₹20,000 risk limits.
- Continuously monitors open positions against hard Stop-Loss (e.g. -30%), asymmetric
  Profit Target (e.g. +50%), DTE expiration cutoffs, and intraday 15:20 IST auto-squareoff rules.
- Fully integrated with SQLite ledger (`data/paper_journal.db`) and statutory transaction cost engine.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any

import pandas as pd

from studytrade.constants import INSTRUMENTS, OptionType
from studytrade.live.session_engine import get_live_tick_cache
from studytrade.math.bsm import bsm_price
from studytrade.paper.journal import PaperJournal

logger = logging.getLogger("studytrade.auto_trader")

CONFIG_PATH = Path("data/auto_trader_config.json")

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "auto_buy": True,
    "auto_squareoff": True,
    "profit_target_pct": 50.0,
    "stop_loss_pct": 30.0,
    "target_delta": 0.35,
    "min_confidence_pct": 60.0,
    "max_capital_per_trade": 5000.0,
    "total_capital_limit": 20000.0,
    "max_open_positions": 3,
    "auto_squareoff_time_ist": "15:20",
}


class AutoTrader:
    """
    Coordinates automatic execution and rules-based auto-square-off.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.config_path = self.data_dir / "auto_trader_config.json"
        self.journal = PaperJournal(self.data_dir / "paper_journal.db")
        self.tick_cache = get_live_tick_cache()
        self.config = self.load_config()

    def load_config(self) -> dict[str, Any]:
        """Load configuration from JSON or initialize defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    merged = DEFAULT_CONFIG.copy()
                    merged.update(cfg)
                    return merged
            except Exception as e:
                logger.warning("Could not read auto trader config: %s", e)
        return DEFAULT_CONFIG.copy()

    def save_config(self, new_config: dict[str, Any]) -> dict[str, Any]:
        """Update and persist configuration."""
        self.config.update(new_config)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
        return self.config

    def toggle_master(self) -> bool:
        """Toggle master enabled state."""
        self.config["enabled"] = not self.config.get("enabled", True)
        self.save_config(self.config)
        return bool(self.config["enabled"])

    def evaluate_and_auto_square_off(self) -> list[dict[str, Any]]:
        """
        Check all open positions against Profit Target, Stop Loss, DTE expiry,
        and intraday time stops. Squares off any triggering position automatically.
        """
        if not self.config.get("enabled", True) or not self.config.get("auto_squareoff", True):
            return []

        open_positions = self.journal.get_open_positions()
        if not open_positions:
            return []

        now = datetime.now()
        cur_time = now.time()
        sq_hour, sq_min = [int(x) for x in self.config.get("auto_squareoff_time_ist", "15:20").split(":")]
        sq_time_limit = dtime(sq_hour, sq_min)

        target_pct = float(self.config.get("profit_target_pct", 50.0))
        sl_pct = float(self.config.get("stop_loss_pct", 30.0))

        closed_trades: list[dict[str, Any]] = []

        for pos in open_positions:
            sym = pos["symbol"]
            entry_price = float(pos.get("entry_price", 0.0))
            if entry_price <= 0.0:
                continue

            target_price = float(pos.get("target_price") or (entry_price * (1.0 + target_pct / 100.0)))
            stop_loss_price = float(pos.get("stop_loss_price") or (entry_price * (1.0 - sl_pct / 100.0)))

            # 1. Determine Current Market Price (Bid/LTP)
            current_price = entry_price
            db_price = float(pos.get("current_price", 0.0))
            tick = self.tick_cache.get_tick(sym)
            if tick and "last_price" in tick and tick.get("is_fresh", False):
                current_price = float(tick.get("bid", tick["last_price"]))
                self.journal.update_current_price(sym, current_price)
            elif db_price > 0.0 and abs(db_price - entry_price) > 1e-4:
                current_price = db_price
            elif tick and "last_price" in tick:
                current_price = float(tick.get("bid", tick["last_price"]))
            else:
                current_price = entry_price

            pnl_pct = ((current_price - entry_price) / entry_price) * 100.0

            # 2. Check Expiry and Time Stops
            exp_date = None
            if "expiry" in pos and pos["expiry"]:
                try:
                    exp_date = pd.to_datetime(pos["expiry"]).date()
                except Exception:
                    pass

            is_expired = False
            is_intraday_cutoff = False
            if exp_date:
                today = now.date()
                if today > exp_date:
                    is_expired = True
                elif today == exp_date and cur_time >= sq_time_limit:
                    is_intraday_cutoff = True

            should_close = False
            exit_reason = "MANUAL_SQUARE_OFF"

            # 3. Evaluate Exit Conditions
            if current_price >= target_price or pnl_pct >= target_pct:
                should_close = True
                exit_reason = f"PROFIT_TARGET_HIT (+{pnl_pct:.1f}%)"
            elif current_price <= stop_loss_price or pnl_pct <= -sl_pct:
                should_close = True
                exit_reason = f"STOP_LOSS_HIT ({pnl_pct:.1f}%)"
            elif is_expired:
                should_close = True
                exit_reason = "EXPIRY_SETTLEMENT"
            elif is_intraday_cutoff:
                should_close = True
                exit_reason = "INTRADAY_AUTO_SQUAREOFF"

            if should_close:
                res = self.journal.close_position(
                    position_id=pos.get("id"),
                    symbol=sym,
                    exit_price=current_price,
                    exit_time=now,
                    exit_reason=exit_reason,
                )
                if res:
                    logger.info("Auto-Squared Off %s: %s | Net PnL: ₹%s", sym, exit_reason, res.get("net_pnl"))
                    closed_trades.append(res)

        return closed_trades

    def evaluate_and_auto_buy(self) -> list[dict[str, Any]]:
        """
        Evaluate live market Strategy Signals and automatically buy eligible
        high-conviction contracts within capital and risk limits.
        """
        if not self.config.get("enabled", True) or not self.config.get("auto_buy", True):
            return []

        open_positions = self.journal.get_open_positions()
        max_pos = int(self.config.get("max_open_positions", 3))
        if len(open_positions) >= max_pos:
            return []

        # Check deployed capital against total ₹20,000 capital limit
        total_cap_limit = float(self.config.get("total_capital_limit", 20000.0))
        max_trade_cap = float(self.config.get("max_capital_per_trade", 5000.0))
        deployed_capital = sum(
            float(p.get("entry_price", 0.0)) * int(p.get("lot_size", 25)) * int(p.get("n_lots", 1))
            for p in open_positions
        )
        available_capital = max(0.0, total_cap_limit - deployed_capital)
        if available_capital < 500.0:
            return []

        existing_underlyings = {p.get("underlying_symbol", "") for p in open_positions}
        existing_symbols = {p.get("symbol", "") for p in open_positions}

        min_conf = float(self.config.get("min_confidence_pct", 60.0))
        target_delta = float(self.config.get("target_delta", 0.35))
        tp_pct = float(self.config.get("profit_target_pct", 50.0))
        sl_pct = float(self.config.get("stop_loss_pct", 30.0))

        opened_trades: list[dict[str, Any]] = []

        from studytrade.data.store import DataStore
        from studytrade.strategy.signals.gamma_squeeze import GammaSqueezeStrategy

        try:
            from dashboard.views import _get_manual_trade_recommendation
        except ImportError:
            from web.dashboard.views import _get_manual_trade_recommendation

        store = DataStore(self.data_dir)

        for sym in ["NIFTY", "BANKNIFTY"]:
            # One active position per underlying max to diversify
            if sym in existing_underlyings:
                continue

            try:
                spot_df = store.load_spot(sym)
                opt_df = store.load_options(sym)
                if spot_df.empty or opt_df.empty:
                    continue

                latest_ts = opt_df["timestamp"].max()
                opt_slice = opt_df[opt_df["timestamp"] == latest_ts]
                spot_slice = spot_df.tail(10)
                spot_close = float(spot_df["close"].iloc[-1])

                strat = GammaSqueezeStrategy(
                    target_delta=target_delta,
                    squeeze_score_threshold=min_conf,
                    allow_bearish=True,
                )
                raw_sigs = strat.generate_signals(spot_slice, opt_slice)
                if not raw_sigs:
                    continue

                best_sig = raw_sigs[0]
                rec = _get_manual_trade_recommendation(
                    opt_slice,
                    best_sig.direction.value,
                    spot_close,
                    target_delta=best_sig.target_delta,
                    profit_target_pct=tp_pct,
                    stop_loss_pct=sl_pct,
                    symbol=sym,
                )

                if not rec or not rec.get("contract_symbol"):
                    continue

                c_sym = rec["contract_symbol"]
                if c_sym in existing_symbols:
                    continue

                req_cap = float(rec.get("capital_per_lot", 2500.0))
                if req_cap > max_trade_cap or req_cap > available_capital:
                    continue

                entry_price = float(rec.get("ask", rec.get("premium", 50.0)))
                lot_size = int(rec.get("lot_size", 25))
                target_price = float(rec.get("target_price", round(entry_price * (1 + tp_pct / 100.0), 2)))
                stop_loss_price = float(rec.get("stop_loss_price", round(entry_price * (1 - sl_pct / 100.0), 2)))

                pos_id = self.journal.add_open_position(
                    symbol=c_sym,
                    underlying_symbol=sym,
                    strike=float(rec.get("strike", 0.0)),
                    expiry=str(opt_slice["expiry"].iloc[0])[:10] if "expiry" in opt_slice.columns else "",
                    option_type="CE" if "CALL" in best_sig.direction.value else "PE",
                    n_lots=1,
                    lot_size=lot_size,
                    entry_price=entry_price,
                    current_price=entry_price,
                    entry_iv=16.0,
                    strategy="Auto-Trader (Gamma Squeeze)",
                    target_price=target_price,
                    stop_loss_price=stop_loss_price,
                )

                opened = {
                    "id": pos_id,
                    "symbol": c_sym,
                    "underlying_symbol": sym,
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "stop_loss_price": stop_loss_price,
                    "lot_size": lot_size,
                    "confidence": round(best_sig.strength * 100, 1),
                    "capital_deployed": req_cap,
                }
                opened_trades.append(opened)
                deployed_capital += req_cap
                available_capital -= req_cap
                existing_underlyings.add(sym)
                existing_symbols.add(c_sym)

                logger.info("Auto-Trader Executed Buy: %s @ ₹%s | TP: ₹%s | SL: ₹%s", c_sym, entry_price, target_price, stop_loss_price)

                if len(open_positions) + len(opened_trades) >= max_pos:
                    break
            except Exception as e:
                logger.error("Error in auto-buy for %s: %s", sym, e)

        return opened_trades

    def run_cycle(self) -> dict[str, Any]:
        """
        Execute one complete autonomous cycle:
        1. Sync delayed quotes from official NSE option chain.
        2. Evaluate exits and auto-square-off hit positions.
        3. Evaluate entries and auto-buy eligible signals.
        """
        try:
            from studytrade.data.fetchers.nse_delayed import get_nse_delayed_fetcher
            get_nse_delayed_fetcher().sync_to_tick_cache(journal=self.journal)
        except Exception as e:
            logger.warning("NSE delayed sync error in run_cycle: %s", e)

        closed = self.evaluate_and_auto_square_off()
        opened = self.evaluate_and_auto_buy()

        open_positions = self.journal.get_open_positions()
        deployed = sum(
            float(p.get("entry_price", 0.0)) * int(p.get("lot_size", 25)) * int(p.get("n_lots", 1))
            for p in open_positions
        )
        total_limit = float(self.config.get("total_capital_limit", 20000.0))

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "RUNNING" if self.config.get("enabled", True) else "PAUSED",
            "closed_trades": closed,
            "opened_trades": opened,
            "open_positions_count": len(open_positions),
            "deployed_capital": round(deployed, 2),
            "available_capital": round(max(0.0, total_limit - deployed), 2),
            "config": self.config,
        }


_auto_trader_instance: AutoTrader | None = None


def get_auto_trader(data_dir: str | Path = "data") -> AutoTrader:
    """Singleton getter for AutoTrader."""
    global _auto_trader_instance
    if _auto_trader_instance is None:
        _auto_trader_instance = AutoTrader(data_dir=data_dir)
    return _auto_trader_instance
