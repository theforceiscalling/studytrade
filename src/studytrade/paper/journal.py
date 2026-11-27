"""
StudyTrade — Paper Trading Journal and SQLite Persistence
===========================================================

Persists all paper trading orders, executions, and closed positions
into an embedded SQLite database (`data/paper_journal.db`).
Tracks cumulative paper duration for mandatory live trading safety gate.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


class PaperJournal:
    """
    Manages persistent SQLite journaling for forward paper-trading.
    """

    def __init__(self, db_path: str | Path = "data/paper_journal.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize tables for paper orders, trades, and equity snapshots."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    underlying_symbol TEXT NOT NULL,
                    strike REAL NOT NULL,
                    expiry TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    n_lots INTEGER NOT NULL,
                    lot_size INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    gross_pnl REAL NOT NULL,
                    net_pnl REAL NOT NULL,
                    total_fees REAL NOT NULL,
                    return_pct REAL NOT NULL,
                    holding_bars INTEGER NOT NULL,
                    exit_reason TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cash REAL NOT NULL,
                    total_equity REAL NOT NULL,
                    open_positions INTEGER NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_open_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    underlying_symbol TEXT NOT NULL,
                    strike REAL NOT NULL,
                    expiry TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    n_lots INTEGER NOT NULL,
                    lot_size INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    entry_iv REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    target_price REAL DEFAULT 0.0,
                    stop_loss_price REAL DEFAULT 0.0
                )
                """
            )
            # Safe schema migrations for existing databases
            for col_def in [
                ("target_price", "REAL DEFAULT 0.0"),
                ("stop_loss_price", "REAL DEFAULT 0.0"),
            ]:
                try:
                    cursor.execute(f"ALTER TABLE paper_open_positions ADD COLUMN {col_def[0]} {col_def[1]}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()
        finally:
            conn.close()

    def log_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        symbol: str,
        underlying_symbol: str,
        strike: float,
        expiry: datetime,
        option_type: str,
        n_lots: int,
        lot_size: int,
        entry_price: float,
        exit_price: float,
        gross_pnl: float,
        net_pnl: float,
        total_fees: float,
        return_pct: float,
        holding_bars: int,
        exit_reason: str,
    ) -> None:
        """Record completed round-trip paper trade."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO paper_trades (
                    entry_time, exit_time, symbol, underlying_symbol, strike, expiry,
                    option_type, n_lots, lot_size, entry_price, exit_price,
                    gross_pnl, net_pnl, total_fees, return_pct, holding_bars, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_time.isoformat(),
                    exit_time.isoformat(),
                    symbol,
                    underlying_symbol,
                    strike,
                    expiry.isoformat() if isinstance(expiry, datetime) else str(expiry),
                    option_type,
                    n_lots,
                    lot_size,
                    entry_price,
                    exit_price,
                    gross_pnl,
                    net_pnl,
                    total_fees,
                    return_pct,
                    holding_bars,
                    exit_reason,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def log_equity_snapshot(
        self,
        timestamp: datetime,
        cash: float,
        total_equity: float,
        open_positions: int,
    ) -> None:
        """Record equity timestamp snapshot."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO paper_equity (timestamp, cash, total_equity, open_positions)
                VALUES (?, ?, ?, ?)
                """,
                (timestamp.isoformat(), cash, total_equity, open_positions),
            )
            conn.commit()
        finally:
            conn.close()

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Retrieve all recorded paper trades."""
        conn = sqlite3.connect(self.db_path)
        try:
            return pd.read_sql_query("SELECT * FROM paper_trades ORDER BY exit_time ASC", conn)
        finally:
            conn.close()

    def add_open_position(
        self,
        symbol: str,
        underlying_symbol: str,
        strike: float,
        expiry: str | datetime,
        option_type: str,
        n_lots: int = 1,
        lot_size: int = 65,
        entry_price: float = 287.90,
        current_price: float = 287.90,
        entry_iv: float = 16.0,
        strategy: str = "Gamma Squeeze Manual",
        target_price: float = 0.0,
        stop_loss_price: float = 0.0,
        entry_time: datetime | None = None,
    ) -> int:
        """Add an active position to the open positions ledger."""
        now = (entry_time or datetime.now()).isoformat()
        exp_str = expiry.isoformat() if isinstance(expiry, datetime) else str(expiry)
        eff_tp = target_price if target_price > 0.0 else round(entry_price * 1.50, 2)
        eff_sl = stop_loss_price if stop_loss_price > 0.0 else round(entry_price * 0.70, 2)

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO paper_open_positions (
                    entry_time, symbol, underlying_symbol, strike, expiry, option_type,
                    n_lots, lot_size, entry_price, current_price, entry_iv, strategy,
                    target_price, stop_loss_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    symbol,
                    underlying_symbol,
                    strike,
                    exp_str,
                    option_type,
                    n_lots,
                    lot_size,
                    entry_price,
                    current_price,
                    entry_iv,
                    strategy,
                    eff_tp,
                    eff_sl,
                ),
            )
            conn.commit()
            return cur.lastrowid or 0
        finally:
            conn.close()

    def update_current_price(self, symbol: str, current_price: float) -> None:
        """Update live marked-to-market current price for an open position."""
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE paper_open_positions SET current_price = ? WHERE symbol = ?",
                (current_price, symbol),
            )
            conn.commit()
        finally:
            conn.close()

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Retrieve all currently active open paper positions."""
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, entry_time, symbol, underlying_symbol, strike, expiry,
                       option_type, n_lots, lot_size, entry_price, current_price, entry_iv,
                       strategy, target_price, stop_loss_price
                FROM paper_open_positions
                ORDER BY id ASC
                """
            )
            rows = cur.fetchall()
            cols = [
                "id",
                "entry_time",
                "symbol",
                "underlying_symbol",
                "strike",
                "expiry",
                "option_type",
                "n_lots",
                "lot_size",
                "entry_price",
                "current_price",
                "entry_iv",
                "strategy",
                "target_price",
                "stop_loss_price",
            ]
            positions = []
            for r in rows:
                p = dict(zip(cols, r))
                p["quantity"] = p["n_lots"]
                # Default target and SL if 0
                ep = float(p.get("entry_price", 1.0))
                cp = float(p.get("current_price", ep))
                tp = float(p.get("target_price") or (ep * 1.50))
                sl = float(p.get("stop_loss_price") or (ep * 0.70))
                p["target_price"] = round(tp, 2)
                p["stop_loss_price"] = round(sl, 2)
                p["pnl_pct"] = round(((cp - ep) / max(ep, 0.01)) * 100.0, 2)
                p["unrealized_pnl"] = round((cp - ep) * int(p.get("lot_size", 25)) * int(p.get("n_lots", 1)), 2)
                positions.append(p)
            return positions
        finally:
            conn.close()

    def remove_open_position(self, position_id: int | None = None, symbol: str | None = None) -> bool:
        """Remove a position from the open positions ledger."""
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            if position_id is not None:
                cur.execute("DELETE FROM paper_open_positions WHERE id = ?", (position_id,))
            elif symbol is not None:
                cur.execute("DELETE FROM paper_open_positions WHERE symbol = ?", (symbol,))
            else:
                return False
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def close_position(
        self,
        symbol: str | None = None,
        position_id: int | None = None,
        exit_price: float | None = None,
        exit_time: datetime | None = None,
        exit_reason: str = "MANUAL_SQUARE_OFF",
    ) -> dict[str, Any] | None:
        """
        Close an open position, compute statutory transaction charges and PnL,
        record the trade into paper_trades, and remove from open positions.
        """
        from studytrade.backtest.costs import compute_nse_option_costs

        open_positions = self.get_open_positions()
        target_pos = None
        for pos in open_positions:
            if position_id is not None and pos.get("id") == position_id:
                target_pos = pos
                break
            if symbol is not None and pos.get("symbol") == symbol:
                target_pos = pos
                break

        if not target_pos:
            return None

        now = exit_time or datetime.now()
        entry_time_dt = datetime.fromisoformat(target_pos["entry_time"])
        eff_exit_price = float(exit_price if exit_price is not None else target_pos.get("current_price", target_pos["entry_price"]))
        entry_price = float(target_pos["entry_price"])
        lot_size = int(target_pos.get("lot_size", 25))
        n_lots = int(target_pos.get("n_lots", 1))

        # Gross PnL
        gross_pnl = (eff_exit_price - entry_price) * lot_size * n_lots

        # Statutory charges
        cost_breakdown = compute_nse_option_costs(
            buy_price=entry_price,
            sell_price=eff_exit_price,
            lot_size=lot_size,
            n_lots=n_lots,
        )
        total_fees = float(cost_breakdown.total_statutory_fees)
        net_pnl = gross_pnl - total_fees
        return_pct = ((eff_exit_price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
        holding_bars = max(1, int((now - entry_time_dt).total_seconds() // 60))

        # Log into paper_trades
        self.log_trade(
            entry_time=entry_time_dt,
            exit_time=now,
            symbol=target_pos["symbol"],
            underlying_symbol=target_pos.get("underlying_symbol", "NIFTY"),
            strike=float(target_pos.get("strike", 0.0)),
            expiry=str(target_pos.get("expiry", "")),
            option_type=target_pos.get("option_type", "CE"),
            n_lots=n_lots,
            lot_size=lot_size,
            entry_price=entry_price,
            exit_price=eff_exit_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_fees=total_fees,
            return_pct=return_pct,
            holding_bars=holding_bars,
            exit_reason=exit_reason,
        )

        # Remove from open positions
        self.remove_open_position(position_id=target_pos.get("id"), symbol=target_pos.get("symbol"))

        return {
            "symbol": target_pos["symbol"],
            "entry_price": entry_price,
            "exit_price": eff_exit_price,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "total_fees": round(total_fees, 2),
            "return_pct": round(return_pct, 2),
            "exit_reason": exit_reason,
        }

    def get_equity_dataframe(self) -> pd.DataFrame:
        """Retrieve equity curve history."""
        conn = sqlite3.connect(self.db_path)
        try:
            return pd.read_sql_query("SELECT * FROM paper_equity ORDER BY timestamp ASC", conn)
        finally:
            conn.close()

    def get_paper_duration_days(self) -> int:
        """
        Calculate total elapsed calendar days since the earliest recorded paper trade.
        Used to enforce the mandatory 3-month (90-day) paper validation gate.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(entry_time) FROM paper_trades")
            row = cursor.fetchone()
            if row is None or row[0] is None:
                return 0

            earliest_dt = datetime.fromisoformat(row[0])
            elapsed_days = (datetime.now() - earliest_dt).days
            return max(0, elapsed_days)
        finally:
            conn.close()
