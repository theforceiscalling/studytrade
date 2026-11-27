"""
Calibrated signal log.

Every distinct Gamma Squeeze signal produced with the calibrated parameters is stored once
(keyed by underlying + data date + direction), together with what the auto-trader did and,
when a paper position was opened, the resulting profit / loss.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from studytrade.paper.journal import PaperJournal


class SignalLog:
    def __init__(self, db_path: str | Path = "data/paper_journal.db") -> None:
        self.db_path = str(db_path)
        # PaperJournal creates the trades / positions tables this log joins against
        PaperJournal(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calibrated_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    data_date TEXT NOT NULL,
                    underlying TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence_pct REAL NOT NULL,
                    spot REAL NOT NULL,
                    contract TEXT,
                    strike REAL,
                    expiry TEXT,
                    entry_price REAL,
                    target_price REAL,
                    stop_loss_price REAL,
                    lot_size INTEGER,
                    capital_required REAL,
                    reason TEXT,
                    params_json TEXT,
                    action TEXT NOT NULL DEFAULT 'LOGGED',
                    action_detail TEXT,
                    position_id INTEGER,
                    position_entry_time TEXT
                )
                """
            )

    @staticmethod
    def make_key(underlying: str, data_date: str, direction: str) -> str:
        return f"{underlying}|{data_date}|{direction}"

    def record_if_new(self, entry: dict[str, Any]) -> int | None:
        """Insert the signal; returns its id, or None if this signal was already recorded."""
        entry = {**entry, "created_at": datetime.now().isoformat(timespec="seconds")}
        if isinstance(entry.get("params_json"), dict):
            entry["params_json"] = json.dumps(entry["params_json"])
        cols = ", ".join(entry)
        marks = ", ".join("?" for _ in entry)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(f"INSERT OR IGNORE INTO calibrated_signals ({cols}) VALUES ({marks})", list(entry.values()))
            return cur.lastrowid if cur.rowcount else None

    def set_action(
        self,
        signal_id: int,
        action: str,
        detail: str = "",
        position_id: int | None = None,
        position_entry_time: str | None = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE calibrated_signals SET action=?, action_detail=?, position_id=?, position_entry_time=? WHERE id=?",
                (action, detail, position_id, position_entry_time, signal_id),
            )

    def list_with_pnl(self, limit: int = 200) -> dict[str, Any]:
        """Signals (newest first) with P&L of the paper trade each one produced, plus totals."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            signals = [dict(r) for r in conn.execute(
                "SELECT * FROM calibrated_signals ORDER BY id DESC LIMIT ?", (limit,)
            )]
            open_pos = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM paper_open_positions")}
            for s in signals:
                s["status"] = "NOT_TRADED"
                s["pnl"] = None
                s["pnl_pct"] = None
                if s["action"] != "EXECUTED":
                    continue
                pos = open_pos.get(s["position_id"])
                if pos and pos["entry_time"] == s["position_entry_time"]:
                    ep, cp = float(pos["entry_price"]), float(pos["current_price"])
                    s["status"] = "OPEN"
                    s["pnl"] = round((cp - ep) * int(pos["lot_size"]) * int(pos["n_lots"]), 2)
                    s["pnl_pct"] = round((cp - ep) / ep * 100.0, 2) if ep else 0.0
                    continue
                trade = conn.execute(
                    "SELECT net_pnl, return_pct, exit_reason FROM paper_trades "
                    "WHERE symbol=? AND entry_time=? ORDER BY id DESC LIMIT 1",
                    (s["contract"], s["position_entry_time"]),
                ).fetchone()
                if trade:
                    s["status"] = "CLOSED"
                    s["pnl"] = round(float(trade["net_pnl"]), 2)
                    s["pnl_pct"] = round(float(trade["return_pct"]), 2)
                    s["exit_reason"] = trade["exit_reason"]

        closed = [s for s in signals if s["status"] == "CLOSED"]
        open_ = [s for s in signals if s["status"] == "OPEN"]
        realized = sum(s["pnl"] for s in closed)
        unrealized = sum(s["pnl"] for s in open_)
        wins = sum(1 for s in closed if s["pnl"] > 0)
        summary = {
            "total_signals": len(signals),
            "executed": len(closed) + len(open_),
            "open": len(open_),
            "closed": len(closed),
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl": round(realized + unrealized, 2),
            "win_rate_pct": round(wins / len(closed) * 100.0, 1) if closed else 0.0,
        }
        return {"signals": signals, "summary": summary}
