"""
StudyTrade — Persistent Backtest Results Database
=================================================
Stores every executed backtest run with strategy configuration,
performance metrics, and broker cost comparisons (Zerodha, Angel One, Stoxkart).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_BACKTEST_DB = str(PROJECT_ROOT / "data" / "backtest_runs.db")


def init_backtest_db(db_path: str = DEFAULT_BACKTEST_DB) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                win_rate_pct REAL NOT NULL,
                gross_pnl REAL NOT NULL,
                zerodha_charges REAL NOT NULL,
                zerodha_net_pnl REAL NOT NULL,
                angel_charges REAL NOT NULL,
                angel_net_pnl REAL NOT NULL,
                stoxkart_charges REAL NOT NULL,
                stoxkart_net_pnl REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                max_drawdown_pct REAL NOT NULL,
                params_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_backtest_run(
    symbol: str,
    strategy: str,
    initial_capital: float,
    total_trades: int,
    win_rate_pct: float,
    gross_pnl: float,
    broker_comp: dict[str, Any],
    sharpe_ratio: float,
    max_drawdown_pct: float,
    params: dict[str, Any],
    db_path: str = DEFAULT_BACKTEST_DB,
) -> int:
    init_backtest_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO backtest_runs (
                timestamp, symbol, strategy, initial_capital, total_trades,
                win_rate_pct, gross_pnl,
                zerodha_charges, zerodha_net_pnl,
                angel_charges, angel_net_pnl,
                stoxkart_charges, stoxkart_net_pnl,
                sharpe_ratio, max_drawdown_pct, params_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                symbol,
                strategy,
                initial_capital,
                total_trades,
                win_rate_pct,
                gross_pnl,
                broker_comp.get("zerodha", {}).get("total_charges", 0.0),
                broker_comp.get("zerodha", {}).get("net_pnl", 0.0),
                broker_comp.get("angel_one", {}).get("total_charges", 0.0),
                broker_comp.get("angel_one", {}).get("net_pnl", 0.0),
                broker_comp.get("stoxkart", {}).get("total_charges", 0.0),
                broker_comp.get("stoxkart", {}).get("net_pnl", 0.0),
                sharpe_ratio,
                max_drawdown_pct,
                json.dumps(params),
            ),
        )
        conn.commit()
        return cur.lastrowid or 0


def get_all_backtest_runs(db_path: str = DEFAULT_BACKTEST_DB) -> list[dict[str, Any]]:
    """Retrieve all saved backtest executions in reverse chronological order."""
    init_backtest_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, symbol, strategy, initial_capital, total_trades,
                   win_rate_pct, gross_pnl, zerodha_charges, zerodha_net_pnl,
                   angel_charges, angel_net_pnl, stoxkart_charges, stoxkart_net_pnl,
                   sharpe_ratio, max_drawdown_pct, params_json
            FROM backtest_runs
            ORDER BY id DESC
            """
        )
        rows = cur.fetchall()
        runs: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["params"] = json.loads(d.get("params_json", "{}"))
            except Exception:
                d["params"] = {}
            runs.append(d)
        return runs


def get_backtest_run_by_id(run_id: int, db_path: str = DEFAULT_BACKTEST_DB) -> dict[str, Any] | None:
    """Retrieve a single backtest run by its ID."""
    init_backtest_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, symbol, strategy, initial_capital, total_trades,
                   win_rate_pct, gross_pnl, zerodha_charges, zerodha_net_pnl,
                   angel_charges, angel_net_pnl, stoxkart_charges, stoxkart_net_pnl,
                   sharpe_ratio, max_drawdown_pct, params_json
            FROM backtest_runs
            WHERE id = ?
            """,
            (run_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["params"] = json.loads(d.get("params_json", "{}"))
        except Exception:
            d["params"] = {}
        return d
