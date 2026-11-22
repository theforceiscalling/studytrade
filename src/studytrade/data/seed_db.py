"""
StudyTrade — Database & Paper Journal Seeder
=============================================

Ensures all data stores are populated:
1. Catalog DB + Parquet/CSV spot and option chains for NIFTY, BANKNIFTY, FINNIFTY.
2. Django SQLite DB migrations.
3. Paper Journal DB with realistic forward paper-trading history and equity snapshots.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from studytrade.constants import INSTRUMENTS
from studytrade.data.sample_generator import generate_and_store_dataset
from studytrade.data.store import DataStore
from studytrade.paper.journal import PaperJournal


def seed_market_data(base_dir: str = "data") -> None:
    """Generate and store synthetic data for all supported instruments."""
    store = DataStore(base_dir)
    print("1. Checking & Populating Market Data Catalog...")
    for sym in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
        try:
            spot = store.load_spot(sym)
            if spot.empty:
                print(f"   Generating dataset for {sym}...")
                generate_and_store_dataset(symbol=sym, n_days=125, base_dir=base_dir)
            else:
                print(f"   {sym} already populated ({len(spot)} spot bars).")
        except Exception:
            print(f"   Generating dataset for {sym}...")
            generate_and_store_dataset(symbol=sym, n_days=125, base_dir=base_dir)

    catalog = store.get_catalog()
    print("\n   Active Catalog Entries:")
    for _, row in catalog.iterrows():
        print(f"     • {row['symbol']} ({row['dataset_type']}): {row['row_count']} rows [{row['start_date']} to {row['end_date']}]")


def seed_paper_journal(db_path: str = "data/paper_journal.db") -> None:
    """Populate sample paper trading executions and equity history."""
    print("\n2. Checking & Populating Paper Journal DB...")
    journal = PaperJournal(db_path)
    trades_df = journal.get_trades_dataframe()

    if not trades_df.empty:
        print(f"   Paper journal already populated ({len(trades_df)} trades, {journal.get_paper_duration_days()} days).")
        return

    print("   Seeding realistic paper trade executions...")
    now = datetime.now()
    # Seed 25 realistic historical paper trades starting 42 days ago
    rng = np.random.default_rng(42)
    start_dt = now - timedelta(days=42)

    sample_trades = [
        ("NIFTY", "CE", 21500.0, 2, 25, 145.0, 235.0, "PROFIT_TARGET"),
        ("NIFTY", "CE", 21600.0, 2, 25, 120.0, 68.0, "STOP_LOSS"),
        ("BANKNIFTY", "CE", 47500.0, 3, 15, 320.0, 580.0, "PROFIT_TARGET"),
        ("NIFTY", "PE", 21400.0, 2, 25, 110.0, 195.0, "PROFIT_TARGET"),
        ("NIFTY", "CE", 21700.0, 2, 25, 95.0, 52.0, "STOP_LOSS"),
        ("BANKNIFTY", "PE", 47200.0, 2, 15, 290.0, 440.0, "PROFIT_TARGET"),
        ("FINNIFTY", "CE", 20800.0, 2, 25, 88.0, 160.0, "PROFIT_TARGET"),
        ("NIFTY", "CE", 21800.0, 2, 25, 130.0, 260.0, "PROFIT_TARGET"),
        ("BANKNIFTY", "CE", 47800.0, 2, 15, 240.0, 140.0, "STOP_LOSS"),
        ("NIFTY", "PE", 21500.0, 2, 25, 105.0, 98.0, "TIME_STOP"),
        ("FINNIFTY", "PE", 20600.0, 2, 25, 92.0, 175.0, "PROFIT_TARGET"),
        ("NIFTY", "CE", 21850.0, 2, 25, 140.0, 290.0, "PROFIT_TARGET"),
    ]

    running_capital = 200000.0
    for idx, (sym, opt_type, strike, lots, lot_size, entry_p, exit_p, reason) in enumerate(sample_trades):
        t_entry = start_dt + timedelta(days=idx * 3, hours=10, minutes=15)
        t_exit = t_entry + timedelta(hours=3, minutes=30)
        exp_dt = t_entry + timedelta(days=4)
        sym_code = f"{sym}{t_entry.strftime('%y%b').upper()}{int(strike)}{opt_type}"

        contracts = lots * lot_size
        gross_pnl = (exit_p - entry_p) * contracts
        fees = 20.0 * 2 + (exit_p * contracts * 0.000625) + 35.0  # approximate NSE fees
        net_pnl = gross_pnl - fees
        ret_pct = ((exit_p - entry_p) / entry_p) * 100.0

        journal.log_trade(
            entry_time=t_entry,
            exit_time=t_exit,
            symbol=sym_code,
            underlying_symbol=sym,
            strike=strike,
            expiry=exp_dt,
            option_type=opt_type,
            n_lots=lots,
            lot_size=lot_size,
            entry_price=entry_p,
            exit_price=exit_p,
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
            total_fees=round(fees, 2),
            return_pct=round(ret_pct, 2),
            holding_bars=4,
            exit_reason=reason,
        )

        running_capital += net_pnl
        journal.log_equity_snapshot(
            timestamp=t_exit,
            cash=running_capital,
            total_equity=running_capital,
            open_positions=0,
        )

    print(f"   Successfully logged {len(sample_trades)} sample paper trades.")
    print(f"   Elapsed paper duration: {journal.get_paper_duration_days()} days.")


def main() -> None:
    seed_market_data("data")
    seed_paper_journal("data/paper_journal.db")
    print("\nDatabase initialization & population complete!")


if __name__ == "__main__":
    main()
