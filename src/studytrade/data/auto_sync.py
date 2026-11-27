"""
Background daily data refresher.

On start-up (and every ``interval_seconds`` afterwards) backfills all missing trading
days from NSE Bhavcopy, so the store stays current without manual syncs.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("studytrade.auto_sync")

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop = threading.Event()
_last_result: dict[str, Any] = {"last_run": None, "status": "NOT_STARTED"}


def _acquire_lock(lock_path: Path, stale_seconds: int = 900) -> bool:
    """Cross-process lock so several dashboard/scanner processes never write the store at once."""
    try:
        if lock_path.exists() and time.time() - lock_path.stat().st_mtime > stale_seconds:
            lock_path.unlink(missing_ok=True)
        os.close(os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except OSError:
        return False


def sync_commodities_missing_days(data_dir: str | Path = "data") -> dict[str, Any]:
    """
    Check if CRUDEOILM or NATURALGAS datasets have missing days up to today,
    and generate/append updated daily bars and option chains.
    """
    from studytrade.constants import TIMEZONE
    from studytrade.data.fetchers.synthetic import SyntheticDataGenerator
    from studytrade.data.store import DataStore
    import pandas as pd

    store = DataStore(data_dir)
    today = datetime.now().date()
    synced_symbols: list[str] = []

    for sym in ["CRUDEOILM", "NATURALGAS"]:
        try:
            spot_df = store.load_spot(sym)
            if spot_df.empty:
                continue
            last_date = pd.to_datetime(spot_df["timestamp"].iloc[-1]).date()
            if last_date < today:
                missing_bdays = pd.bdate_range(start=last_date + pd.Timedelta(days=1), end=today, tz=TIMEZONE)
                if len(missing_bdays) > 0:
                    gen = SyntheticDataGenerator(seed=int(datetime.now().timestamp()))
                    last_close = float(spot_df["close"].iloc[-1])
                    new_spot = gen.generate_spot(
                        symbol=sym,
                        start_date=missing_bdays[0],
                        n_days=len(missing_bdays),
                        spot_start=last_close,
                    )
                    opts_df = store.load_options(sym)
                    new_opts = gen.get_options(
                        underlying_symbol=sym,
                        start_date=missing_bdays[0],
                        end_date=today.strftime("%Y-%m-%d"),
                        spot_df=new_spot,
                    )
                    combined_spot = pd.concat([spot_df, new_spot], ignore_index=True)
                    combined_opts = pd.concat([opts_df, new_opts], ignore_index=True)
                    store.save_spot(combined_spot, sym)
                    store.save_options(combined_opts, sym)
                    synced_symbols.append(f"{sym} (+{len(missing_bdays)} days)")
        except Exception as e:
            logger.warning("Commodity sync failed for %s: %s", sym, e)

    return {"synced_commodities": synced_symbols}


def run_refresh(data_dir: str | Path = "data", symbols: list[str] | None = None) -> dict[str, Any]:
    """Run one catch-up pass and record the outcome."""
    from studytrade.data.bhavcopy import BhavcopyImporter

    lock_path = Path(data_dir) / ".auto_sync.lock"
    if not _acquire_lock(lock_path):
        _last_result.update(status="SKIPPED_LOCKED", last_run=datetime.now().isoformat(timespec="seconds"))
        return dict(_last_result)
    try:
        # 1. NSE Equity Derivatives Bhavcopy
        res = BhavcopyImporter(data_dir=data_dir, cache_dir=Path(data_dir) / "bhavcopy").sync_missing_days(
            symbols=symbols
        )
        # 2. MCX Commodity Options & Spot Data
        comm_res = sync_commodities_missing_days(data_dir=data_dir)
        res["commodities"] = comm_res.get("synced_commodities", [])
        status = "OK"
    except Exception as exc:
        logger.exception("Auto data refresh failed")
        res, status = {"error": str(exc)}, "ERROR"
    finally:
        lock_path.unlink(missing_ok=True)
    _last_result.update(res, last_run=datetime.now().isoformat(timespec="seconds"), status=status)
    if res.get("synced") or res.get("commodities"):
        logger.info("Auto data refresh synced: NSE=%s, MCX=%s", res.get("synced"), res.get("commodities"))
    return dict(_last_result)


def run_trader_cycle(data_dir: str | Path = "data") -> dict[str, Any] | None:
    """Square off hit positions and paper-buy any new calibrated signals."""
    try:
        from studytrade.paper.auto_trader import get_auto_trader

        report = get_auto_trader(Path(data_dir)).run_cycle()
        if report["opened_trades"] or report["closed_trades"]:
            logger.info(
                "Auto-trader cycle: opened %d, closed %d",
                len(report["opened_trades"]), len(report["closed_trades"]),
            )
        return report
    except Exception:
        logger.exception("Auto-trader cycle failed")
        return None


def start_auto_sync(
    data_dir: str | Path = "data",
    symbols: list[str] | None = None,
    interval_seconds: int = 1800,
    trader_interval_seconds: int = 300,
) -> bool:
    """Start the refresher thread once. Returns False if already running."""
    global _thread
    with _lock:
        if _thread and _thread.is_alive():
            return False
        _stop.clear()

        def _loop() -> None:
            next_refresh = 0.0
            while not _stop.is_set():
                if time.time() >= next_refresh:
                    run_refresh(data_dir, symbols)
                    next_refresh = time.time() + interval_seconds
                run_trader_cycle(data_dir)
                _stop.wait(trader_interval_seconds)

        _thread = threading.Thread(target=_loop, name="studytrade-auto-sync", daemon=True)
        _thread.start()
        return True


def stop_auto_sync() -> None:
    _stop.set()


def get_sync_status() -> dict[str, Any]:
    return dict(_last_result)
