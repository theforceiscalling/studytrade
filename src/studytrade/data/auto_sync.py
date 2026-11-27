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


def run_refresh(data_dir: str | Path = "data", symbols: list[str] | None = None) -> dict[str, Any]:
    """Run one catch-up pass and record the outcome."""
    from studytrade.data.bhavcopy import BhavcopyImporter

    lock_path = Path(data_dir) / ".auto_sync.lock"
    if not _acquire_lock(lock_path):
        _last_result.update(status="SKIPPED_LOCKED", last_run=datetime.now().isoformat(timespec="seconds"))
        return dict(_last_result)
    try:
        res = BhavcopyImporter(data_dir=data_dir, cache_dir=Path(data_dir) / "bhavcopy").sync_missing_days(
            symbols=symbols
        )
        status = "OK"
    except Exception as exc:
        logger.exception("Auto data refresh failed")
        res, status = {"error": str(exc)}, "ERROR"
    finally:
        lock_path.unlink(missing_ok=True)
    _last_result.update(res, last_run=datetime.now().isoformat(timespec="seconds"), status=status)
    if res.get("synced"):
        logger.info("Auto data refresh synced: %s", res["synced"])
    return dict(_last_result)


def start_auto_sync(
    data_dir: str | Path = "data",
    symbols: list[str] | None = None,
    interval_seconds: int = 1800,
) -> bool:
    """Start the refresher thread once. Returns False if already running."""
    global _thread
    with _lock:
        if _thread and _thread.is_alive():
            return False
        _stop.clear()

        def _loop() -> None:
            while not _stop.is_set():
                run_refresh(data_dir, symbols)
                _stop.wait(interval_seconds)

        _thread = threading.Thread(target=_loop, name="studytrade-auto-sync", daemon=True)
        _thread.start()
        return True


def stop_auto_sync() -> None:
    _stop.set()


def get_sync_status() -> dict[str, Any]:
    return dict(_last_result)
