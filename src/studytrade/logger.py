"""
StudyTrade — Structured Logging
=================================

Design decisions:
  1. Use stdlib `logging` — no external dependencies, integrates everywhere.
  2. Dual output: console (colored via rich) + rotating file.
  3. Each module gets its own named logger: `studytrade.data`, `studytrade.math`, etc.
  4. Log level is set from config, overridable by LOG_LEVEL env var.
  5. Trade-critical events (entries, exits, fills) use a dedicated "TRADE" level.

Usage:
    from studytrade.logger import setup_logger
    logger = setup_logger("studytrade.backtest")
    logger.info("Backtest started", extra={"symbol": "NIFTY", "date": "2024-01-15"})
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from studytrade.config import StudyTradeConfig, find_project_root

# ============================================================
# Custom log level for trade events
# ============================================================

TRADE_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(TRADE_LEVEL, "TRADE")


def trade(self: logging.Logger, message: str, *args: object, **kwargs: object) -> None:
    """Log a trade-critical event (entry, exit, fill)."""
    if self.isEnabledFor(TRADE_LEVEL):
        self._log(TRADE_LEVEL, message, args, **kwargs)


# Monkey-patch Logger class to add .trade() method
logging.Logger.trade = trade  # type: ignore[attr-defined]


# ============================================================
# Formatter with optional colors for console
# ============================================================


class ColorFormatter(logging.Formatter):
    """Add ANSI colors to log levels for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "TRADE": "\033[35m",  # Magenta
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


# ============================================================
# Logger setup
# ============================================================

_initialized: set[str] = set()


def setup_logger(
    name: str = "studytrade",
    config: StudyTradeConfig | None = None,
    *,
    force_reinit: bool = False,
) -> logging.Logger:
    """
    Create and configure a named logger.

    Parameters
    ----------
    name : str
        Logger name (e.g., "studytrade.data.fetcher").
    config : StudyTradeConfig, optional
        Config object. If None, uses defaults.
    force_reinit : bool
        If True, reconfigure even if already initialized.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if name in _initialized and not force_reinit:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.handlers.clear()

    # Determine settings from config or use defaults
    if config is not None:
        log_cfg = config.logging
    else:
        # Minimal defaults if no config loaded yet
        from types import SimpleNamespace

        log_cfg = SimpleNamespace(
            level="INFO",
            console=True,
            file=False,
            log_dir="logs",
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            date_format="%Y-%m-%d %H:%M:%S",
        )

    logger.setLevel(getattr(logging, log_cfg.level.upper(), logging.INFO))

    # Console handler
    if log_cfg.console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColorFormatter(log_cfg.format, datefmt=log_cfg.date_format))
        logger.addHandler(console_handler)

    # File handler
    if log_cfg.file:
        root = find_project_root()
        log_dir = root / log_cfg.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"studytrade_{today}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_cfg.format, datefmt=log_cfg.date_format))
        logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoid duplicate messages)
    logger.propagate = False

    _initialized.add(name)
    return logger
