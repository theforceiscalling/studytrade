"""
Tests for studytrade.logger — Logging configuration and custom TRADE level.
"""

from __future__ import annotations

import logging
from pathlib import Path

from studytrade.config import StudyTradeConfig
from studytrade.logger import TRADE_LEVEL, setup_logger


class TestLogger:
    """Test logger functionality."""

    def test_trade_level_exists(self):
        assert logging.getLevelName(TRADE_LEVEL) == "TRADE"
        assert logging.getLevelName("TRADE") == TRADE_LEVEL

    def test_setup_logger_basic(self):
        logger = setup_logger("test.basic", force_reinit=True)
        assert logger.name == "test.basic"
        assert hasattr(logger, "trade")

    def test_logger_file_output(self, tmp_path: Path):
        cfg_dict = {
            "logging": {
                "level": "DEBUG",
                "console": False,
                "file": True,
                "log_dir": str(tmp_path / "test_logs"),
                "format": "%(levelname)s:%(message)s",
                "date_format": "%H:%M:%S",
            }
        }
        cfg = StudyTradeConfig(**cfg_dict)
        logger = setup_logger("test.file", config=cfg, force_reinit=True)

        logger.info("Test info message")
        logger.trade("Buy 1 lot NIFTY 24000 CE @ 120.0")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()
            handler.close()

        log_dir = Path(tmp_path / "test_logs")
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) == 1

        content = log_files[0].read_text(encoding="utf-8")
        assert "INFO:Test info message" in content
        assert "TRADE:Buy 1 lot NIFTY 24000 CE @ 120.0" in content
