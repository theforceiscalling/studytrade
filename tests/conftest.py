"""
StudyTrade — Shared Test Fixtures
====================================

Pytest fixtures available to ALL test modules.
Put module-specific fixtures in their own conftest.py files.

Usage:
    def test_something(sample_config, tmp_data_dir):
        assert sample_config.capital.initial == 500000
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from studytrade.config import StudyTradeConfig, load_config, reset_config

# ============================================================
# Config fixtures
# ============================================================


@pytest.fixture
def default_config_data() -> dict:
    """Raw dict of default configuration values."""
    return {
        "project": {"name": "studytrade_test", "version": "0.1.0", "environment": "development"},
        "capital": {"initial": 500000, "currency": "INR"},
        "risk": {
            "max_risk_per_trade_pct": 1.5,
            "max_daily_loss_pct": 5.0,
            "max_weekly_loss_pct": 10.0,
            "max_concurrent_positions": 3,
            "kill_switch_loss_pct": 15.0,
        },
        "costs": {
            "brokerage_per_order": 20.0,
            "stt_sell_pct": 0.0625,
            "exchange_txn_pct": 0.0495,
            "sebi_per_crore": 10.0,
            "gst_pct": 18.0,
            "stamp_duty_buy_pct": 0.003,
            "slippage_ticks": 1.0,
            "use_bid_ask": True,
        },
        "data": {"source": "synthetic", "storage_format": "parquet"},
        "logging": {"level": "DEBUG", "console": False, "file": False},
    }


@pytest.fixture
def sample_config(default_config_data: dict) -> StudyTradeConfig:
    """A validated StudyTradeConfig for testing."""
    return StudyTradeConfig(**default_config_data)


@pytest.fixture
def config_with_yaml(tmp_path: Path, default_config_data: dict) -> StudyTradeConfig:
    """
    Create a temporary config directory with default.yaml and load from it.
    Useful for testing the full load_config() flow.
    """
    # Create a minimal pyproject.toml so find_project_root works
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    with open(config_dir / "default.yaml", "w") as f:
        yaml.dump(default_config_data, f)

    return load_config(project_root=tmp_path)


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before each test to avoid state leaks."""
    reset_config()
    yield
    reset_config()


# ============================================================
# Data fixtures
# ============================================================


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory with standard subdirs."""
    for subdir in ["raw", "processed", "synthetic"]:
        (tmp_path / "data" / subdir).mkdir(parents=True)
    return tmp_path / "data"


# ============================================================
# Tolerance constants for floating-point comparisons
# ============================================================


@pytest.fixture
def price_tolerance() -> float:
    """Tolerance for price comparisons (₹0.01)."""
    return 0.01


@pytest.fixture
def greek_tolerance() -> float:
    """Tolerance for Greek comparisons."""
    return 1e-6


@pytest.fixture
def iv_tolerance() -> float:
    """Tolerance for implied volatility comparisons."""
    return 1e-4
