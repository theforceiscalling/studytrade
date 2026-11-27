"""
Tests for studytrade.config — Configuration loading and validation.

What we test:
  1. Default config loads and validates correctly.
  2. Local overrides merge properly (deep merge).
  3. Invalid values are rejected by pydantic.
  4. Config is immutable after creation.
  5. Environment variable overrides work.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from studytrade.config import (
    StudyTradeConfig,
    _deep_merge,
    load_config,
)


class TestDeepMerge:
    """Test the YAML deep merge helper."""

    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        assert _deep_merge(base, override) == {"a": 1, "b": 99}

    def test_nested_merge(self):
        base = {"capital": {"initial": 500000, "currency": "INR"}}
        override = {"capital": {"initial": 200000}}
        result = _deep_merge(base, override)
        assert result["capital"]["initial"] == 200000
        assert result["capital"]["currency"] == "INR"  # Preserved

    def test_new_key_added(self):
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_list_replaced_not_appended(self):
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = _deep_merge(base, override)
        assert result["items"] == [4, 5]

    def test_empty_override(self):
        base = {"a": 1}
        assert _deep_merge(base, {}) == {"a": 1}


class TestStudyTradeConfig:
    """Test the pydantic config model."""

    def test_default_construction(self):
        cfg = StudyTradeConfig()
        assert cfg.capital.initial == 500000
        assert cfg.costs.brokerage_per_order == 20.0
        assert cfg.market.exchange == "NSE"

    def test_custom_capital(self):
        cfg = StudyTradeConfig(capital={"initial": 200000})
        assert cfg.capital.initial == 200000

    def test_invalid_environment_rejected(self):
        with pytest.raises(ValidationError):
            StudyTradeConfig(project={"environment": "yolo"})

    def test_invalid_data_source_rejected(self):
        with pytest.raises(ValidationError):
            StudyTradeConfig(data={"source": "bloomberg"})

    def test_negative_capital_rejected(self):
        with pytest.raises(ValidationError):
            StudyTradeConfig(capital={"initial": -100})

    def test_config_is_frozen(self, sample_config: StudyTradeConfig):
        with pytest.raises(ValidationError):
            sample_config.capital = None  # type: ignore

    def test_risk_limits_valid_range(self):
        cfg = StudyTradeConfig(risk={"max_risk_per_trade_pct": 2.0})
        assert cfg.risk.max_risk_per_trade_pct == 2.0

    def test_risk_over_100_rejected(self):
        with pytest.raises(ValidationError):
            StudyTradeConfig(risk={"max_risk_per_trade_pct": 150})

    def test_lot_sizes(self):
        cfg = StudyTradeConfig()
        nifty = next(u for u in cfg.market.underlyings if u.symbol == "NIFTY")
        banknifty = next(u for u in cfg.market.underlyings if u.symbol == "BANKNIFTY")
        assert nifty.lot_size == 65
        assert banknifty.lot_size == 30

    def test_costs_all_non_negative(self):
        cfg = StudyTradeConfig()
        assert cfg.costs.brokerage_per_order >= 0
        assert cfg.costs.stt_sell_pct >= 0
        assert cfg.costs.exchange_txn_pct >= 0
        assert cfg.costs.gst_pct >= 0


class TestLoadConfig:
    """Test the full YAML loading pipeline."""

    def test_load_from_yaml(self, tmp_path: Path):
        """Create a minimal project structure and load config."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        config_data = {
            "capital": {"initial": 300000},
            "data": {"source": "synthetic"},
        }
        with open(config_dir / "default.yaml", "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(project_root=tmp_path)
        assert cfg.capital.initial == 300000

    def test_local_override(self, tmp_path: Path):
        """local.yaml should override default.yaml."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        with open(config_dir / "default.yaml", "w") as f:
            yaml.dump({"capital": {"initial": 500000}}, f)

        with open(config_dir / "local.yaml", "w") as f:
            yaml.dump({"capital": {"initial": 200000}}, f)

        cfg = load_config(project_root=tmp_path)
        assert cfg.capital.initial == 200000

    def test_missing_default_raises(self, tmp_path: Path):
        """Should raise if default.yaml is missing."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_path / "config").mkdir()
        # No default.yaml created

        with pytest.raises(FileNotFoundError):
            load_config(project_root=tmp_path)

    def test_env_var_override(self, tmp_path: Path, monkeypatch):
        """LOG_LEVEL env var should override config."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        with open(config_dir / "default.yaml", "w") as f:
            yaml.dump({"logging": {"level": "INFO"}}, f)

        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        cfg = load_config(project_root=tmp_path)
        assert cfg.logging.level == "DEBUG"
