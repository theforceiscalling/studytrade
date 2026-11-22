"""
StudyTrade — Configuration Loader
===================================

Design decisions:
  1. YAML for human readability (not TOML/JSON — options traders aren't devs).
  2. Pydantic for validation — catch config errors at startup, not mid-backtest.
  3. Layered: default.yaml → local.yaml → environment variables → CLI overrides.
  4. Frozen after load — no mutation during a backtest run.

Usage:
    from studytrade.config import load_config
    cfg = load_config()                           # Uses default + local
    cfg = load_config("config/my_experiment.yaml") # Custom override file
    print(cfg.capital.initial)                     # 500000
    print(cfg.costs.brokerage_per_order)           # 20.0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

# ============================================================
# Project root detection
# ============================================================


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: cwd) until we find pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume cwd
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()


# ============================================================
# Pydantic config models
# ============================================================
# Each section of default.yaml maps to a model.
# Pydantic validates types, ranges, and required fields.
# ============================================================


class ProjectConfig(BaseModel):
    name: str = "studytrade"
    version: str = "0.1.0"
    environment: str = "development"

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "paper", "live"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}, got '{v}'")
        return v


class UnderlyingConfig(BaseModel):
    symbol: str
    lot_size: int = Field(gt=0)
    tick_size: float = Field(gt=0)


class TradingHoursConfig(BaseModel):
    open: str = "09:15"
    close: str = "15:30"
    timezone: str = "Asia/Kolkata"


class MarketConfig(BaseModel):
    exchange: str = "NSE"
    underlyings: list[UnderlyingConfig] = [
        UnderlyingConfig(symbol="NIFTY", lot_size=25, tick_size=0.05),
        UnderlyingConfig(symbol="BANKNIFTY", lot_size=15, tick_size=0.05),
    ]
    trading_hours: TradingHoursConfig = TradingHoursConfig()
    holidays_file: str = ""


class CapitalConfig(BaseModel):
    initial: float = Field(500000.0, gt=0)
    currency: str = "INR"


class RiskConfig(BaseModel):
    max_risk_per_trade_pct: float = Field(1.5, gt=0, le=100)
    max_daily_loss_pct: float = Field(5.0, gt=0, le=100)
    max_weekly_loss_pct: float = Field(10.0, gt=0, le=100)
    max_concurrent_positions: int = Field(3, gt=0)
    kill_switch_loss_pct: float = Field(15.0, gt=0, le=100)


class CostsConfig(BaseModel):
    """
    NSE India options transaction cost model.

    Math (for a single buy-then-sell option trade):
    ─────────────────────────────────────────────────
    Let P_buy = buy premium, P_sell = sell premium (per unit),
        L = lot size, N = number of lots.

    Turnover_buy  = P_buy  × L × N
    Turnover_sell = P_sell × L × N

    Brokerage     = 2 × brokerage_per_order  (buy + sell)
    STT           = Turnover_sell × stt_sell_pct / 100
    Exchange_txn  = (Turnover_buy + Turnover_sell) × exchange_txn_pct / 100
    SEBI          = (Turnover_buy + Turnover_sell) / 1e7 × sebi_per_crore
    GST           = (Brokerage + Exchange_txn) × gst_pct / 100
    Stamp_duty    = Turnover_buy × stamp_duty_buy_pct / 100

    Total_cost    = Brokerage + STT + Exchange_txn + SEBI + GST + Stamp_duty
    """

    brokerage_per_order: float = Field(20.0, ge=0)
    stt_sell_pct: float = Field(0.0625, ge=0)
    exchange_txn_pct: float = Field(0.0495, ge=0)
    sebi_per_crore: float = Field(10.0, ge=0)
    gst_pct: float = Field(18.0, ge=0)
    stamp_duty_buy_pct: float = Field(0.003, ge=0)
    slippage_ticks: float = Field(1.0, ge=0)
    use_bid_ask: bool = True


class CSVConfig(BaseModel):
    spot_file: str = ""
    option_chain_dir: str = ""


class SyntheticConfig(BaseModel):
    n_days: int = Field(252, gt=0)
    spot_start: float = Field(20000.0, gt=0)
    annual_vol: float = Field(0.18, gt=0, lt=5.0)
    annual_drift: float = 0.10
    seed: int = 42


class DataConfig(BaseModel):
    source: str = "synthetic"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    synthetic_dir: str = "data/synthetic"
    storage_format: str = "parquet"
    csv: CSVConfig = CSVConfig()
    synthetic: SyntheticConfig = SyntheticConfig()

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        allowed = {"synthetic", "csv", "kite", "upstox", "angel"}
        if v not in allowed:
            raise ValueError(f"data.source must be one of {allowed}, got '{v}'")
        return v

    @field_validator("storage_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed = {"parquet", "sqlite"}
        if v not in allowed:
            raise ValueError(f"storage_format must be one of {allowed}, got '{v}'")
        return v


class EntryConfig(BaseModel):
    signal_threshold: float = 0.6


class ExitConfig(BaseModel):
    stop_loss_pct: float = Field(50.0, ge=0, le=100)
    target_pct: float = Field(100.0, ge=0)
    time_stop_bars: int = Field(5, ge=1)
    theta_decay_exit_dte: int = Field(3, ge=0)


class StrategyConfig(BaseModel):
    default: str = "momentum"
    timeframe: str = "swing"
    entry: EntryConfig = EntryConfig()
    exit: ExitConfig = ExitConfig()

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        allowed = {"intraday", "swing", "positional"}
        if v not in allowed:
            raise ValueError(f"timeframe must be one of {allowed}, got '{v}'")
        return v


class SelectionConfig(BaseModel):
    strike_method: str = "delta"
    target_delta: float = Field(0.30, gt=0, lt=1)
    min_dte: int = Field(5, ge=0)
    max_dte: int = Field(30, ge=1)
    prefer_weekly: bool = True
    min_volume: int = Field(100, ge=0)
    min_oi: int = Field(500, ge=0)


class CVConfig(BaseModel):
    method: str = "walk_forward"
    n_splits: int = Field(5, ge=2)
    embargo_days: int = Field(5, ge=0)
    min_train_days: int = Field(126, ge=30)


class MLConfig(BaseModel):
    enabled: bool = False
    target: str = "direction"
    models: list[str] = ["logistic_regression", "random_forest", "gradient_boosting"]
    cv: CVConfig = CVConfig()
    calibrate_probabilities: bool = True


class BacktestConfig(BaseModel):
    engine: str = "vectorized"
    start_date: str = ""
    end_date: str = ""
    out_of_sample_pct: float = Field(20.0, ge=0, le=50)
    fill_model: str = "bid_ask"

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str) -> str:
        allowed = {"vectorized", "event_driven"}
        if v not in allowed:
            raise ValueError(f"backtest.engine must be one of {allowed}, got '{v}'")
        return v


class LoggingConfig(BaseModel):
    level: str = "INFO"
    console: bool = True
    file: bool = True
    log_dir: str = "logs"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


class PaperConfig(BaseModel):
    enabled: bool = False
    feed_source: str = "kite"
    poll_interval_sec: int = Field(5, ge=1)
    dashboard_port: int = Field(8501, ge=1024, le=65535)


# ============================================================
# Top-level config
# ============================================================


class StudyTradeConfig(BaseModel):
    """Root configuration — merges all sections."""

    model_config = {"frozen": True}  # Immutable after creation

    project: ProjectConfig = ProjectConfig()
    market: MarketConfig = MarketConfig()
    capital: CapitalConfig = CapitalConfig()
    risk: RiskConfig = RiskConfig()
    costs: CostsConfig = CostsConfig()
    data: DataConfig = DataConfig()
    strategy: StrategyConfig = StrategyConfig()
    selection: SelectionConfig = SelectionConfig()
    backtest: BacktestConfig = BacktestConfig()
    ml: MLConfig = MLConfig()
    logging: LoggingConfig = LoggingConfig()
    paper: PaperConfig = PaperConfig()


# ============================================================
# YAML deep merge
# ============================================================


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge `override` into `base`.
    Lists are replaced, not appended. Dicts are merged recursively.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ============================================================
# Config loader
# ============================================================


def load_config(
    override_path: str | Path | None = None,
    project_root: Path | None = None,
) -> StudyTradeConfig:
    """
    Load configuration with layered overrides:
      1. config/default.yaml  (always loaded)
      2. config/local.yaml    (if exists, gitignored)
      3. override_path        (if provided, e.g., experiment-specific)

    Returns a frozen StudyTradeConfig instance.

    Parameters
    ----------
    override_path : str or Path, optional
        Path to an additional YAML override file.
    project_root : Path, optional
        Project root directory. Auto-detected if not provided.

    Returns
    -------
    StudyTradeConfig
        Validated, immutable configuration object.

    Raises
    ------
    FileNotFoundError
        If default.yaml is missing.
    pydantic.ValidationError
        If any config value fails validation.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    config_dir = root / "config"

    # Layer 1: defaults (required)
    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(
            f"Default config not found: {default_path}\n"
            f"Make sure you're running from the project root or set project_root."
        )
    with open(default_path) as f:
        config_data = yaml.safe_load(f) or {}

    # Layer 2: local overrides (optional)
    local_path = config_dir / "local.yaml"
    if local_path.exists():
        with open(local_path) as f:
            local_data = yaml.safe_load(f) or {}
        config_data = _deep_merge(config_data, local_data)

    # Layer 3: explicit override file (optional)
    if override_path is not None:
        override_path = Path(override_path)
        if not override_path.is_absolute():
            override_path = root / override_path
        if override_path.exists():
            with open(override_path) as f:
                override_data = yaml.safe_load(f) or {}
            config_data = _deep_merge(config_data, override_data)

    # Layer 4: environment variable overrides (selected critical ones)
    env_overrides = {
        "LOG_LEVEL": ("logging", "level"),
    }
    for env_var, key_path in env_overrides.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Navigate into nested dict and set
            d = config_data
            for key in key_path[:-1]:
                d = d.setdefault(key, {})
            d[key_path[-1]] = value

    return StudyTradeConfig(**config_data)


# ============================================================
# Convenience: config singleton (optional pattern)
# ============================================================

_config: StudyTradeConfig | None = None


def get_config(**kwargs: Any) -> StudyTradeConfig:
    """Get or create the global config singleton."""
    global _config
    if _config is None:
        _config = load_config(**kwargs)
    return _config


def reset_config() -> None:
    """Reset the global config singleton (for testing)."""
    global _config
    _config = None
