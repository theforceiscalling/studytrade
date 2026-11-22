"""
StudyTrade — Data Schemas and Type Definitions
================================================

Strict type definitions, column schemas, and validation logic
for underlying spot, futures, and option chain time-series.

Guarantees:
  1. Standardized column naming across all data sources (CSV, API, synthetic).
  2. Timezone-aware timestamps normalized to Asia/Kolkata (IST).
  3. Physical market invariants (e.g. low <= open, close <= high, bid <= ask).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
import pandas as pd

from studytrade.constants import TIMEZONE, OptionType

# Standard column layouts
SPOT_COLUMNS: Final[list[str]] = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

OPTION_COLUMNS: Final[list[str]] = [
    "timestamp",
    "symbol",
    "underlying_symbol",
    "expiry",
    "strike",
    "option_type",
    "open",
    "high",
    "low",
    "close",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "implied_volatility",
]


@dataclass(frozen=True)
class SpotBar:
    """A single OHLCV bar for an underlying asset."""

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class OptionBar:
    """A single price and microstructure bar for an individual option contract."""

    timestamp: datetime
    symbol: str
    underlying_symbol: str
    expiry: datetime
    strike: float
    option_type: OptionType
    open: float
    high: float
    low: float
    close: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float = np.nan


@dataclass
class ValidationReport:
    """Summary of data quality checks and corrections."""

    total_rows: int
    valid_rows: int
    dropped_rows: int
    issues: list[str]

    @property
    def is_clean(self) -> bool:
        return self.dropped_rows == 0 and len(self.issues) == 0


def normalize_timestamp_series(s: pd.Series) -> pd.Series:
    """
    Ensure timestamps are parsed and localized to Asia/Kolkata.
    """
    ts = pd.to_datetime(s)
    if ts.dt.tz is None:
        return ts.dt.tz_localize(TIMEZONE)
    return ts.dt.tz_convert(TIMEZONE)


def validate_spot_dataframe(
    df: pd.DataFrame,
    *,
    drop_invalid: bool = True,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Validate and clean an underlying spot price DataFrame.

    Checks:
      - Required columns present
      - Valid timestamps and monotonic ordering
      - Non-negative prices and volume
      - High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
    """
    issues: list[str] = []
    initial_len = len(df)

    if initial_len == 0:
        return df.copy(), ValidationReport(0, 0, 0, ["DataFrame is empty"])

    # Column check
    missing_cols = [c for c in SPOT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Spot DataFrame missing required columns: {missing_cols}")

    df_clean = df.copy()
    df_clean["timestamp"] = normalize_timestamp_series(df_clean["timestamp"])

    # Drop duplicates by (symbol, timestamp)
    dup_mask = df_clean.duplicated(subset=["symbol", "timestamp"], keep="last")
    if dup_mask.any():
        num_dups = int(dup_mask.sum())
        issues.append(f"Dropped {num_dups} duplicate (symbol, timestamp) rows")
        df_clean = df_clean[~dup_mask]

    # Sort
    df_clean = df_clean.sort_values(by=["symbol", "timestamp"]).reset_index(drop=True)

    # Invariant masks
    price_positive = (
        (df_clean["open"] > 0)
        & (df_clean["high"] > 0)
        & (df_clean["low"] > 0)
        & (df_clean["close"] > 0)
    )
    hl_valid = (
        (df_clean["high"] >= df_clean["low"])
        & (df_clean["high"] >= df_clean["open"])
        & (df_clean["high"] >= df_clean["close"])
        & (df_clean["low"] <= df_clean["open"])
        & (df_clean["low"] <= df_clean["close"])
    )
    vol_valid = df_clean["volume"] >= 0

    valid_mask = price_positive & hl_valid & vol_valid
    dropped = int((~valid_mask).sum())

    if dropped > 0:
        issues.append(f"Detected {dropped} rows violating price/OHLC bounds")
        if drop_invalid:
            df_clean = df_clean[valid_mask].reset_index(drop=True)

    report = ValidationReport(
        total_rows=initial_len,
        valid_rows=len(df_clean),
        dropped_rows=initial_len - len(df_clean),
        issues=issues,
    )
    return df_clean, report


def validate_options_dataframe(
    df: pd.DataFrame,
    *,
    drop_invalid: bool = True,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Validate and clean an options chain DataFrame.

    Checks:
      - Required columns present
      - Option type valid ('CE' or 'PE')
      - Strike > 0
      - Invariants: bid <= ask (flag crossed books)
      - Non-negative prices, volume, OI
      - Expiry is on or after bar timestamp
    """
    issues: list[str] = []
    initial_len = len(df)

    if initial_len == 0:
        return df.copy(), ValidationReport(0, 0, 0, ["DataFrame is empty"])

    missing_cols = [c for c in OPTION_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Options DataFrame missing required columns: {missing_cols}")

    df_clean = df.copy()
    df_clean["timestamp"] = normalize_timestamp_series(df_clean["timestamp"])
    df_clean["expiry"] = normalize_timestamp_series(df_clean["expiry"])

    # Normalize option type to CE / PE
    df_clean["option_type"] = (
        df_clean["option_type"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"CALL": "CE", "C": "CE", "PUT": "PE", "P": "PE"})
    )

    valid_opt_type = df_clean["option_type"].isin([OptionType.CALL.value, OptionType.PUT.value])
    if (~valid_opt_type).any():
        bad_types = int((~valid_opt_type).sum())
        issues.append(f"Found {bad_types} rows with invalid option_type")

    # Invariant checks
    strike_valid = df_clean["strike"] > 0
    price_non_neg = (
        (df_clean["open"] >= 0)
        & (df_clean["high"] >= 0)
        & (df_clean["low"] >= 0)
        & (df_clean["close"] >= 0)
    )
    bid_ask_valid = (df_clean["bid"] >= 0) & (df_clean["ask"] >= df_clean["bid"])
    expiry_valid = df_clean["expiry"] >= df_clean["timestamp"]
    vol_oi_valid = (df_clean["volume"] >= 0) & (df_clean["open_interest"] >= 0)

    valid_mask = (
        valid_opt_type & strike_valid & price_non_neg & bid_ask_valid & expiry_valid & vol_oi_valid
    )
    dropped = int((~valid_mask).sum())

    if dropped > 0:
        issues.append(f"Detected {dropped} invalid rows (crossed book / expired / negative)")
        if drop_invalid:
            df_clean = df_clean[valid_mask].reset_index(drop=True)

    # Sort deterministically
    sort_cols = ["timestamp", "underlying_symbol", "expiry", "strike", "option_type"]
    df_clean = df_clean.sort_values(by=sort_cols).reset_index(drop=True)

    report = ValidationReport(
        total_rows=initial_len,
        valid_rows=len(df_clean),
        dropped_rows=initial_len - len(df_clean),
        issues=issues,
    )
    return df_clean, report
