"""
StudyTrade — Market Constants
===============================

All NSE India-specific constants in one place.
If the exchange changes lot sizes or fee structures, update HERE only.

Sources:
  - NSE circular archives: https://www.nseindia.com/regulations/circulars
  - Zerodha charge list: https://zerodha.com/charges
  - SEBI guidelines

IMPORTANT: Lot sizes change periodically. Always verify against the latest
NSE circular before live trading. Last verified: 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# ============================================================
# Option types
# ============================================================


class OptionType(StrEnum):
    CALL = "CE"
    PUT = "PE"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class TimeFrame(StrEnum):
    INTRADAY = "intraday"
    SWING = "swing"
    POSITIONAL = "positional"


# ============================================================
# Underlying instrument specs
# ============================================================


@dataclass(frozen=True)
class InstrumentSpec:
    """Specification for an underlying instrument on NSE."""

    symbol: str
    lot_size: int
    tick_size: float
    strike_interval: float  # Gap between consecutive strikes (points)
    max_weekly_expiries: int  # Number of weekly expiries typically available


# Standard NSE instruments (verify lot sizes against latest circular)
NIFTY = InstrumentSpec(
    symbol="NIFTY",
    lot_size=65,
    tick_size=0.05,
    strike_interval=50.0,
    max_weekly_expiries=7,
)

BANKNIFTY = InstrumentSpec(
    symbol="BANKNIFTY",
    lot_size=30,
    tick_size=0.05,
    strike_interval=100.0,
    max_weekly_expiries=7,
)

FINNIFTY = InstrumentSpec(
    symbol="FINNIFTY",
    lot_size=60,
    tick_size=0.05,
    strike_interval=50.0,
    max_weekly_expiries=4,
)

# Registry for lookup by symbol
INSTRUMENTS: dict[str, InstrumentSpec] = {
    "NIFTY": NIFTY,
    "BANKNIFTY": BANKNIFTY,
    "FINNIFTY": FINNIFTY,
}


# ============================================================
# NSE transaction cost constants
# ============================================================


@dataclass(frozen=True)
class NSECostSchedule:
    """
    NSE India transaction cost schedule for F&O segment.

    All percentages are stored as decimals (e.g., 0.0625% → 0.000625).

    Total cost for an option buy-sell round trip:
    ─────────────────────────────────────────────
    brokerage       = 2 × brokerage_per_order
    stt             = sell_turnover × stt_sell_rate
    exchange_txn    = total_turnover × exchange_txn_rate
    sebi_fee        = total_turnover × sebi_rate
    gst             = (brokerage + exchange_txn) × gst_rate
    stamp_duty      = buy_turnover × stamp_duty_buy_rate
    ─────────────────────────────────────────────
    total           = sum of all above
    """

    brokerage_per_order: float = 20.0  # ₹ flat
    stt_sell_rate: float = 0.000625  # 0.0625%
    exchange_txn_rate: float = 0.000495  # 0.0495%
    sebi_rate: float = 0.000001  # ₹10 per crore = 0.0001%
    gst_rate: float = 0.18  # 18%
    stamp_duty_buy_rate: float = 0.00003  # 0.003%


# Default cost schedule
DEFAULT_COSTS = NSECostSchedule()


# ============================================================
# Trading calendar constants
# ============================================================

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
TIMEZONE = "Asia/Kolkata"

# Option expiry happens at 15:30 on expiry day
# Weekly expiry: Thursday (Nifty), Wednesday (Bank Nifty) — changes periodically
# Monthly expiry: last Thursday of the month

# Trading days per year (approximate, for annualization)
TRADING_DAYS_PER_YEAR = 252
TRADING_MINUTES_PER_DAY = 375  # 09:15 to 15:30
DEFAULT_RISK_FREE_RATE = 0.07  # RBI 91-day T-Bill benchmark (~6.5% - 7.0%)


# ============================================================
# Risk limits (hard-coded safety nets, independent of config)
# ============================================================

ABSOLUTE_MAX_POSITION_SIZE_LOTS = 50  # Never exceed this many lots
ABSOLUTE_MAX_DAILY_LOSS_INR = 100_000  # ₹1 lakh hard daily limit
ABSOLUTE_MIN_PREMIUM_FOR_TRADE = 0.50  # Don't trade options below ₹0.50
