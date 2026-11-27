"""
Live NSE option pricing for index and stock options.

Pulls the official delayed NSE option chain and enriches every contract with
Black-Scholes theoretical value, Greeks and market-vs-model edge.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from studytrade.data.fetchers.nse_delayed import get_nse_delayed_fetcher
from studytrade.math.bsm import bsm_price
from studytrade.math.greeks import greeks
from studytrade.math.iv_solver import implied_volatility

IST = timezone(timedelta(hours=5, minutes=30))
DEFAULT_RISK_FREE = 0.07
_EXPIRY_FORMATS = ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d")
_MAX_SPOT_SAMPLES = 500

_spot_history: dict[str, list[tuple[datetime, float]]] = {}


def _years_to_expiry(expiry: str, now: datetime) -> float:
    for fmt in _EXPIRY_FORMATS:
        try:
            exp = datetime.strptime(expiry, fmt).replace(hour=15, minute=30, tzinfo=IST)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Unrecognised expiry format: {expiry!r}")
    return max((exp - now).total_seconds(), 0.0) / (365.0 * 86400.0)


def _spot_sample_vol(symbol: str, spot: float, now: datetime) -> float | None:
    """gs-quant realized vol over the spot samples collected from live polls."""
    from studytrade.analytics.gsquant_adapter import realized_vol

    hist = _spot_history.setdefault(symbol, [])
    hist.append((now, spot))
    del hist[:-_MAX_SPOT_SAMPLES]
    if len(hist) < 10:
        return None
    series = pd.Series([p for _, p in hist], index=pd.DatetimeIndex([t for t, _ in hist]))
    return realized_vol(series)


def _price_leg(
    opt_type: str, leg: dict[str, Any], spot: float, strike: float, t: float, r: float
) -> dict[str, Any] | None:
    ltp = float(leg.get("lastPrice") or 0.0)
    bid = float(leg.get("buyPrice1") or 0.0)
    ask = float(leg.get("sellPrice1") or 0.0)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else ltp
    if mid <= 0:
        return None

    iv = float(leg.get("impliedVolatility") or 0.0)
    iv = iv / 100.0 if iv > 0 else 0.0
    if iv <= 0 and t > 0:
        solved = float(implied_volatility(opt_type, mid, spot, strike, t, r))
        iv = solved if math.isfinite(solved) else 0.0

    row: dict[str, Any] = {
        "strike": strike,
        "option_type": opt_type,
        "ltp": ltp,
        "bid": bid,
        "ask": ask,
        "mid": round(mid, 2),
        "iv": round(iv, 4),
        "open_interest": int(leg.get("openInterest") or 0),
        "volume": int(leg.get("totalTradedVolume") or 0),
    }
    if iv > 0 and t > 0:
        theo = float(bsm_price(opt_type, spot, strike, t, r, iv))
        g = greeks(opt_type, spot, strike, t, r, iv, calendar_days=True)
        row.update(
            theo_price=round(theo, 2),
            edge=round(mid - theo, 2),
            delta=round(float(g.delta), 4),
            gamma=round(float(g.gamma), 6),
            vega_1pct=round(float(g.vega_1pct), 3),
            theta_daily=round(float(g.theta_daily), 3),
        )
    return row


def get_live_option_prices(
    symbol: str = "NIFTY",
    expiry: str | None = None,
    risk_free: float = DEFAULT_RISK_FREE,
    strikes_around_atm: int | None = 10,
) -> dict[str, Any]:
    """
    Live priced option chain for an NSE index or stock.

    Raises RuntimeError if NSE returns no data (market closed, blocked, bad symbol).
    """
    symbol = symbol.upper().strip()
    chain = get_nse_delayed_fetcher().fetch_index_chain(symbol, expiry)
    if not chain or chain["spot_price"] <= 0:
        raise RuntimeError(f"No live option chain available from NSE for {symbol}")

    now = datetime.now(IST)
    spot = chain["spot_price"]
    t = _years_to_expiry(chain["expiry"], now)

    rows_by_strike = sorted(chain["strikes"], key=lambda x: float(x.get("strikePrice", 0.0)))
    if strikes_around_atm:
        atm_idx = min(
            range(len(rows_by_strike)),
            key=lambda i: abs(float(rows_by_strike[i].get("strikePrice", 0.0)) - spot),
            default=0,
        )
        lo = max(atm_idx - strikes_around_atm, 0)
        rows_by_strike = rows_by_strike[lo : atm_idx + strikes_around_atm + 1]

    contracts = []
    for entry in rows_by_strike:
        strike = float(entry.get("strikePrice", 0.0))
        for opt_type in ("CE", "PE"):
            leg = entry.get(opt_type)
            priced = _price_leg(opt_type, leg, spot, strike, t, risk_free) if leg else None
            if priced:
                contracts.append(priced)

    return {
        "symbol": symbol,
        "instrument_type": get_nse_delayed_fetcher().chain_type(symbol),
        "spot": spot,
        "expiry": chain["expiry"],
        "years_to_expiry": round(t, 6),
        "risk_free": risk_free,
        "nse_timestamp": chain["timestamp"],
        "spot_sample_vol": _spot_sample_vol(symbol, spot, now),
        "contracts": contracts,
    }
