from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from studytrade.analytics.gsquant_adapter import GS_QUANT_AVAILABLE, realized_vol
from studytrade.data.fetchers.nse_delayed import NSEOptionChainFetcher
from studytrade.live import pricing


def _fake_chain(expiry: str | None = None) -> dict:
    expiry = expiry or (datetime.now() + timedelta(days=30)).strftime("%d-%b-%Y")
    leg = {"lastPrice": 120.0, "buyPrice1": 119.0, "sellPrice1": 121.0, "impliedVolatility": 15.0,
           "openInterest": 100, "totalTradedVolume": 10}
    return {
        "symbol": "RELIANCE", "spot_price": 1500.0, "expiry": expiry, "timestamp": "t",
        "strikes": [{"strikePrice": 1500.0, "CE": leg, "PE": leg}, {"strikePrice": 1600.0, "CE": {}}],
    }


def test_chain_type_distinguishes_index_and_stock():
    assert NSEOptionChainFetcher.chain_type("nifty") == "Indices"
    assert NSEOptionChainFetcher.chain_type("RELIANCE") == "Equity"


def test_live_prices_include_theo_and_greeks():
    with patch.object(NSEOptionChainFetcher, "fetch_index_chain", return_value=_fake_chain()):
        out = pricing.get_live_option_prices("RELIANCE", strikes_around_atm=None)
    assert out["instrument_type"] == "Equity"
    assert len(out["contracts"]) == 2  # empty leg skipped
    ce = next(c for c in out["contracts"] if c["option_type"] == "CE")
    assert ce["mid"] == 120.0 and ce["theo_price"] > 0 and 0 < ce["delta"] < 1


def test_no_data_raises():
    with patch.object(NSEOptionChainFetcher, "fetch_index_chain", return_value=None), pytest.raises(RuntimeError):
        pricing.get_live_option_prices("NIFTY")


@pytest.mark.skipif(not GS_QUANT_AVAILABLE, reason="gs-quant not installed")
def test_gsquant_realized_vol():
    idx = pd.bdate_range("2026-01-01", periods=30)
    prices = pd.Series([100 + (i % 3) for i in range(30)], index=idx, dtype=float)
    vol = realized_vol(prices)
    assert vol is not None and vol > 0
