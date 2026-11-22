"""
Tests for studytrade.strategy.signals.gamma_squeeze — Gamma Squeeze and Cascade Strategy.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from studytrade.strategy.base import SignalDirection
from studytrade.strategy.signals.gamma_squeeze import GammaSqueezeStrategy


class TestGammaSqueezeStrategy:
    """Validate systematic trade signal generation under extreme GEX conditions."""

    @pytest.fixture
    def sample_spot(self) -> pd.DataFrame:
        now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        return pd.DataFrame(
            {
                "timestamp": [now],
                "symbol": ["NIFTY"],
                "open": [21750.0],
                "high": [21820.0],
                "low": [21740.0],
                "close": [21810.0],  # Crossing 21800 Call Wall
                "volume": [1500000],
            }
        )

    @pytest.fixture
    def sample_chain(self) -> pd.DataFrame:
        now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        exp = datetime(2024, 1, 18, 15, 30, tzinfo=timezone.utc)  # 3 DTE
        strikes = [21400.0, 21600.0, 21800.0, 22000.0]
        rows = []
        for k in strikes:
            c_oi = 80000 if k == 21800.0 else 10000
            c_vol = 95000 if k == 21800.0 else 3000
            rows.append(
                {
                    "timestamp": now,
                    "symbol": f"NIFTY24JAN{int(k)}CE",
                    "underlying_symbol": "NIFTY",
                    "expiry": exp,
                    "strike": k,
                    "option_type": "CE",
                    "open_interest": c_oi,
                    "volume": c_vol,
                    "implied_volatility": 0.17,
                }
            )
            p_oi = 60000 if k == 21400.0 else 8000
            p_vol = 5000 if k == 21400.0 else 1000
            rows.append(
                {
                    "timestamp": now,
                    "symbol": f"NIFTY24JAN{int(k)}PE",
                    "underlying_symbol": "NIFTY",
                    "expiry": exp,
                    "strike": k,
                    "option_type": "PE",
                    "open_interest": p_oi,
                    "volume": p_vol,
                    "implied_volatility": 0.19,
                }
            )
        return pd.DataFrame(rows)

    def test_missing_or_empty_chain_returns_empty(self, sample_spot):
        strat = GammaSqueezeStrategy()
        assert strat.generate_signals(sample_spot, None) == []
        assert strat.generate_signals(sample_spot, pd.DataFrame()) == []
        assert strat.generate_signals(pd.DataFrame(), None) == []

    def test_bullish_gamma_squeeze_naked_signal(self, sample_spot, sample_chain):
        # Spot is at 21810, penetrating the 21800 Call Wall with massive call volume
        strat = GammaSqueezeStrategy(squeeze_score_threshold=60.0, use_spreads=False)
        signals = strat.generate_signals(sample_spot, sample_chain)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == SignalDirection.BUY_CALL
        assert sig.underlying_symbol == "NIFTY"
        assert sig.strength >= 0.60
        assert "Bullish Gamma Squeeze" in sig.reason
        assert sig.target_delta == 0.40
        assert sig.min_dte == 0
        assert sig.max_dte == 4

    def test_bullish_gamma_squeeze_spread_signal(self, sample_spot, sample_chain):
        strat = GammaSqueezeStrategy(squeeze_score_threshold=60.0, use_spreads=True)
        signals = strat.generate_signals(sample_spot, sample_chain)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == SignalDirection.BULL_CALL_SPREAD
        assert "Bullish Gamma Squeeze" in sig.reason

    def test_bearish_gamma_cascade_signal(self, sample_chain):
        now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        # Spot drops below 21400 Put Wall to 21380
        bear_spot = pd.DataFrame(
            {
                "timestamp": [now],
                "symbol": ["NIFTY"],
                "open": [21450.0],
                "high": [21460.0],
                "low": [21370.0],
                "close": [21380.0],
                "volume": [2000000],
            }
        )
        # Update chain with heavy volume on Put Wall (21400)
        chain_with_put_vol = sample_chain.copy()
        mask = (chain_with_put_vol["strike"] == 21400.0) & (chain_with_put_vol["option_type"] == "PE")
        chain_with_put_vol.loc[mask, "volume"] = 75000

        strat = GammaSqueezeStrategy(squeeze_score_threshold=60.0, allow_bearish=True, use_spreads=False)
        signals = strat.generate_signals(bear_spot, chain_with_put_vol)

        assert len(signals) >= 1
        sig = [s for s in signals if s.direction == SignalDirection.BUY_PUT][0]
        assert "Bearish Gamma Cascade" in sig.reason

    def test_bearish_cascade_suppressed_when_disabled(self, sample_chain):
        now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        bear_spot = pd.DataFrame(
            {
                "timestamp": [now],
                "symbol": ["NIFTY"],
                "open": [21450.0],
                "high": [21460.0],
                "low": [21370.0],
                "close": [21380.0],
                "volume": [2000000],
            }
        )
        strat = GammaSqueezeStrategy(squeeze_score_threshold=50.0, allow_bearish=False)
        signals = strat.generate_signals(bear_spot, sample_chain)
        # Must not generate PUT signals
        assert not any(s.direction in (SignalDirection.BUY_PUT, SignalDirection.BEAR_PUT_SPREAD) for s in signals)

    def test_neutral_market_produces_no_signals(self, sample_chain):
        now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        # Spot is at 21600 (safely midway between 21400 Put Wall and 21800 Call Wall)
        neutral_spot = pd.DataFrame(
            {
                "timestamp": [now],
                "symbol": ["NIFTY"],
                "open": [21600.0],
                "high": [21610.0],
                "low": [21590.0],
                "close": [21600.0],
                "volume": [500000],
            }
        )
        strat = GammaSqueezeStrategy(squeeze_score_threshold=65.0)
        signals = strat.generate_signals(neutral_spot, sample_chain)
        assert signals == []
