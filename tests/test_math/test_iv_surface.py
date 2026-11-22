"""
Tests for studytrade.math.iv_surface — Realized volatility, IV rank, percentile, and skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from studytrade.math.iv_surface import (
    calculate_iv_skew,
    iv_percentile,
    iv_rank,
    iv_rv_spread,
    realized_volatility_close,
    realized_volatility_parkinson,
)


class TestIVSurfaceAnalytics:
    """Test volatility analytics, ranking, and skew calculations."""

    def test_realized_volatility_close(self):
        # Generate 100 constant 1% moves
        prices = 100.0 * np.exp(np.cumsum(np.full(50, 0.01)))
        rv = realized_volatility_close(prices, window=20)
        assert not np.isnan(rv.iloc[-1])
        assert rv.iloc[-1] > 0.0

    def test_realized_volatility_parkinson(self):
        highs = pd.Series([102.0, 103.0, 101.5] * 10)
        lows = pd.Series([98.0, 99.0, 97.5] * 10)
        rv_park = realized_volatility_parkinson(highs, lows, window=20)
        assert not np.isnan(rv_park.iloc[-1])
        assert rv_park.iloc[-1] > 0.0

    def test_iv_rank_and_percentile(self):
        history = np.array([10.0, 15.0, 20.0, 25.0, 30.0])

        # Min = 10, Max = 30
        assert iv_rank(10.0, history) == 0.0
        assert iv_rank(30.0, history) == 100.0
        assert iv_rank(20.0, history) == 50.0

        # Percentile: 3 out of 5 values are strictly < 22.0 -> 60%
        assert iv_percentile(22.0, history) == 60.0

    def test_iv_rv_spread(self):
        # IV = 18%, RV = 14% -> VRP = +4%
        spread = iv_rv_spread(0.18, 0.14)
        assert abs(spread - 0.04) < 1e-6

    def test_calculate_iv_skew(self):
        chain_data = pd.DataFrame(
            {
                "strike": [21200.0, 21500.0, 21800.0, 21200.0, 21500.0, 21800.0],
                "option_type": ["PE", "PE", "PE", "CE", "CE", "CE"],
                "implied_volatility": [0.19, 0.16, 0.14, 0.17, 0.16, 0.15],
            }
        )
        skew = calculate_iv_skew(chain_data, spot_price=21500.0, strike_distance=300.0)
        assert skew.atm_iv == 0.16
        assert skew.otm_put_iv == 0.19
        assert skew.otm_call_iv == 0.15
        assert abs(skew.absolute_skew - 0.04) < 1e-6
        assert skew.normalized_skew > 0.0  # Normal equity negative skew
