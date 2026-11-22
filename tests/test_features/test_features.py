"""
Tests for studytrade.features — Technical, volatility, microstructure, and regime features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from studytrade.features.microstructure import classify_oi_buildup, compute_max_pain, compute_pcr
from studytrade.features.regime import MarketRegime, detect_regimes
from studytrade.features.technical import (
    compute_atr,
    compute_donchian_breakout,
    compute_ema,
    compute_macd,
    compute_rsi,
)
from studytrade.features.volatility import (
    GARCH11,
    compute_volatility_expansion_ratio,
    realized_vol_garman_klass,
)


class TestTechnicalFeatures:
    """Validate technical indicator algorithms."""

    @pytest.fixture
    def sample_series(self):
        prices = 100.0 * np.exp(np.cumsum(np.random.default_rng(42).normal(0.001, 0.01, 100)))
        return pd.Series(prices)

    def test_ema_length_and_decay(self, sample_series):
        ema = compute_ema(sample_series, span=20)
        assert len(ema) == len(sample_series)
        assert not ema.isna().any()

    def test_macd(self, sample_series):
        macd_df = compute_macd(sample_series)
        assert set(macd_df.columns) == {"macd", "macd_signal", "macd_hist"}
        # Histogram must equal macd - signal
        diff = macd_df["macd"] - macd_df["macd_signal"]
        np.testing.assert_allclose(macd_df["macd_hist"].values, diff.values, atol=1e-8)

    def test_rsi_bounds(self, sample_series):
        rsi = compute_rsi(sample_series, period=14)
        valid = rsi.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_atr(self):
        highs = pd.Series([105.0, 106.0, 107.0] * 10)
        lows = pd.Series([95.0, 96.0, 97.0] * 10)
        closes = pd.Series([100.0, 102.0, 104.0] * 10)
        atr_df = compute_atr(highs, lows, closes, period=14)
        assert (atr_df["atr"].dropna() > 0.0).all()
        assert (atr_df["natr"].dropna() > 0.0).all()

    def test_donchian_breakout_no_lookahead(self):
        highs = pd.Series([100, 102, 104, 103, 110])  # Day 5 breaks out to 110
        lows = pd.Series([90, 92, 94, 93, 100])
        closes = pd.Series([95, 98, 100, 99, 109])

        breakout_df = compute_donchian_breakout(highs, lows, closes, lookback=3)
        # Prior 3-day high before day 5 (index 4) was max(102, 104, 103) = 104
        assert breakout_df["donchian_high"].iloc[4] == 104
        assert bool(breakout_df["breakout_bull"].iloc[4]) is True


class TestVolatilityFeatures:
    """Validate volatility estimators and GARCH."""

    def test_garman_klass_vol(self):
        n = 50
        o = pd.Series([100.0] * n)
        h = pd.Series([102.0] * n)
        l_ = pd.Series([98.0] * n)
        c = pd.Series([101.0] * n)
        gk = realized_vol_garman_klass(o, h, l_, c, window=10)
        valid = gk.dropna()
        assert len(valid) > 0
        assert (valid > 0.0).all()

    def test_garch11_fit_and_forecast(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.012, 150)
        garch = GARCH11()
        params = garch.fit(returns)

        assert params.alpha > 0.0
        assert params.beta > 0.0
        assert params.persistence < 1.0  # Stationary
        assert params.long_run_vol > 0.0

        forecasts = garch.forecast(horizon=5)
        assert len(forecasts) == 5
        assert (forecasts > 0.0).all()

    def test_volatility_expansion_ratio(self):
        vr = compute_volatility_expansion_ratio(pd.Series([15.0]), pd.Series([10.0]))
        assert vr.iloc[0] == 1.5


class TestMicrostructureFeatures:
    """Validate PCR, Max Pain, and OI buildup."""

    def test_pcr_calculation(self):
        chain = pd.DataFrame(
            {
                "option_type": ["CE", "CE", "PE", "PE"],
                "open_interest": [1000, 2000, 1500, 3000],
                "volume": [100, 200, 150, 450],
            }
        )
        pcr = compute_pcr(chain)
        # Put OI = 4500, Call OI = 3000 -> PCR_OI = 1.5
        assert pcr.pcr_oi == 1.5
        # Put Vol = 600, Call Vol = 300 -> PCR_Vol = 2.0
        assert pcr.pcr_volume == 2.0

    def test_max_pain_calculation(self):
        # Setup symmetric chain where middle strike has max pain
        chain = pd.DataFrame(
            {
                "strike": [21000.0, 21000.0, 21100.0, 21100.0, 21200.0, 21200.0],
                "option_type": ["CE", "PE", "CE", "PE", "CE", "PE"],
                "open_interest": [5000, 1000, 10000, 10000, 1000, 5000],
            }
        )
        max_pain = compute_max_pain(chain)
        assert max_pain == 21100.0

    def test_oi_buildup_classification(self):
        assert classify_oi_buildup(10.0, 500.0) == "LONG_BUILDUP"
        assert classify_oi_buildup(-10.0, 500.0) == "SHORT_BUILDUP"
        assert classify_oi_buildup(-10.0, -500.0) == "LONG_UNWINDING"
        assert classify_oi_buildup(10.0, -500.0) == "SHORT_COVERING"


class TestRegimeFeatures:
    """Validate market regime detection."""

    def test_detect_regimes(self):
        n = 150
        # Create an upward trending series
        trend = np.linspace(100, 200, n)
        closes = pd.Series(trend)
        highs = closes + 2.0
        lows = closes - 2.0

        regimes = detect_regimes(highs, lows, closes, trend_period=20, vol_period=10)
        assert len(regimes) == n
        assert MarketRegime.TRENDING_BULL.value in regimes.values
