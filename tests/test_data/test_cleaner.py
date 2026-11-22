"""
Tests for studytrade.data.cleaner — Look-Ahead Bias Guard and Data Cleaner.
"""

from __future__ import annotations

import pandas as pd
import pytest

from studytrade.data.cleaner import DataCleaner, LookAheadBiasError, PointInTimeView


class TestLookAheadGuard:
    """Test point-in-time filtering to avoid look-ahead bias."""

    @pytest.fixture
    def sample_time_series(self):
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="Asia/Kolkata"),
                "close": [100 + i for i in range(10)],
            }
        )

    def test_point_in_time_slicing(self, sample_time_series):
        pit = PointInTimeView(sample_time_series)
        # Slicing at day 5
        view = pit.as_of("2024-01-05")
        assert len(view) == 5
        assert view["timestamp"].max() <= pd.to_datetime("2024-01-05").tz_localize("Asia/Kolkata")

    def test_verify_no_future_leakage_success(self, sample_time_series):
        pit = PointInTimeView(sample_time_series)
        view = pit.as_of("2024-01-05")
        # Should not raise
        pit.verify_no_future_leakage(view, "2024-01-05")

    def test_verify_no_future_leakage_raises(self, sample_time_series):
        pit = PointInTimeView(sample_time_series)
        # Passing full dataset to a decision time on day 3 should trigger error
        with pytest.raises(LookAheadBiasError, match="Look-ahead bias detected"):
            pit.verify_no_future_leakage(sample_time_series, "2024-01-03")


class TestDataCleaner:
    """Test price sanity and anomaly filtering."""

    def test_clean_spot_extreme_spike(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="Asia/Kolkata"),
                "symbol": ["NIFTY"] * 5,
                "open": [20000.0, 20100.0, 20150.0, 20200.0, 20250.0],
                "high": [20050.0, 20150.0, 35000.0, 20250.0, 20300.0],
                "low": [19950.0, 20050.0, 20100.0, 20150.0, 20200.0],
                "close": [20000.0, 20100.0, 35000.0, 20200.0, 20250.0],  # 75% spike (bad tick)
                "volume": [1000] * 5,
            }
        )
        cleaned = DataCleaner.clean_spot(df, max_daily_return_pct=30.0)
        # Close should be replaced with previous close (20100)
        assert cleaned["close"].iloc[2] == 20100.0
