"""
Tests for studytrade.data.schemas — Data validation and normalization.
"""

from __future__ import annotations

import pandas as pd
import pytest

from studytrade.data.schemas import (
    validate_options_dataframe,
    validate_spot_dataframe,
)


class TestSpotValidation:
    """Test spot DataFrame validation rules."""

    def test_valid_spot_dataframe(self):
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01 09:15:00", "2024-01-02 09:15:00"],
                "symbol": ["NIFTY", "NIFTY"],
                "open": [21500.0, 21550.0],
                "high": [21600.0, 21650.0],
                "low": [21450.0, 21500.0],
                "close": [21550.0, 21600.0],
                "volume": [100000, 120000],
            }
        )
        clean_df, report = validate_spot_dataframe(df)
        assert len(clean_df) == 2
        assert report.is_clean
        assert str(clean_df["timestamp"].dt.tz) == "Asia/Kolkata"

    def test_missing_column_raises(self):
        df = pd.DataFrame({"timestamp": ["2024-01-01"], "close": [21500.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_spot_dataframe(df)

    def test_negative_price_dropped(self):
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02"],
                "symbol": ["NIFTY", "NIFTY"],
                "open": [21500.0, -100.0],
                "high": [21600.0, 21650.0],
                "low": [21450.0, 21500.0],
                "close": [21550.0, 21600.0],
                "volume": [1000, 1000],
            }
        )
        clean_df, report = validate_spot_dataframe(df, drop_invalid=True)
        assert len(clean_df) == 1
        assert report.dropped_rows == 1

    def test_high_lower_than_low_dropped(self):
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01"],
                "symbol": ["NIFTY"],
                "open": [21500.0],
                "high": [21400.0],  # Invalid: high < open & low
                "low": [21450.0],
                "close": [21500.0],
                "volume": [1000],
            }
        )
        clean_df, report = validate_spot_dataframe(df, drop_invalid=True)
        assert len(clean_df) == 0
        assert report.dropped_rows == 1


class TestOptionsValidation:
    """Test options chain DataFrame validation."""

    def test_valid_options_dataframe(self):
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01 09:15:00"],
                "symbol": ["NIFTY24JAN21500CE"],
                "underlying_symbol": ["NIFTY"],
                "expiry": ["2024-01-25 15:30:00"],
                "strike": [21500.0],
                "option_type": ["CE"],
                "open": [150.0],
                "high": [160.0],
                "low": [140.0],
                "close": [155.0],
                "bid": [154.0],
                "ask": [156.0],
                "volume": [5000],
                "open_interest": [25000],
                "implied_volatility": [0.15],
            }
        )
        clean_df, report = validate_options_dataframe(df)
        assert len(clean_df) == 1
        assert report.is_clean

    def test_crossed_book_detected_and_dropped(self):
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01"],
                "symbol": ["NIFTY24JAN21500CE"],
                "underlying_symbol": ["NIFTY"],
                "expiry": ["2024-01-25"],
                "strike": [21500.0],
                "option_type": ["CE"],
                "open": [150.0],
                "high": [160.0],
                "low": [140.0],
                "close": [150.0],
                "bid": [160.0],  # Crossed: bid > ask
                "ask": [150.0],
                "volume": [500],
                "open_interest": [1000],
                "implied_volatility": [0.15],
            }
        )
        clean_df, report = validate_options_dataframe(df, drop_invalid=True)
        assert len(clean_df) == 0
        assert report.dropped_rows == 1
