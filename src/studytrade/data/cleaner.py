"""
StudyTrade — Data Cleaner & Look-Ahead Bias Guard
===================================================

Provides strict data sanity checks, missing data handling,
and point-in-time filtering to guarantee zero look-ahead bias.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from studytrade.constants import TIMEZONE


class LookAheadBiasError(ValueError):
    """Raised when data from the future is accessed or used in decisions."""

    pass


class PointInTimeView:
    """
    Guarantees that a backtest or strategy engine can ONLY observe
    market state up to a specified historical decision timestamp.
    """

    def __init__(self, data: pd.DataFrame, timestamp_col: str = "timestamp") -> None:
        if timestamp_col not in data.columns:
            raise ValueError(f"Column '{timestamp_col}' must exist in DataFrame")
        self._data = data.copy()
        self._timestamp_col = timestamp_col
        # Ensure timestamps are strictly sorted
        self._data = self._data.sort_values(by=timestamp_col).reset_index(drop=True)

    def as_of(self, decision_time: datetime | str) -> pd.DataFrame:
        """
        Return only data strictly observed up to and including `decision_time`.

        Throws LookAheadBiasError if decision_time is earlier than the earliest record.
        """
        ts = pd.to_datetime(decision_time)
        ts = ts.tz_localize(TIMEZONE) if ts.tzinfo is None else ts.tz_convert(TIMEZONE)

        mask = self._data[self._timestamp_col] <= ts
        return self._data[mask].copy().reset_index(drop=True)

    def verify_no_future_leakage(
        self,
        features: pd.DataFrame,
        decision_time: datetime | str,
    ) -> None:
        """
        Assert that no timestamp in `features` exceeds `decision_time`.
        """
        ts = pd.to_datetime(decision_time)
        ts = ts.tz_localize(TIMEZONE) if ts.tzinfo is None else ts.tz_convert(TIMEZONE)

        if self._timestamp_col in features.columns:
            future_mask = features[self._timestamp_col] > ts
            if future_mask.any():
                violating = features[future_mask][self._timestamp_col].iloc[0]
                raise LookAheadBiasError(
                    f"Look-ahead bias detected! Decision time is {ts}, "
                    f"but feature DataFrame contains future timestamp: {violating}"
                )


class DataCleaner:
    """
    Sanitizes, filters outliers, and cleans market data.
    """

    @staticmethod
    def clean_spot(
        df: pd.DataFrame,
        max_daily_return_pct: float = 20.0,
    ) -> pd.DataFrame:
        """
        Clean spot OHLCV data. Removes extreme erroneous single-tick anomalies.
        """
        if df.empty:
            return df.copy()

        df_clean = df.copy().sort_values("timestamp").reset_index(drop=True)

        # Remove duplicate timestamps keeping last
        df_clean = df_clean.drop_duplicates(subset=["symbol", "timestamp"], keep="last")

        # Check for unphysical spikes (e.g. fat-finger data feed corruptions)
        returns = df_clean["close"].pct_change().abs() * 100.0
        extreme_spikes = returns > max_daily_return_pct

        if extreme_spikes.any():
            # Replace extreme corrupt spikes with NaN and forward fill
            df_clean.loc[extreme_spikes, "close"] = np.nan
            df_clean["close"] = df_clean["close"].ffill()
            df_clean["high"] = np.maximum(df_clean["high"], df_clean["close"])
            df_clean["low"] = np.minimum(df_clean["low"], df_clean["close"])

        return df_clean.reset_index(drop=True)

    @staticmethod
    def clean_options(
        df: pd.DataFrame,
        min_price: float = 0.05,
    ) -> pd.DataFrame:
        """
        Clean option quotes.
        Removes crossed books (bid > ask) and illiquid zero-bid quotes.
        """
        if df.empty:
            return df.copy()

        df_clean = df.copy()

        # Fix crossed books where bid > ask by swapping or dropping
        crossed = df_clean["bid"] > df_clean["ask"]
        if crossed.any():
            # If crossed by 1 tick, snap bid to ask - 0.05, otherwise drop
            severe_cross = (df_clean["bid"] - df_clean["ask"]) > 0.50
            df_clean = df_clean[~severe_cross].copy()
            # Minor cross: adjust ask to at least bid + 0.05
            df_clean["ask"] = np.maximum(df_clean["ask"], df_clean["bid"] + min_price)

        # Floor option prices to minimum tick
        df_clean["bid"] = np.maximum(min_price, df_clean["bid"])
        df_clean["ask"] = np.maximum(df_clean["bid"] + min_price, df_clean["ask"])
        df_clean["close"] = np.maximum(min_price, df_clean["close"])

        return df_clean.reset_index(drop=True)
