"""
StudyTrade — Abstract Base Data Fetcher
=========================================

Interface that all data sources (Synthetic, CSV, Zerodha Kite, Upstox)
must implement. Guarantees interchangeable data backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class BaseDataFetcher(ABC):
    """Abstract interface for fetching spot and options historical / live data."""

    @abstractmethod
    def get_spot(
        self,
        symbol: str,
        start_date: datetime | str,
        end_date: datetime | str,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for an underlying spot instrument.

        Parameters
        ----------
        symbol : str
            Underlying symbol (e.g. 'NIFTY', 'BANKNIFTY').
        start_date : datetime or str
            Start date (inclusive).
        end_date : datetime or str
            End date (inclusive).

        Returns
        -------
        pd.DataFrame
            DataFrame complying with SPOT_COLUMNS schema.
        """
        pass

    @abstractmethod
    def get_options(
        self,
        underlying_symbol: str,
        start_date: datetime | str,
        end_date: datetime | str,
        expiries: list[datetime | str] | None = None,
        strikes: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch options chain quotes / OHLCV for an underlying asset.

        Parameters
        ----------
        underlying_symbol : str
            Symbol of the underlying (e.g. 'NIFTY').
        start_date : datetime or str
            Start date.
        end_date : datetime or str
            End date.
        expiries : list of dates, optional
            Filter for specific expiry dates.
        strikes : list of floats, optional
            Filter for specific strikes.

        Returns
        -------
        pd.DataFrame
            DataFrame complying with OPTION_COLUMNS schema.
        """
        pass
