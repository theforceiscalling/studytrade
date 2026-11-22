"""
StudyTrade — Data Fetchers
============================
"""

from studytrade.data.fetchers.base import BaseDataFetcher
from studytrade.data.fetchers.csv_loader import CSVDataFetcher
from studytrade.data.fetchers.synthetic import SyntheticDataGenerator

__all__ = [
    "BaseDataFetcher",
    "CSVDataFetcher",
    "SyntheticDataGenerator",
]
