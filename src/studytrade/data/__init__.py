"""
StudyTrade — Data Layer
=========================

Data ingestion, normalization, validation, and storage.
"""

from studytrade.data.cleaner import DataCleaner, LookAheadBiasError, PointInTimeView
from studytrade.data.fetchers.base import BaseDataFetcher
from studytrade.data.fetchers.csv_loader import CSVDataFetcher
from studytrade.data.fetchers.synthetic import SyntheticDataGenerator
from studytrade.data.schemas import (
    OPTION_COLUMNS,
    SPOT_COLUMNS,
    OptionBar,
    SpotBar,
    ValidationReport,
    validate_options_dataframe,
    validate_spot_dataframe,
)
from studytrade.data.store import DataStore

__all__ = [
    "BaseDataFetcher",
    "CSVDataFetcher",
    "DataCleaner",
    "DataStore",
    "LookAheadBiasError",
    "OPTION_COLUMNS",
    "OptionBar",
    "PointInTimeView",
    "SPOT_COLUMNS",
    "SpotBar",
    "SyntheticDataGenerator",
    "ValidationReport",
    "validate_options_dataframe",
    "validate_spot_dataframe",
]
