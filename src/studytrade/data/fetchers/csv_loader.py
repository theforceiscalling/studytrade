"""
StudyTrade — CSV Data Loader
==============================

Loads and normalizes spot and option chain data from CSV files,
supporting standard NSE Bhavcopy formats, broker exports, and custom files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from studytrade.constants import TIMEZONE
from studytrade.data.fetchers.base import BaseDataFetcher
from studytrade.data.schemas import (
    validate_options_dataframe,
    validate_spot_dataframe,
)

# Standard alias dictionaries for auto-mapping CSV headers
SPOT_COLUMN_ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp", "date", "datetime", "time", "date_time"],
    "symbol": ["symbol", "ticker", "instrument", "tradingsymbol"],
    "open": ["open", "open_price", "op"],
    "high": ["high", "high_price", "hp"],
    "low": ["low", "low_price", "lp"],
    "close": ["close", "close_price", "ltp", "last_price", "cp"],
    "volume": ["volume", "vol", "shares_traded", "qty"],
}

OPTION_COLUMN_ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp", "date", "datetime", "trade_date"],
    "symbol": ["symbol", "tradingsymbol", "contract_symbol"],
    "underlying_symbol": ["underlying_symbol", "underlying", "index_name", "stock"],
    "expiry": ["expiry", "expiry_date", "exp_date", "expiration"],
    "strike": ["strike", "strike_price", "strike_pr"],
    "option_type": ["option_type", "opt_type", "type", "call_put", "instrument_type"],
    "open": ["open", "open_price"],
    "high": ["high", "high_price"],
    "low": ["low", "low_price"],
    "close": ["close", "close_price", "ltp", "settle_pr", "settlement_price"],
    "bid": ["bid", "bid_price", "best_bid", "buy_price"],
    "ask": ["ask", "ask_price", "best_ask", "offer_price", "sell_price"],
    "volume": ["volume", "vol", "contracts_traded", "contracts"],
    "open_interest": ["open_interest", "oi", "open_int"],
    "implied_volatility": ["implied_volatility", "iv", "implied_vol"],
}


def _remap_columns(df: pd.DataFrame, alias_map: dict[str, list[str]]) -> pd.DataFrame:
    """Normalize DataFrame column headers to canonical names based on alias lists."""
    col_mapping: dict[str, str] = {}
    lower_cols = {c.lower().strip().replace(" ", "_"): c for c in df.columns}

    for standard_col, aliases in alias_map.items():
        for alias in aliases:
            if alias in lower_cols:
                original_col = lower_cols[alias]
                col_mapping[original_col] = standard_col
                break

    return df.rename(columns=col_mapping)


class CSVDataFetcher(BaseDataFetcher):
    """
    Loads spot and option chain files from disk.
    """

    def __init__(self, data_dir: str | Path = "data/raw") -> None:
        self.data_dir = Path(data_dir)

    def load_spot_file(
        self,
        filepath: str | Path,
        symbol: str | None = None,
        custom_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Load an underlying spot CSV file.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Spot CSV file not found: {path}")

        df = pd.read_csv(path)
        if custom_mapping:
            df = df.rename(columns=custom_mapping)
        else:
            df = _remap_columns(df, SPOT_COLUMN_ALIASES)

        if "symbol" not in df.columns and symbol is not None:
            df["symbol"] = symbol.upper()

        if "volume" not in df.columns:
            df["volume"] = 0

        clean_df, report = validate_spot_dataframe(df)
        return clean_df

    def load_options_file(
        self,
        filepath: str | Path,
        underlying_symbol: str | None = None,
        custom_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Load an option chain CSV file.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Options CSV file not found: {path}")

        df = pd.read_csv(path)
        if custom_mapping:
            df = df.rename(columns=custom_mapping)
        else:
            df = _remap_columns(df, OPTION_COLUMN_ALIASES)

        if "underlying_symbol" not in df.columns and underlying_symbol is not None:
            df["underlying_symbol"] = underlying_symbol.upper()

        if "symbol" not in df.columns:
            # Synthetic contract symbol construction
            df["symbol"] = (
                df["underlying_symbol"].astype(str)
                + pd.to_datetime(df["expiry"]).dt.strftime("%y%b").str.upper()
                + df["strike"].astype(int).astype(str)
                + df["option_type"].astype(str).str.upper()
            )

        # Fallback if bid/ask not provided in historical bhavcopy
        if "bid" not in df.columns:
            df["bid"] = df["close"] * 0.99
        if "ask" not in df.columns:
            df["ask"] = df["close"] * 1.01
        if "implied_volatility" not in df.columns:
            df["implied_volatility"] = float("nan")

        clean_df, report = validate_options_dataframe(df)
        return clean_df

    def get_spot(
        self,
        symbol: str,
        start_date: datetime | str,
        end_date: datetime | str,
    ) -> pd.DataFrame:
        """Load spot data from data_dir matching symbol."""
        file_path = self.data_dir / f"{symbol.lower()}_spot.csv"
        if not file_path.exists():
            file_path = self.data_dir / f"{symbol.upper()}_spot.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Could not locate spot file for {symbol} in {self.data_dir}")

        df = self.load_spot_file(file_path, symbol=symbol)
        start_dt = (
            pd.to_datetime(start_date).tz_localize(TIMEZONE)
            if pd.to_datetime(start_date).tz is None
            else pd.to_datetime(start_date)
        )
        end_dt = (
            pd.to_datetime(end_date).tz_localize(TIMEZONE)
            if pd.to_datetime(end_date).tz is None
            else pd.to_datetime(end_date)
        )
        return df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].reset_index(
            drop=True
        )

    def get_options(
        self,
        underlying_symbol: str,
        start_date: datetime | str,
        end_date: datetime | str,
        expiries: list[datetime | str] | None = None,
        strikes: list[float] | None = None,
    ) -> pd.DataFrame:
        """Load options data from data_dir matching symbol."""
        file_path = self.data_dir / f"{underlying_symbol.lower()}_options.csv"
        if not file_path.exists():
            file_path = self.data_dir / f"{underlying_symbol.upper()}_options.csv"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not locate options file for {underlying_symbol} in {self.data_dir}"
            )

        df = self.load_options_file(file_path, underlying_symbol=underlying_symbol)
        start_dt = (
            pd.to_datetime(start_date).tz_localize(TIMEZONE)
            if pd.to_datetime(start_date).tz is None
            else pd.to_datetime(start_date)
        )
        end_dt = (
            pd.to_datetime(end_date).tz_localize(TIMEZONE)
            if pd.to_datetime(end_date).tz is None
            else pd.to_datetime(end_date)
        )
        filtered = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

        if expiries:
            exp_series = [pd.to_datetime(e).tz_convert(TIMEZONE) for e in expiries]
            filtered = filtered[filtered["expiry"].isin(exp_series)]

        if strikes:
            filtered = filtered[filtered["strike"].isin(strikes)]

        return filtered.reset_index(drop=True)
