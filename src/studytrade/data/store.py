"""
StudyTrade — Parquet and SQLite Data Store
===========================================

High-performance storage layer for spot and option chains.
Uses Apache Parquet for columnar time-series storage and
SQLite for fast catalog metadata lookups.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from studytrade.constants import TIMEZONE
from studytrade.data.schemas import OPTION_COLUMNS, SPOT_COLUMNS


def _atomic_write_parquet(table: pa.Table, file_path: Path) -> None:
    """Write via a temp file + rename so concurrent readers never see a partial file."""
    tmp_path = file_path.with_name(f"{file_path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, tmp_path, compression="snappy")
        os.replace(tmp_path, file_path)
    finally:
        tmp_path.unlink(missing_ok=True)


class DataStore:
    """
    Manages local storage of cleaned market data in Parquet format,
    indexed by an embedded SQLite catalog.
    """

    def __init__(self, base_dir: str | Path = "data") -> None:
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "catalog.db"
        self._init_catalog_db()

    def _init_catalog_db(self) -> None:
        """Initialize SQLite catalog schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_catalog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(dataset_type, symbol, file_path)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save_spot(self, df: pd.DataFrame, symbol: str) -> Path:
        """
        Save spot DataFrame as a compressed Parquet file and register in catalog.
        """
        if df.empty:
            raise ValueError("Cannot save empty DataFrame")

        symbol = symbol.upper()
        out_dir = self.processed_dir / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / "spot.parquet"

        # Write Parquet with snappy compression
        table = pa.Table.from_pandas(df[SPOT_COLUMNS])
        _atomic_write_parquet(table, file_path)

        # Update SQLite catalog
        start_date = str(df["timestamp"].min())
        end_date = str(df["timestamp"].max())
        row_count = len(df)
        now_str = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO dataset_catalog (
                    dataset_type, symbol, start_date, end_date, row_count, file_path, updated_at
                )
                VALUES ('spot', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_type, symbol, file_path) DO UPDATE SET
                    start_date=excluded.start_date,
                    end_date=excluded.end_date,
                    row_count=excluded.row_count,
                    updated_at=excluded.updated_at
                """,
                (symbol, start_date, end_date, row_count, str(file_path), now_str),
            )
            conn.commit()
        finally:
            conn.close()

        return file_path

    def load_spot(
        self,
        symbol: str,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
    ) -> pd.DataFrame:
        """
        Load spot DataFrame for symbol, optionally filtered by date range.
        """
        file_path = self.processed_dir / symbol.upper() / "spot.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"No stored spot data for {symbol} at {file_path}")

        df = pd.read_parquet(file_path)

        if start_date is not None:
            s_dt = (
                pd.to_datetime(start_date).tz_localize(TIMEZONE)
                if pd.to_datetime(start_date).tz is None
                else pd.to_datetime(start_date)
            )
            df = df[df["timestamp"] >= s_dt]

        if end_date is not None:
            e_dt = (
                pd.to_datetime(end_date).tz_localize(TIMEZONE)
                if pd.to_datetime(end_date).tz is None
                else pd.to_datetime(end_date)
            )
            df = df[df["timestamp"] <= e_dt]

        return df.reset_index(drop=True)

    def save_options(self, df: pd.DataFrame, underlying_symbol: str) -> Path:
        """
        Save options DataFrame partitioned by underlying symbol.
        """
        if df.empty:
            raise ValueError("Cannot save empty DataFrame")

        symbol = underlying_symbol.upper()
        out_dir = self.processed_dir / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / "options.parquet"

        table = pa.Table.from_pandas(df[OPTION_COLUMNS])
        _atomic_write_parquet(table, file_path)

        start_date = str(df["timestamp"].min())
        end_date = str(df["timestamp"].max())
        row_count = len(df)
        now_str = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO dataset_catalog (
                    dataset_type, symbol, start_date, end_date, row_count, file_path, updated_at
                )
                VALUES ('options', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_type, symbol, file_path) DO UPDATE SET
                    start_date=excluded.start_date,
                    end_date=excluded.end_date,
                    row_count=excluded.row_count,
                    updated_at=excluded.updated_at
                """,
                (symbol, start_date, end_date, row_count, str(file_path), now_str),
            )
            conn.commit()
        finally:
            conn.close()

        return file_path

    def load_options(
        self,
        underlying_symbol: str,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        strikes: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Load options DataFrame for underlying_symbol with optional filters.
        """
        file_path = self.processed_dir / underlying_symbol.upper() / "options.parquet"
        if not file_path.exists():
            raise FileNotFoundError(
                f"No stored options data for {underlying_symbol} at {file_path}"
            )

        df = pd.read_parquet(file_path)

        if start_date is not None:
            s_dt = (
                pd.to_datetime(start_date).tz_localize(TIMEZONE)
                if pd.to_datetime(start_date).tz is None
                else pd.to_datetime(start_date)
            )
            df = df[df["timestamp"] >= s_dt]

        if end_date is not None:
            e_dt = (
                pd.to_datetime(end_date).tz_localize(TIMEZONE)
                if pd.to_datetime(end_date).tz is None
                else pd.to_datetime(end_date)
            )
            df = df[df["timestamp"] <= e_dt]

        if strikes is not None:
            df = df[df["strike"].isin(strikes)]

        return df.reset_index(drop=True)

    def get_catalog(self) -> pd.DataFrame:
        """Query SQLite catalog to view all available cached datasets."""
        conn = sqlite3.connect(self.db_path)
        try:
            return pd.read_sql_query("SELECT * FROM dataset_catalog", conn)
        finally:
            conn.close()
