"""
Tests for studytrade.data.store — Parquet and SQLite DataStore.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studytrade.data.fetchers.synthetic import SyntheticDataGenerator
from studytrade.data.store import DataStore


class TestDataStore:
    """Test Parquet write/read and SQLite catalog indexing."""

    @pytest.fixture
    def store(self, tmp_path: Path):
        return DataStore(base_dir=tmp_path)

    @pytest.fixture
    def gen(self):
        return SyntheticDataGenerator(seed=42)

    def test_spot_roundtrip(self, store, gen):
        spot_df = gen.generate_spot("NIFTY", n_days=30)
        file_path = store.save_spot(spot_df, "NIFTY")
        assert file_path.exists()

        # Load back
        loaded = store.load_spot("NIFTY")
        assert len(loaded) == 30
        assert (loaded["close"] == spot_df["close"]).all()

        # Verify catalog entry
        catalog = store.get_catalog()
        assert len(catalog) == 1
        assert catalog["dataset_type"].iloc[0] == "spot"
        assert catalog["symbol"].iloc[0] == "NIFTY"
        assert catalog["row_count"].iloc[0] == 30

    def test_options_roundtrip_with_filters(self, store, gen):
        options_df = gen.get_options("NIFTY", "2024-01-01", "2024-01-04")
        store.save_options(options_df, "NIFTY")

        # Load back
        loaded = store.load_options("NIFTY")
        assert len(loaded) == len(options_df)

        # Filter by strikes
        sample_strike = options_df["strike"].iloc[0]
        filtered = store.load_options("NIFTY", strikes=[sample_strike])
        assert set(filtered["strike"].unique()) == {sample_strike}

    def test_missing_data_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_spot("BANKNIFTY")
