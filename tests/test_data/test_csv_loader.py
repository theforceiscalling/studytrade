"""
Tests for studytrade.data.fetchers.csv_loader — CSV loader and auto-column mapping.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studytrade.data.fetchers.csv_loader import CSVDataFetcher


class TestCSVLoader:
    """Test CSV file ingestion and header auto-mapping."""

    def test_load_spot_with_aliases(self, tmp_path: Path):
        # Create CSV with non-standard column headers common in Indian market dumps
        csv_content = """Date,Ticker,Open_Price,HP,LP,LTP,Qty
2024-01-01 09:15:00,NIFTY,21500,21600,21400,21550,50000
2024-01-02 09:15:00,NIFTY,21550,21650,21500,21600,60000
"""
        csv_file = tmp_path / "nifty_spot.csv"
        csv_file.write_text(csv_content)

        loader = CSVDataFetcher(data_dir=tmp_path)
        df = loader.load_spot_file(csv_file)

        assert len(df) == 2
        assert "close" in df.columns
        assert df["close"].iloc[0] == 21550.0
        assert df["volume"].iloc[0] == 50000

    def test_missing_file_raises(self, tmp_path: Path):
        loader = CSVDataFetcher(data_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load_spot_file(tmp_path / "non_existent.csv")

    def test_load_options_csv(self, tmp_path: Path):
        csv_header = (
            "Trade_Date,Underlying,Expiry_Date,Strike_Price,Opt_Type,Open,High,Low,"
            "Close,Buy_Price,Sell_Price,Contracts,OI,IV\n"
        )
        csv_row1 = (
            "2024-01-01 09:15:00,NIFTY,2024-01-25 15:30:00,21500,CE,"
            "150,160,140,155,154,156,2000,10000,0.16\n"
        )
        csv_row2 = (
            "2024-01-01 09:15:00,NIFTY,2024-01-25 15:30:00,21500,PE,"
            "140,150,130,145,144,146,1800,9000,0.165\n"
        )
        csv_content = csv_header + csv_row1 + csv_row2
        csv_file = tmp_path / "nifty_options.csv"
        csv_file.write_text(csv_content)

        loader = CSVDataFetcher(data_dir=tmp_path)
        df = loader.load_options_file(csv_file)

        assert len(df) == 2
        assert df["strike"].iloc[0] == 21500.0
        assert df["option_type"].iloc[0] == "CE"
        assert df["option_type"].iloc[1] == "PE"
