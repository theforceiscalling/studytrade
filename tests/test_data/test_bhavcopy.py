"""
Unit tests for Bhavcopy importer and parser.
"""

from __future__ import annotations

import pandas as pd
import pytest

from studytrade.data.bhavcopy import BhavcopyImporter


def test_bhavcopy_parse_modern(tmp_path):
    # Construct synthetic modern NSE bhavcopy format
    data = {
        "FinInstrmTp": ["IDO", "IDO", "STO"],
        "TckrSymb": ["NIFTY", "NIFTY", "RELIANCE"],
        "TradDt": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "XpryDt": ["2024-01-25", "2024-01-25", "2024-01-25"],
        "StrkPric": [21700.0, 21800.0, 2500.0],
        "OptnTp": ["CE", "PE", "CE"],
        "OpnPric": [150.0, 120.0, 45.0],
        "HghPric": [160.0, 130.0, 50.0],
        "LwPric": [140.0, 110.0, 40.0],
        "ClsPric": [155.0, 125.0, 48.0],
        "SttlmPric": [155.0, 125.0, 48.0],
        "UndrlygPric": [21750.0, 21750.0, 2520.0],
        "OpnIntrst": [50000, 60000, 10000],
        "TtlTradgVol": [1500, 2000, 300],
    }
    df_raw = pd.DataFrame(data)
    importer = BhavcopyImporter(data_dir=tmp_path, cache_dir=tmp_path / "cache")
    parsed = importer.parse_bhavcopy(df_raw, underlying_symbols=["NIFTY"])

    assert "NIFTY" in parsed
    nifty_df = parsed["NIFTY"]
    assert len(nifty_df) == 2
    assert "strike" in nifty_df.columns
    assert "implied_volatility" in nifty_df.columns
    assert nifty_df["strike"].iloc[0] == 21700.0
    assert nifty_df["option_type"].iloc[0] == "CE"
    assert nifty_df["close"].iloc[0] == 155.0


def test_bhavcopy_parse_historical(tmp_path):
    data = {
        "INSTRUMENT": ["OPTIDX", "OPTIDX", "FUTIDX"],
        "SYMBOL": ["BANKNIFTY", "BANKNIFTY", "BANKNIFTY"],
        "TIMESTAMP": ["01-JAN-2024", "01-JAN-2024", "01-JAN-2024"],
        "EXPIRY_DT": ["25-Jan-2024", "25-Jan-2024", "25-Jan-2024"],
        "STRIKE_PR": [48000.0, 48500.0, 0.0],
        "OPTION_TYP": ["CE", "PE", "XX"],
        "OPEN": [300.0, 250.0, 48200.0],
        "HIGH": [350.0, 280.0, 48300.0],
        "LOW": [280.0, 220.0, 48000.0],
        "CLOSE": [320.0, 260.0, 48150.0],
        "SETTLE_PR": [320.0, 260.0, 48150.0],
        "CONTRACTS": [500, 600, 1000],
        "OPEN_INT": [12000, 15000, 25000],
    }
    df_raw = pd.DataFrame(data)
    importer = BhavcopyImporter(data_dir=tmp_path, cache_dir=tmp_path / "cache")
    parsed = importer.parse_bhavcopy(df_raw, underlying_symbols=["BANKNIFTY"])

    assert "BANKNIFTY" in parsed
    bn_df = parsed["BANKNIFTY"]
    assert len(bn_df) == 2
    assert bn_df["strike"].iloc[0] == 48000.0
    assert bn_df["option_type"].iloc[0] == "CE"
    assert bn_df["close"].iloc[0] == 320.0
