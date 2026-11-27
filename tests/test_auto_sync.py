from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from studytrade.data.bhavcopy import BhavcopyImporter


def _raw(day: str) -> pd.DataFrame:
    rows = []
    for strike, opt in [(25000, "CE"), (25000, "PE"), (25100, "CE")]:
        rows.append({
            "TradDt": day, "FinInstrmTp": "IDO", "TckrSymb": "NIFTY", "XpryDt": "2026-12-29",
            "StrkPric": strike, "OptnTp": opt, "OpnPric": 100.0, "HghPric": 110.0, "LwPric": 90.0,
            "ClsPric": 100.0, "SttlmPric": 100.0, "UndrlygPric": 25010.0, "TtlTradgVol": 10, "OpnIntrst": 500,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def importer(tmp_path):
    return BhavcopyImporter(data_dir=tmp_path, cache_dir=tmp_path / "bhav")


def test_sync_date_writes_spot_and_replaces_same_day(importer):
    with patch.object(importer, "download_bhavcopy", return_value=(_raw("2026-11-25"), "test")):
        importer.sync_date("2026-11-25", symbols=["NIFTY"])
        importer.sync_date("2026-11-25", symbols=["NIFTY"])
    spot = importer.store.load_spot("NIFTY")
    opts = importer.store.load_options("NIFTY")
    assert len(spot) == 1 and spot["close"].iloc[0] == 25010.0
    assert len(opts) == 3


def test_unreadable_existing_options_are_not_overwritten(importer):
    with patch.object(importer, "download_bhavcopy", return_value=(_raw("2026-11-24"), "test")):
        importer.sync_date("2026-11-24", symbols=["NIFTY"])
    with patch.object(importer, "download_bhavcopy", return_value=(_raw("2026-11-25"), "test")), \
         patch.object(importer.store, "load_options", side_effect=OSError("partial file")):
        importer.sync_date("2026-11-25", symbols=["NIFTY"])
    assert len(importer.store.load_options("NIFTY")) == 3


def test_sync_missing_days_backfills_weekdays_and_skips_holidays(importer):
    class FakeDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 27)

    def fake_download(d):
        if str(d) == "2026-11-26":
            raise FileNotFoundError
        return _raw(str(d)), "test"

    with patch.object(importer, "download_bhavcopy", side_effect=fake_download), \
         patch.object(importer, "sync_today_from_live_chain", return_value={"status": "failed"}), \
         patch("studytrade.data.bhavcopy.datetime") as dt_mock:
        dt_mock.now.return_value.date.return_value = FakeDate(2026, 11, 27)
        res = importer.sync_missing_days(symbols=["NIFTY"], max_lookback_days=3)
    assert res["synced"] == ["2026-11-24", "2026-11-25", "2026-11-27"]
    assert res["skipped"] == ["2026-11-26"]
