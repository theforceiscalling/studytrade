"""
Tests for studytrade.data.fetchers.synthetic — Synthetic data generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from studytrade.constants import OptionType
from studytrade.data.fetchers.synthetic import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Test synthetic spot and option generation."""

    @pytest.fixture
    def gen(self):
        return SyntheticDataGenerator(seed=42)

    def test_spot_generation_basic(self, gen):
        df = gen.generate_spot(
            symbol="NIFTY", start_date="2024-01-01", n_days=50, spot_start=20000.0
        )
        assert len(df) == 50
        assert (df["open"] > 0).all()
        assert (df["close"] > 0).all()
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()
        assert (df["volume"] > 0).all()

    def test_spot_reproducibility(self):
        gen1 = SyntheticDataGenerator(seed=123)
        gen2 = SyntheticDataGenerator(seed=123)
        df1 = gen1.generate_spot(n_days=20)
        df2 = gen2.generate_spot(n_days=20)
        np.testing.assert_array_almost_equal(df1["close"].values, df2["close"].values)

    def test_option_chain_snapshot(self, gen):
        import pandas as pd

        ts = pd.to_datetime("2024-01-04 09:15:00").tz_localize("Asia/Kolkata")
        exp = pd.to_datetime("2024-01-11 15:30:00").tz_localize("Asia/Kolkata")

        chain = gen.generate_option_chain_snapshot(
            timestamp=ts,
            underlying_symbol="NIFTY",
            spot_price=21500.0,
            expiries=[exp],
            base_iv=0.15,
            num_strikes_each_side=5,
        )

        assert not chain.empty
        # Verify call and put pairs exist for each strike
        strikes = chain["strike"].unique()
        assert len(strikes) == 11  # 5 below + ATM + 5 above
        assert (chain["bid"] <= chain["ask"]).all()
        assert (chain["ask"] - chain["bid"] >= 0.05).all()

        # Check volatility skew: OTM put IV should be higher than OTM call IV
        otm_put = chain[
            (chain["strike"] == 21300.0) & (chain["option_type"] == OptionType.PUT.value)
        ]
        otm_call = chain[
            (chain["strike"] == 21700.0) & (chain["option_type"] == OptionType.CALL.value)
        ]
        assert not otm_put.empty and not otm_call.empty
        assert otm_put["implied_volatility"].iloc[0] > otm_call["implied_volatility"].iloc[0]

    def test_get_options_multi_day(self, gen):
        chain = gen.get_options(
            underlying_symbol="NIFTY",
            start_date="2024-01-01",
            end_date="2024-01-05",
        )
        assert not chain.empty
        assert set(chain["underlying_symbol"].unique()) == {"NIFTY"}
