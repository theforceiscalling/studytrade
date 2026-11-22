"""
Tests for studytrade.features.gamma_exposure — GEX, Walls, Flip Level, and Squeeze Scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from studytrade.constants import OptionType
from studytrade.features.gamma_exposure import (
    GEXProfile,
    compute_gex_profile,
    compute_option_gamma,
)


class TestGammaExposure:
    """Validate analytical Gamma Exposure and Squeeze vulnerability algorithms."""

    @pytest.fixture
    def sample_chain(self) -> pd.DataFrame:
        """Symmetric chain with spot at 21500."""
        strikes = [21000.0, 21200.0, 21400.0, 21500.0, 21600.0, 21800.0, 22000.0]
        rows = []
        for k in strikes:
            # Calls: heavy OI at 21800 (Call Wall)
            c_oi = 50000 if k == 21800.0 else (20000 if k == 21500.0 else 5000)
            c_vol = 60000 if k == 21800.0 else 2000
            rows.append(
                {
                    "strike": k,
                    "option_type": "CE",
                    "open_interest": c_oi,
                    "volume": c_vol,
                    "implied_volatility": 0.16,
                    "dte": 2,
                }
            )
            # Puts: heavy OI at 21200 (Put Wall)
            p_oi = 45000 if k == 21200.0 else (18000 if k == 21500.0 else 4000)
            p_vol = 50000 if k == 21200.0 else 1500
            rows.append(
                {
                    "strike": k,
                    "option_type": "PE",
                    "open_interest": p_oi,
                    "volume": p_vol,
                    "implied_volatility": 0.18,
                    "dte": 2,
                }
            )
        return pd.DataFrame(rows)

    def test_compute_option_gamma_properties(self, sample_chain):
        gammas = compute_option_gamma(sample_chain, spot_price=21500.0)
        assert len(gammas) == len(sample_chain)
        assert (gammas > 0.0).all()

        # ATM strike (21500) must have strictly higher gamma than deep OTM (21000 or 22000)
        atm_gamma = gammas[sample_chain["strike"] == 21500.0].iloc[0]
        otm_gamma = gammas[sample_chain["strike"] == 22000.0].iloc[0]
        assert atm_gamma > otm_gamma

    def test_compute_option_gamma_edge_cases(self):
        empty_res = compute_option_gamma(pd.DataFrame(), spot_price=21500.0)
        assert empty_res.empty

        zero_spot_res = compute_option_gamma(pd.DataFrame({"strike": [21500]}), spot_price=0.0)
        assert zero_spot_res.empty

    def test_gex_profile_walls_and_flip(self, sample_chain):
        profile = compute_gex_profile(sample_chain, spot_price=21500.0, lot_size=50)

        assert isinstance(profile, GEXProfile)
        assert profile.spot_price == 21500.0
        # Call Wall must identify the strike with massive call OI (21800)
        assert profile.call_wall_strike == 21800.0
        # Put Wall must identify the strike with massive put OI (21200)
        assert profile.put_wall_strike == 21200.0

        # Sign conventions
        assert profile.total_call_gex > 0.0
        assert profile.total_put_gex < 0.0
        np.testing.assert_allclose(
            profile.net_gex,
            profile.total_call_gex + profile.total_put_gex,
            rtol=1e-6,
        )

        # Gamma Flip should lie between Put Wall and Call Wall
        assert 21000.0 <= profile.gamma_flip_level <= 22000.0

    def test_empty_chain_gex_profile(self):
        profile = compute_gex_profile(pd.DataFrame(), spot_price=21500.0)
        assert profile.net_gex == 0.0
        assert profile.call_squeeze_score == 0.0
        assert profile.put_cascade_score == 0.0
        assert profile.strike_gex_df.empty

    def test_bullish_squeeze_score_activation(self, sample_chain):
        # When spot price approaches or crosses Call Wall (21800)
        profile_at_wall = compute_gex_profile(sample_chain, spot_price=21810.0, lot_size=50)
        assert profile_at_wall.call_squeeze_score >= 65.0

        # Far away from Call Wall (e.g. at 21300), squeeze score should be much lower
        profile_far = compute_gex_profile(sample_chain, spot_price=21300.0, lot_size=50)
        assert profile_far.call_squeeze_score < profile_at_wall.call_squeeze_score

    def test_bearish_cascade_score_activation(self, sample_chain):
        # When spot price breaks below Put Wall (21200)
        profile_breakdown = compute_gex_profile(sample_chain, spot_price=21180.0, lot_size=50)
        assert profile_breakdown.put_cascade_score >= 65.0

        profile_far = compute_gex_profile(sample_chain, spot_price=21700.0, lot_size=50)
        assert profile_far.put_cascade_score < profile_breakdown.put_cascade_score
