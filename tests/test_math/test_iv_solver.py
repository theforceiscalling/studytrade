"""
Tests for studytrade.math.iv_solver — Implied volatility inversion.
"""

from __future__ import annotations

import numpy as np
import pytest

from studytrade.constants import OptionType
from studytrade.math.bsm import bsm_price
from studytrade.math.iv_solver import implied_volatility


class TestIVSolver:
    """Validate IV inversion accuracy across strike moneyness."""

    @pytest.mark.parametrize("true_vol", [0.12, 0.18, 0.25, 0.45])
    @pytest.mark.parametrize("opt_type", [OptionType.CALL, OptionType.PUT])
    def test_roundtrip_iv_inversion(self, true_vol, opt_type):
        s = 21500.0
        k = 21600.0
        t = 25.0 / 252.0
        r = 0.07
        q = 0.01

        # Generate theoretical price
        price = bsm_price(opt_type, s, k, t, r, true_vol, q)

        # Invert to recover IV
        solved_iv = implied_volatility(opt_type, price, s, k, t, r, q)

        # Must recover true volatility within 1e-4
        assert abs(solved_iv - true_vol) < 1e-4

    def test_arbitrage_bounds_violation_returns_nan(self):
        """Prices below intrinsic or above upper bound must return NaN without crashing."""
        s = 21500.0
        k = 21000.0
        t = 10.0 / 252.0
        r = 0.07

        # For a call with S=21500, K=21000, intrinsic is > 500.
        # A market price of 100 violates no-arbitrage lower bound.
        iv = implied_volatility(OptionType.CALL, 100.0, s, k, t, r)
        assert np.isnan(iv)

    def test_vectorized_iv_solver(self):
        s = 21000.0
        strikes = np.array([20800.0, 21000.0, 21200.0])
        t = 15.0 / 252.0
        r = 0.07
        true_vols = np.array([0.18, 0.16, 0.15])

        prices = bsm_price(OptionType.CALL, s, strikes, t, r, true_vols)
        solved_vols = implied_volatility(OptionType.CALL, prices, s, strikes, t, r)

        assert isinstance(solved_vols, np.ndarray)
        np.testing.assert_allclose(solved_vols, true_vols, atol=1e-4)
