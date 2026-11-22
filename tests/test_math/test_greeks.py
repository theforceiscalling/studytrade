"""
Tests for studytrade.math.greeks — Analytical vs Numerical Greeks validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from studytrade.constants import OptionType
from studytrade.math.greeks import greeks, numerical_greeks


class TestGreeksValidation:
    """Validate analytical Greeks against central finite difference approximations."""

    @pytest.mark.parametrize("opt_type", [OptionType.CALL, OptionType.PUT])
    def test_analytical_vs_numerical_match(self, opt_type):
        s = 21500.0
        k = 21450.0
        t = 20.0 / 252.0
        r = 0.07
        sigma = 0.16
        q = 0.01

        ana = greeks(opt_type, s, k, t, r, sigma, q)
        num = numerical_greeks(opt_type, s, k, t, r, sigma, q)

        # Delta match within 0.1%
        assert abs(ana.delta - num["delta"]) < 1e-3
        # Gamma match within 0.5%
        assert abs(ana.gamma - num["gamma"]) < 1e-4
        # Vega match within 0.1%
        assert abs(ana.vega - num["vega"]) < 1e-2
        # Theta match within 0.5%
        assert abs(ana.theta - num["theta"]) < 1.0
        # Rho match within 0.1%
        assert abs(ana.rho - num["rho"]) < 1e-2

    def test_greek_structural_properties(self):
        s, k, t, r, sigma = 21000.0, 21000.0, 30.0 / 252.0, 0.07, 0.15

        call_g = greeks(OptionType.CALL, s, k, t, r, sigma)
        put_g = greeks(OptionType.PUT, s, k, t, r, sigma)

        # Delta ranges
        assert 0.0 <= call_g.delta <= 1.0
        assert -1.0 <= put_g.delta <= 0.0
        # Call delta - Put delta = exp(-q*T) = 1.0 (since q=0)
        assert abs((call_g.delta - put_g.delta) - 1.0) < 1e-5

        # Gamma & Vega must be strictly positive and identical for Call and Put
        assert call_g.gamma > 0.0
        assert abs(call_g.gamma - put_g.gamma) < 1e-8
        assert call_g.vega > 0.0
        assert abs(call_g.vega - put_g.vega) < 1e-8

        # Long options suffer negative theta decay
        assert call_g.theta_daily < 0.0
        assert put_g.theta_daily < 0.0

    def test_vectorized_greeks(self):
        strikes = np.array([20000.0, 21000.0, 22000.0])
        res = greeks(OptionType.CALL, 21000.0, strikes, 0.1, 0.07, 0.16)
        assert isinstance(res.delta, np.ndarray)
        assert len(res.delta) == 3
        # ITM delta > ATM delta > OTM delta
        assert res.delta[0] > res.delta[1] > res.delta[2]
