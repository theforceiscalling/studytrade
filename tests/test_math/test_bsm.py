"""
Tests for studytrade.math.bsm — Vectorized Black-Scholes-Merton pricing.
"""

from __future__ import annotations

import numpy as np

from studytrade.constants import OptionType
from studytrade.math.bsm import bsm_price, check_put_call_parity


class TestBSMPricing:
    """Validate BSM analytical option prices against standard benchmarks."""

    def test_textbook_call_put_prices(self):
        """
        Classic benchmark:
          S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.20, q = 0
          Theoretical Call: ~10.4506
          Theoretical Put:  ~5.5735
        """
        s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        call = bsm_price(OptionType.CALL, s, k, t, r, sigma)
        put = bsm_price(OptionType.PUT, s, k, t, r, sigma)

        assert abs(call - 10.45058) < 1e-4
        assert abs(put - 5.57353) < 1e-4

    def test_put_call_parity(self):
        """Verify Put-Call Parity across a grid of parameters."""
        s = 21500.0
        k = 21600.0
        t = 15.0 / 252.0
        r = 0.065
        sigma = 0.14
        q = 0.012

        call = bsm_price(OptionType.CALL, s, k, t, r, sigma, q)
        put = bsm_price(OptionType.PUT, s, k, t, r, sigma, q)

        is_valid, diff = check_put_call_parity(call, put, s, k, t, r, q)
        assert is_valid
        assert diff < 1e-6

    def test_vectorization_across_strikes(self):
        """Verify vectorization over NumPy arrays."""
        s = 21000.0
        strikes = np.array([20500.0, 21000.0, 21500.0])
        t = 10.0 / 252.0
        r = 0.07
        sigma = 0.15

        calls = bsm_price(OptionType.CALL, s, strikes, t, r, sigma)
        assert isinstance(calls, np.ndarray)
        assert len(calls) == 3
        # ITM call > ATM call > OTM call
        assert calls[0] > calls[1] > calls[2]

        puts = bsm_price(OptionType.PUT, s, strikes, t, r, sigma)
        # OTM put < ATM put < ITM put
        assert puts[0] < puts[1] < puts[2]

    def test_expired_options(self):
        """At T=0, option price must equal intrinsic value."""
        s = 21500.0
        call_itm = bsm_price(OptionType.CALL, s, 21400.0, 0.0, 0.07, 0.15)
        call_otm = bsm_price(OptionType.CALL, s, 21600.0, 0.0, 0.07, 0.15)

        assert call_itm == 100.0
        assert call_otm == 0.0
