"""
StudyTrade — Options Math Library
===================================

Vectorized Black-Scholes-Merton pricing, analytical Greeks,
implied volatility solver, and volatility surface analytics.
"""

from studytrade.math.bsm import bsm_d1_d2, bsm_price, check_put_call_parity
from studytrade.math.greeks import GreeksResult, greeks, numerical_greeks
from studytrade.math.iv_solver import implied_volatility
from studytrade.math.iv_surface import (
    SkewMetrics,
    calculate_iv_skew,
    iv_percentile,
    iv_rank,
    iv_rv_spread,
    realized_volatility_close,
    realized_volatility_parkinson,
)

__all__ = [
    "GreeksResult",
    "SkewMetrics",
    "bsm_d1_d2",
    "bsm_price",
    "calculate_iv_skew",
    "check_put_call_parity",
    "greeks",
    "implied_volatility",
    "iv_percentile",
    "iv_rank",
    "iv_rv_spread",
    "numerical_greeks",
    "realized_volatility_close",
    "realized_volatility_parkinson",
]
