"""
StudyTrade — Vectorized Black-Scholes-Merton (BSM) Pricing
============================================================

Vectorized, pure-NumPy implementation of the Black-Scholes-Merton (1973)
and Merton (1973) continuous-dividend option pricing models.

Mathematical Formulation:
-------------------------
Let:
  S = Spot price of underlying
  K = Strike price
  T = Time to expiration in trading years
  r = Annualized continuously compounded risk-free rate
  q = Annualized continuous dividend yield (0 for non-dividend paying / forward equivalence)
  sigma = Annualized volatility of underlying returns

d1 = (ln(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

European Call Price:
  C = S * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)

European Put Price:
  P = K * exp(-r * T) * N(-d2) - S * exp(-q * T) * N(-d1)

Put-Call Parity:
  C - P = S * exp(-q * T) - K * exp(-r * T)
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from studytrade.constants import OptionType


def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """
    Standard normal cumulative distribution function Phi(x).
    Using scipy.special.erf for high numerical precision and vectorization.
    """
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal probability density function phi(x)."""
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x**2))


def bsm_d1_d2(
    s: np.ndarray | float,
    k: np.ndarray | float,
    t: np.ndarray | float,
    r: np.ndarray | float,
    sigma: np.ndarray | float,
    q: np.ndarray | float = 0.0,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    Compute standard BSM d1 and d2 parameters.

    Returns (d1, d2).
    """
    s_arr = np.asarray(s, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)

    # Protect against T <= 0 or sigma <= 0 in division
    sqrt_t = np.sqrt(np.maximum(t_arr, 1e-12))
    vol_sqrt_t = np.maximum(sigma_arr * sqrt_t, 1e-12)

    d1 = (np.log(s_arr / k_arr) + (r_arr - q_arr + 0.5 * (sigma_arr**2)) * t_arr) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    return d1, d2


def bsm_price(
    option_type: OptionType | str,
    s: np.ndarray | float,
    k: np.ndarray | float,
    t: np.ndarray | float,
    r: np.ndarray | float,
    sigma: np.ndarray | float,
    q: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """
    Calculate theoretical European option price via Black-Scholes-Merton.

    Parameters
    ----------
    option_type : OptionType or str
        'CE' or 'PE' (call or put).
    s : float or array-like
        Spot price of underlying.
    k : float or array-like
        Strike price.
    t : float or array-like
        Time to maturity in years (e.g. 10 DTE -> 10 / 252).
    r : float or array-like
        Annualized risk-free interest rate (e.g. 0.07 for 7%).
    sigma : float or array-like
        Annualized volatility (e.g. 0.16 for 16%).
    q : float or array-like, default 0.0
        Annualized dividend yield.

    Returns
    -------
    float or np.ndarray
        Theoretical option price.
    """
    is_scalar = np.isscalar(s) and np.isscalar(k) and np.isscalar(t) and np.isscalar(sigma)

    s_arr = np.asarray(s, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)

    opt_type_str = (
        option_type.value
        if isinstance(option_type, OptionType)
        else str(option_type).upper().strip()
    )
    is_call = opt_type_str in ("CE", "CALL", "C")

    # Edge cases: expired options (T <= 0) or zero vol
    expired_mask = t_arr <= 0.0
    zero_vol_mask = sigma_arr <= 0.0

    d1, d2 = bsm_d1_d2(s_arr, k_arr, t_arr, r_arr, sigma_arr, q_arr)

    df_r = np.exp(-r_arr * t_arr)
    df_q = np.exp(-q_arr * t_arr)

    if is_call:
        price = s_arr * df_q * _norm_cdf(d1) - k_arr * df_r * _norm_cdf(d2)
        # Intrinsic values for expired or zero vol
        intrinsic = np.maximum(0.0, s_arr * df_q - k_arr * df_r)
    else:
        price = k_arr * df_r * _norm_cdf(-d2) - s_arr * df_q * _norm_cdf(-d1)
        intrinsic = np.maximum(0.0, k_arr * df_r - s_arr * df_q)

    # Apply edge case overrides
    override_mask = expired_mask | zero_vol_mask
    if np.any(override_mask):
        price = np.where(override_mask, intrinsic, price)

    # Floor at 0.0 (options have limited liability)
    price = np.maximum(0.0, price)

    return float(price) if is_scalar else price


def check_put_call_parity(
    call_price: np.ndarray | float,
    put_price: np.ndarray | float,
    s: np.ndarray | float,
    k: np.ndarray | float,
    t: np.ndarray | float,
    r: np.ndarray | float,
    q: np.ndarray | float = 0.0,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray | bool, np.ndarray | float]:
    """
    Verify European Put-Call Parity:
      C - P = S * exp(-q*T) - K * exp(-r*T)

    Returns (is_parity_satisfied, discrepancy).
    """
    s_arr = np.asarray(s, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)

    left_side = np.asarray(call_price, dtype=np.float64) - np.asarray(put_price, dtype=np.float64)
    right_side = s_arr * np.exp(-q_arr * t_arr) - k_arr * np.exp(-r_arr * t_arr)
    discrepancy = np.abs(left_side - right_side)

    is_valid = discrepancy <= tolerance
    if np.isscalar(call_price) and np.isscalar(put_price):
        return bool(is_valid), float(discrepancy)
    return is_valid, discrepancy
