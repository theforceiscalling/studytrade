"""
StudyTrade — Robust Implied Volatility (IV) Solver
===================================================

Fast, vectorized Implied Volatility solver combining Newton-Raphson
with an automatic bisection/Brent fallback for deep OTM or near-expiry options.

Mathematical Formulation:
-------------------------
Find sigma such that:
  f(sigma) = V_BSM(sigma) - P_market = 0

1. Arbitrage Bounds Check:
   Call Lower Bound: max(0, S*exp(-q*T) - K*exp(-r*T))
   Call Upper Bound: S*exp(-q*T)
   Put Lower Bound:  max(0, K*exp(-r*T) - S*exp(-q*T))
   Put Upper Bound:  K*exp(-r*T)

   If P_market is outside bounds, IV is mathematically undefined -> returns NaN.

2. Initial Guess: Brenner-Subrahmanyam (1988) approximation:
   sigma_0 = sqrt(2 * pi / T) * (P_market / S)

3. Newton-Raphson:
   sigma_{n+1} = sigma_n - (V_BSM(sigma_n) - P_market) / Vega(sigma_n)

4. Fallback:
   When Vega < 1e-7 or Newton-Raphson step exits [sigma_min, sigma_max],
   fall back to bounded Brent / bisection method on [0.001, 5.00].
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from studytrade.constants import OptionType
from studytrade.math.bsm import bsm_d1_d2, bsm_price


def _iv_single(
    option_type: OptionType | str,
    price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 50,
    sigma_min: float = 0.001,
    sigma_max: float = 5.0,
) -> float:
    """Solve IV for a single contract."""
    if np.isnan(price) or price <= 0.0 or t <= 1e-6:
        return float("nan")

    opt_type_str = (
        option_type.value
        if isinstance(option_type, OptionType)
        else str(option_type).upper().strip()
    )
    is_call = opt_type_str in ("CE", "CALL", "C")

    # Check arbitrage bounds
    df_r = np.exp(-r * t)
    df_q = np.exp(-q * t)

    if is_call:
        intrinsic = max(0.0, s * df_q - k * df_r)
        upper_bound = s * df_q
    else:
        intrinsic = max(0.0, k * df_r - s * df_q)
        upper_bound = k * df_r

    if price <= intrinsic or price >= upper_bound:
        return float("nan")

    # Initial guess via Brenner-Subrahmanyam
    sigma = np.clip(np.sqrt(2.0 * np.pi / t) * (price / s), 0.05, 1.5)

    # Newton-Raphson loop
    for _ in range(max_iter):
        p_est = bsm_price(opt_type_str, s, k, t, r, sigma, q)
        diff = p_est - price

        if abs(diff) < tol:
            return float(sigma)

        # Analytical Vega
        d1, _ = bsm_d1_d2(s, k, t, r, sigma, q)
        vega = s * df_q * np.sqrt(t) * (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (d1**2))

        # If vega is too small, Newton-Raphson diverges -> switch to Brent
        if vega < 1e-7:
            break

        step = diff / vega
        sigma_next = sigma - step

        if sigma_next < sigma_min or sigma_next > sigma_max:
            break

        if abs(sigma_next - sigma) < tol:
            return float(sigma_next)

        sigma = sigma_next

    # Bounded fallback using Brent's method
    def objective(vol: float) -> float:
        return float(bsm_price(opt_type_str, s, k, t, r, vol, q) - price)

    try:
        f_min = objective(sigma_min)
        f_max = objective(sigma_max)
        if f_min * f_max <= 0:
            return float(brentq(objective, sigma_min, sigma_max, xtol=tol))
    except (ValueError, RuntimeError):
        pass

    return float("nan")


def implied_volatility(
    option_type: OptionType | str | np.ndarray,
    price: np.ndarray | float,
    s: np.ndarray | float,
    k: np.ndarray | float,
    t: np.ndarray | float,
    r: np.ndarray | float,
    q: np.ndarray | float = 0.0,
    tol: float = 1e-6,
) -> float | np.ndarray:
    """
    Vectorized calculation of Black-Scholes implied volatility.

    Parameters
    ----------
    option_type : OptionType, str, or array-like
        Option contract type(s).
    price : float or array-like
        Market price of option(s).
    s, k, t, r, q : float or array-like
        BSM market parameters.

    Returns
    -------
    float or np.ndarray
        Implied volatility annualized (e.g. 0.165 for 16.5%).
    """
    if np.isscalar(price) and np.isscalar(s) and np.isscalar(k) and np.isscalar(t):
        return _iv_single(
            option_type=option_type,  # type: ignore
            price=float(price),
            s=float(s),
            k=float(k),
            t=float(t),
            r=float(r) if np.isscalar(r) else float(r[0]),
            q=float(q) if np.isscalar(q) else float(q[0]),
            tol=tol,
        )

    # Vectorized execution over arrays
    p_arr = np.asarray(price, dtype=np.float64)
    s_arr = np.broadcast_to(np.asarray(s, dtype=np.float64), p_arr.shape)
    k_arr = np.broadcast_to(np.asarray(k, dtype=np.float64), p_arr.shape)
    t_arr = np.broadcast_to(np.asarray(t, dtype=np.float64), p_arr.shape)
    r_arr = np.broadcast_to(np.asarray(r, dtype=np.float64), p_arr.shape)
    q_arr = np.broadcast_to(np.asarray(q, dtype=np.float64), p_arr.shape)

    if isinstance(option_type, (str, OptionType)):
        types = [option_type] * p_arr.size
    else:
        types = list(option_type)

    results = np.empty(p_arr.size, dtype=np.float64)

    for i in range(p_arr.size):
        results[i] = _iv_single(
            option_type=types[i],
            price=p_arr.flat[i],
            s=s_arr.flat[i],
            k=k_arr.flat[i],
            t=t_arr.flat[i],
            r=r_arr.flat[i],
            q=q_arr.flat[i],
            tol=tol,
        )

    return results.reshape(p_arr.shape)
