"""
StudyTrade — Vectorized Analytical Greeks
===========================================

Vectorized analytical calculation of primary and second-order Greeks:
Delta, Gamma, Vega, Theta, Rho, Vanna, and Volga.

Includes a numerical finite-difference engine to validate analytical formulas.

Mathematical Reference:
-----------------------
d1 = (ln(S/K) + (r - q + 0.5 * sigma^2)*T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

Delta:
  Call Delta = exp(-q*T) * N(d1)
  Put Delta  = -exp(-q*T) * N(-d1) = Call Delta - exp(-q*T)

Gamma:
  Gamma = exp(-q*T) * n(d1) / (S * sigma * sqrt(T))  (same for Call and Put)

Vega:
  Vega = S * exp(-q*T) * sqrt(T) * n(d1)             (same for Call and Put)
  Vega 1% = Vega / 100

Theta (annualized):
  Call Theta = -S*exp(-q*T)*n(d1)*sigma / (2*sqrt(T)) + q*S*exp(-q*T)*N(d1) - r*K*exp(-r*T)*N(d2)
  Put Theta  = -S*exp(-q*T)*n(d1)*sigma / (2*sqrt(T)) - q*S*exp(-q*T)*N(-d1) + r*K*exp(-r*T)*N(-d2)
  Daily Theta = Theta / 252 (or / 365)

Rho:
  Call Rho = K * T * exp(-r*T) * N(d2)
  Put Rho  = -K * T * exp(-r*T) * N(-d2)

Vanna = -exp(-q*T) * n(d1) * d2 / sigma
Volga = Vega * d1 * d2 / sigma
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from studytrade.constants import TRADING_DAYS_PER_YEAR, OptionType
from studytrade.math.bsm import _norm_cdf, _norm_pdf, bsm_d1_d2, bsm_price


@dataclass(frozen=True)
class GreeksResult:
    """Container for option Greeks."""

    delta: float | np.ndarray
    gamma: float | np.ndarray
    vega: float | np.ndarray  # Per 1.0 (100%) IV change
    vega_1pct: float | np.ndarray  # Per 0.01 (1%) IV change
    theta: float | np.ndarray  # Annualized decay
    theta_daily: float | np.ndarray  # 1-day decay (T / 252)
    rho: float | np.ndarray  # Per 1.0 (100%) rate change
    rho_1pct: float | np.ndarray  # Per 0.01 (1%) rate change
    vanna: float | np.ndarray
    volga: float | np.ndarray


def greeks(
    option_type: OptionType | str,
    s: np.ndarray | float,
    k: np.ndarray | float,
    t: np.ndarray | float,
    r: np.ndarray | float,
    sigma: np.ndarray | float,
    q: np.ndarray | float = 0.0,
    calendar_days: bool = False,
) -> GreeksResult:
    """
    Calculate all primary and secondary analytical Greeks.

    Parameters
    ----------
    option_type : OptionType or str
        'CE' or 'PE'.
    s, k, t, r, sigma, q : float or array-like
        Standard BSM parameters.
    calendar_days : bool, default False
        If True, daily theta divides annualized theta by 365; else by 252.

    Returns
    -------
    GreeksResult
        Dataclass containing delta, gamma, vega, theta, rho, vanna, volga.
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

    d1, d2 = bsm_d1_d2(s_arr, k_arr, t_arr, r_arr, sigma_arr, q_arr)

    df_r = np.exp(-r_arr * t_arr)
    df_q = np.exp(-q_arr * t_arr)
    sqrt_t = np.sqrt(np.maximum(t_arr, 1e-12))
    pdf_d1 = _norm_pdf(d1)

    # Delta
    delta = df_q * _norm_cdf(d1) if is_call else -df_q * _norm_cdf(-d1)

    # Gamma (identical for Call and Put)
    gamma = (df_q * pdf_d1) / (s_arr * sigma_arr * sqrt_t)

    # Vega (identical for Call and Put)
    vega = s_arr * df_q * sqrt_t * pdf_d1
    vega_1pct = vega / 100.0

    # Theta (annualized)
    common_theta_term = -(s_arr * df_q * pdf_d1 * sigma_arr) / (2.0 * sqrt_t)
    if is_call:
        theta = (
            common_theta_term
            + q_arr * s_arr * df_q * _norm_cdf(d1)
            - r_arr * k_arr * df_r * _norm_cdf(d2)
        )
    else:
        theta = (
            common_theta_term
            - q_arr * s_arr * df_q * _norm_cdf(-d1)
            + r_arr * k_arr * df_r * _norm_cdf(-d2)
        )

    days_in_year = 365.0 if calendar_days else float(TRADING_DAYS_PER_YEAR)
    theta_daily = theta / days_in_year

    # Rho
    if is_call:
        rho = k_arr * t_arr * df_r * _norm_cdf(d2)
    else:
        rho = -k_arr * t_arr * df_r * _norm_cdf(-d2)
    rho_1pct = rho / 100.0

    # Second-order Greeks: Vanna and Volga
    vanna = -df_q * pdf_d1 * (d2 / np.maximum(sigma_arr, 1e-12))
    volga = vega * (d1 * d2) / np.maximum(sigma_arr, 1e-12)

    # Handle expired options (T <= 0)
    expired_mask = t_arr <= 0.0
    if np.any(expired_mask):
        if is_call:
            delta = np.where(expired_mask, np.where(s_arr > k_arr, 1.0, 0.0), delta)
        else:
            delta = np.where(expired_mask, np.where(s_arr < k_arr, -1.0, 0.0), delta)
        gamma = np.where(expired_mask, 0.0, gamma)
        vega = np.where(expired_mask, 0.0, vega)
        vega_1pct = np.where(expired_mask, 0.0, vega_1pct)
        theta = np.where(expired_mask, 0.0, theta)
        theta_daily = np.where(expired_mask, 0.0, theta_daily)
        rho = np.where(expired_mask, 0.0, rho)
        rho_1pct = np.where(expired_mask, 0.0, rho_1pct)
        vanna = np.where(expired_mask, 0.0, vanna)
        volga = np.where(expired_mask, 0.0, volga)

    if is_scalar:
        return GreeksResult(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            vega_1pct=float(vega_1pct),
            theta=float(theta),
            theta_daily=float(theta_daily),
            rho=float(rho),
            rho_1pct=float(rho_1pct),
            vanna=float(vanna),
            volga=float(volga),
        )

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=vega,
        vega_1pct=vega_1pct,
        theta=theta,
        theta_daily=theta_daily,
        rho=rho,
        rho_1pct=rho_1pct,
        vanna=vanna,
        volga=volga,
    )


def numerical_greeks(
    option_type: OptionType | str,
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    eps_s: float = 0.01,
    eps_sigma: float = 1e-4,
    eps_t: float = 1e-5,
    eps_r: float = 1e-4,
) -> dict[str, float]:
    """
    Calculate Greeks using central finite differences for validation.

    Delta: [V(S + eps) - V(S - eps)] / (2 * eps)
    Gamma: [V(S + eps) - 2*V(S) + V(S - eps)] / (eps^2)
    Vega:  [V(sigma + eps) - V(sigma - eps)] / (2 * eps)
    Theta: -[V(T + eps) - V(T - eps)] / (2 * eps)
    Rho:   [V(r + eps) - V(r - eps)] / (2 * eps)
    """
    v = bsm_price(option_type, s, k, t, r, sigma, q)

    # Delta & Gamma
    v_s_up = bsm_price(option_type, s + eps_s, k, t, r, sigma, q)
    v_s_down = bsm_price(option_type, s - eps_s, k, t, r, sigma, q)
    delta_num = (v_s_up - v_s_down) / (2.0 * eps_s)
    gamma_num = (v_s_up - 2.0 * v + v_s_down) / (eps_s**2)

    # Vega
    v_vol_up = bsm_price(option_type, s, k, t, r, sigma + eps_sigma, q)
    v_vol_down = bsm_price(option_type, s, k, t, r, sigma - eps_sigma, q)
    vega_num = (v_vol_up - v_vol_down) / (2.0 * eps_sigma)

    # Theta: partial w.r.t time to expiry (dT = -dt)
    v_t_up = bsm_price(option_type, s, k, t + eps_t, r, sigma, q)
    v_t_down = bsm_price(option_type, s, k, max(1e-10, t - eps_t), r, sigma, q)
    theta_num = -(v_t_up - v_t_down) / (2.0 * eps_t)

    # Rho
    v_r_up = bsm_price(option_type, s, k, t, r + eps_r, sigma, q)
    v_r_down = bsm_price(option_type, s, k, t, r - eps_r, sigma, q)
    rho_num = (v_r_up - v_r_down) / (2.0 * eps_r)

    return {
        "delta": float(delta_num),
        "gamma": float(gamma_num),
        "vega": float(vega_num),
        "theta": float(theta_num),
        "rho": float(rho_num),
    }
