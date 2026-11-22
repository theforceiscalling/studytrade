"""
StudyTrade — Options Gamma Exposure (GEX) and Gamma Squeeze Analytics
======================================================================

Calculates contract-level and market-wide Gamma Exposure (GEX), the Gamma Flip
(Zero Gamma) boundary, Call and Put Walls, and quantitative Gamma Squeeze
vulnerability scores for NSE India index and equity options.

Mathematical Reference:
-----------------------
1. Analytical Gamma (BSM):
   Gamma = exp(-q * T) * n(d1) / (S * sigma * sqrt(T))

2. Rupee Gamma Exposure (GEX):
   GEX_call = +Gamma_call * OI_call * S^2 * 0.01
   GEX_put  = -Gamma_put  * OI_put  * S^2 * 0.01
   Net_GEX  = GEX_call + GEX_put

   Units: INR of underlying delta-hedging required per 1% change in spot price.

3. Gamma Flip Level:
   The spot price S* where aggregate Net GEX transitions across zero:
   Net_GEX(S*) = 0.
   - S > S*: Positive Gamma regime (volatility dampening, mean-reverting).
   - S < S*: Negative Gamma regime (volatility accelerating, cascade-prone).
   - S approaching Call Wall with high call volume: Gamma Squeeze trigger zone.

4. Gamma Squeeze / Cascade Vulnerability Index (0-100):
   Composite of:
   - Proximity to Call Wall (Squeeze) or Put Wall (Cascade).
   - Volume / OI surge ratio (aggressive opening flow).
   - Expiry proximity / DTE decay factor (concentrated near expiry).
   - Volatility expansion (IV surge confirming directional demand).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
import pandas as pd

from studytrade.constants import (
    DEFAULT_RISK_FREE_RATE,
    INSTRUMENTS,
    TRADING_DAYS_PER_YEAR,
    OptionType,
)
from studytrade.math.bsm import _norm_pdf, bsm_d1_d2


@dataclass(frozen=True)
class GEXProfile:
    """
    Complete snapshot of market Gamma Exposure and Squeeze metrics.
    """

    spot_price: float
    total_call_gex: float
    total_put_gex: float
    net_gex: float
    rupee_gex_per_1pct: float
    call_wall_strike: float
    put_wall_strike: float
    max_gamma_strike: float
    gamma_flip_level: float
    call_squeeze_score: float  # 0.0 to 100.0 (Bullish Squeeze Vulnerability)
    put_cascade_score: float   # 0.0 to 100.0 (Bearish Cascade Vulnerability)
    strike_gex_df: pd.DataFrame


def compute_option_gamma(
    chain_df: pd.DataFrame,
    spot_price: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> pd.Series:
    """
    Compute analytical Black-Scholes Gamma for each contract in the chain.

    Parameters
    ----------
    chain_df : pd.DataFrame
        Option chain containing 'strike', 'option_type', 'implied_volatility',
        and optionally 'expiry' / 'timestamp'.
    spot_price : float
        Current underlying spot price.
    risk_free_rate : float, default 0.07
        Annualized risk-free interest rate.

    Returns
    -------
    pd.Series
        Analytical Gamma values per contract.
    """
    if chain_df.empty or spot_price <= 0.0:
        return pd.Series(dtype=np.float64)

    # Calculate time to expiration (T in years) using calendar annualization for calendar DTE
    if "expiry" in chain_df.columns and "timestamp" in chain_df.columns:
        exp_dt = pd.to_datetime(chain_df["expiry"])
        cur_dt = pd.to_datetime(chain_df["timestamp"])
        dte_days = (exp_dt.dt.date - cur_dt.dt.date).apply(lambda d: max(0, d.days))
        t_years = np.maximum(dte_days.values / 365.0, 1.0 / (365.0 * 6.25 * 60))
    elif "dte" in chain_df.columns:
        t_years = np.maximum(chain_df["dte"].values / 365.0, 1e-4)
    else:
        # Default fallback to 5 calendar days if no expiry info
        t_years = np.full(len(chain_df), 5.0 / 365.0, dtype=np.float64)

    strikes = chain_df["strike"].values.astype(np.float64)

    # Use implied_volatility if available, bounded to realistic range
    if "implied_volatility" in chain_df.columns:
        iv = np.clip(chain_df["implied_volatility"].values.astype(np.float64), 0.05, 3.0)
    else:
        iv = np.full(len(chain_df), 0.15, dtype=np.float64)

    r_arr = np.full(len(chain_df), risk_free_rate, dtype=np.float64)
    q_arr = np.zeros(len(chain_df), dtype=np.float64)
    s_arr = np.full(len(chain_df), spot_price, dtype=np.float64)

    d1, _ = bsm_d1_d2(s_arr, strikes, t_years, r_arr, iv, q_arr)
    pdf_d1 = _norm_pdf(d1)
    sqrt_t = np.sqrt(t_years)

    gamma = pdf_d1 / (s_arr * iv * sqrt_t)
    return pd.Series(gamma, index=chain_df.index, name="gamma")


def compute_gex_profile(
    chain_df: pd.DataFrame,
    spot_price: float,
    lot_size: int = 50,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> GEXProfile:
    """
    Calculate full Gamma Exposure (GEX) profile across strikes for an option chain snapshot.

    Parameters
    ----------
    chain_df : pd.DataFrame
        Option chain snapshot containing:
        'strike', 'option_type', 'open_interest', 'volume', 'implied_volatility'.
    spot_price : float
        Current spot price of the underlying.
    lot_size : int, default 50
        Underlying contract lot multiplier (e.g. 50 for NIFTY, 15 for BANKNIFTY).
    risk_free_rate : float, default 0.07
        Risk free interest rate.

    Returns
    -------
    GEXProfile
        Dataclass containing total GEX, Wall strikes, Gamma Flip, and Squeeze scores.
    """
    empty_profile = GEXProfile(
        spot_price=spot_price,
        total_call_gex=0.0,
        total_put_gex=0.0,
        net_gex=0.0,
        rupee_gex_per_1pct=0.0,
        call_wall_strike=spot_price,
        put_wall_strike=spot_price,
        max_gamma_strike=spot_price,
        gamma_flip_level=spot_price,
        call_squeeze_score=0.0,
        put_cascade_score=0.0,
        strike_gex_df=pd.DataFrame(),
    )

    if chain_df.empty or spot_price <= 0.0:
        return empty_profile

    df = chain_df.copy()

    # Normalize option types
    opt_types = df["option_type"].astype(str).str.upper().str.strip()
    is_call = opt_types.isin(["CE", "CALL", "C"])
    is_put = opt_types.isin(["PE", "PUT", "P"])

    # Ensure required columns
    if "open_interest" not in df.columns:
        df["open_interest"] = 0
    if "volume" not in df.columns:
        df["volume"] = 0

    oi = df["open_interest"].fillna(0).values.astype(np.float64)
    vol = df["volume"].fillna(0).values.astype(np.float64)

    # Compute analytical gamma if not provided
    if "gamma" not in df.columns:
        df["gamma"] = compute_option_gamma(df, spot_price, risk_free_rate=risk_free_rate)
    gamma = df["gamma"].fillna(0.0).values.astype(np.float64)

    # Rupee GEX per contract for a 1% move in spot:
    # GEX = Gamma * OI * Spot^2 * 0.01
    contract_gex_1pct = gamma * oi * (spot_price**2) * 0.01

    # Standard Dealer Positioning Convention:
    # - Call GEX is positive (dealers assumed short calls when customers buy; upward spot forces dealer buying).
    # - Put GEX is negative (dealers assumed short puts; downward spot forces dealer selling).
    call_gex_arr = np.where(is_call, contract_gex_1pct, 0.0)
    put_gex_arr = np.where(is_put, -contract_gex_1pct, 0.0)

    df["call_gex"] = call_gex_arr
    df["put_gex"] = put_gex_arr
    df["net_gex"] = call_gex_arr + put_gex_arr

    # Aggregate by Strike
    strike_agg = (
        df.groupby("strike")
        .agg(
            call_oi=("open_interest", lambda s: int(s[is_call.loc[s.index]].sum())),
            put_oi=("open_interest", lambda s: int(s[is_put.loc[s.index]].sum())),
            call_volume=("volume", lambda s: int(s[is_call.loc[s.index]].sum())),
            put_volume=("volume", lambda s: int(s[is_put.loc[s.index]].sum())),
            call_gex=("call_gex", "sum"),
            put_gex=("put_gex", "sum"),
            net_gex=("net_gex", "sum"),
        )
        .reset_index()
    )

    if strike_agg.empty:
        return empty_profile

    strike_agg["total_oi"] = strike_agg["call_oi"] + strike_agg["put_oi"]
    strike_agg["total_abs_gex"] = strike_agg["call_gex"].abs() + strike_agg["put_gex"].abs()

    total_call_gex = float(strike_agg["call_gex"].sum())
    total_put_gex = float(strike_agg["put_gex"].sum())
    net_gex = total_call_gex + total_put_gex

    # Rupee Delta-hedging flow required per 1% spot move
    rupee_gex_per_1pct = net_gex

    # Identify Key Walls
    # Call Wall: Strike with maximum Call GEX (or Call OI if GEX is flat)
    if (strike_agg["call_gex"] > 0).any():
        call_wall_idx = strike_agg["call_gex"].idxmax()
    else:
        call_wall_idx = strike_agg["call_oi"].idxmax()
    call_wall = float(strike_agg.loc[call_wall_idx, "strike"])

    # Put Wall: Strike with largest absolute Put GEX (or Put OI)
    if (strike_agg["put_gex"] < 0).any():
        put_wall_idx = strike_agg["put_gex"].abs().idxmax()
    else:
        put_wall_idx = strike_agg["put_oi"].idxmax()
    put_wall = float(strike_agg.loc[put_wall_idx, "strike"])

    # Max Gamma Strike: Strike with highest combined absolute GEX
    max_gamma_idx = strike_agg["total_abs_gex"].idxmax()
    max_gamma_strike = float(strike_agg.loc[max_gamma_idx, "strike"])

    # Calculate Gamma Flip level (Zero-Gamma point)
    gamma_flip_level = _calculate_gamma_flip(strike_agg, spot_price)

    # Compute quantitative Squeeze & Cascade scores
    call_score, put_score = _compute_squeeze_scores(
        df=df,
        strike_agg=strike_agg,
        spot_price=spot_price,
        call_wall=call_wall,
        put_wall=put_wall,
        gamma_flip=gamma_flip_level,
        net_gex=net_gex,
    )

    return GEXProfile(
        spot_price=spot_price,
        total_call_gex=total_call_gex,
        total_put_gex=total_put_gex,
        net_gex=net_gex,
        rupee_gex_per_1pct=rupee_gex_per_1pct,
        call_wall_strike=call_wall,
        put_wall_strike=put_wall,
        max_gamma_strike=max_gamma_strike,
        gamma_flip_level=gamma_flip_level,
        call_squeeze_score=call_score,
        put_cascade_score=put_score,
        strike_gex_df=strike_agg.sort_values("strike"),
    )


def _calculate_gamma_flip(strike_agg: pd.DataFrame, current_spot: float) -> float:
    """
    Find the price level where cumulative Net GEX transitions across zero.

    Interpolates along the strike curve sorted by strike.
    """
    if len(strike_agg) < 2:
        return current_spot

    sorted_strikes = strike_agg.sort_values("strike").reset_index(drop=True)
    strikes = sorted_strikes["strike"].values
    net_gex = sorted_strikes["net_gex"].values

    # Check for sign change between adjacent strikes
    sign_changes = np.where(np.diff(np.signbit(net_gex)))[0]

    if len(sign_changes) > 0:
        # Choose the sign change closest to current spot
        best_idx = sign_changes[np.argmin(np.abs(strikes[sign_changes] - current_spot))]
        s0, s1 = strikes[best_idx], strikes[best_idx + 1]
        g0, g1 = net_gex[best_idx], net_gex[best_idx + 1]

        # Linear interpolation for zero crossing
        if abs(g1 - g0) > 1e-9:
            flip = s0 - g0 * (s1 - s0) / (g1 - g0)
            return float(np.clip(flip, strikes.min(), strikes.max()))
        return float((s0 + s1) / 2.0)

    # Fallback: if no zero crossing, locate strike with minimum absolute Net GEX
    min_idx = np.argmin(np.abs(net_gex))
    return float(strikes[min_idx])


def _compute_squeeze_scores(
    df: pd.DataFrame,
    strike_agg: pd.DataFrame,
    spot_price: float,
    call_wall: float,
    put_wall: float,
    gamma_flip: float,
    net_gex: float,
) -> tuple[float, float]:
    """
    Compute Bullish Gamma Squeeze and Bearish Gamma Cascade vulnerability scores (0-100).
    """
    # 1. Proximity Factors (0.0 to 35.0 pts)
    # Bullish: closer to or above Call Wall increases squeeze probability
    call_dist_pct = (call_wall - spot_price) / max(spot_price, 1.0)
    if spot_price >= call_wall:
        # Penetrated resistance
        call_prox_score = 35.0
    elif 0.0 < call_dist_pct <= 0.0075:
        # Within 0.75% of Call Wall (~160 pts on Nifty 21500)
        call_prox_score = 35.0 * (1.0 - (call_dist_pct / 0.0075))
    else:
        call_prox_score = 0.0

    # Bearish: closer to or below Put Wall increases cascade probability
    put_dist_pct = (spot_price - put_wall) / max(spot_price, 1.0)
    if spot_price <= put_wall:
        # Penetrated support floor
        put_prox_score = 35.0
    elif 0.0 < put_dist_pct <= 0.0075:
        put_prox_score = 35.0 * (1.0 - (put_dist_pct / 0.0075))
    else:
        put_prox_score = 0.0

    # 2. Volume-to-OI Surge Factor at the Wall Strikes (0.0 to 30.0 pts)
    # Aggressive buying occurs when daily volume surpasses existing OI
    call_wall_row = strike_agg[strike_agg["strike"] == call_wall]
    if not call_wall_row.empty:
        c_oi = max(1, call_wall_row["call_oi"].iloc[0])
        c_vol = call_wall_row["call_volume"].iloc[0]
        c_vol_oi_ratio = c_vol / c_oi
        # Ratio >= 1.0 gives max score
        call_vol_score = float(np.clip(c_vol_oi_ratio * 30.0, 0.0, 30.0))
    else:
        call_vol_score = 0.0

    put_wall_row = strike_agg[strike_agg["strike"] == put_wall]
    if not put_wall_row.empty:
        p_oi = max(1, put_wall_row["put_oi"].iloc[0])
        p_vol = put_wall_row["put_volume"].iloc[0]
        p_vol_oi_ratio = p_vol / p_oi
        put_vol_score = float(np.clip(p_vol_oi_ratio * 30.0, 0.0, 30.0))
    else:
        put_vol_score = 0.0

    # 3. DTE Decay / Gamma Concentration Factor (0.0 to 20.0 pts)
    # Low DTE concentrates gamma into a narrow strike range, supercharging acceleration
    if "expiry" in df.columns and "timestamp" in df.columns:
        exp_dt = pd.to_datetime(df["expiry"].iloc[0])
        cur_dt = pd.to_datetime(df["timestamp"].iloc[0])
        dte = max(0, (exp_dt.date() - cur_dt.date()).days)
    elif "dte" in df.columns:
        dte = max(0, int(df["dte"].iloc[0]))
    else:
        dte = 5

    if dte <= 1:
        dte_score = 20.0
    elif dte <= 3:
        dte_score = 15.0
    elif dte <= 7:
        dte_score = 8.0
    else:
        dte_score = 3.0

    # 4. Net Gamma Regime Factor (0.0 to 15.0 pts)
    # Negative Gamma accelerates momentum; Positive Gamma dampens it
    # If spot > Gamma Flip, dealers are vulnerable to call squeezes
    if spot_price > gamma_flip or net_gex < 0:
        call_regime_score = 15.0
    else:
        call_regime_score = 5.0

    # If spot < Gamma Flip or net_gex < 0, dealers are vulnerable to put cascades
    if spot_price < gamma_flip or net_gex < 0:
        put_regime_score = 15.0
    else:
        put_regime_score = 5.0

    call_total_score = min(100.0, call_prox_score + call_vol_score + dte_score + call_regime_score)
    put_total_score = min(100.0, put_prox_score + put_vol_score + dte_score + put_regime_score)

    # Gatekeeper: A gamma squeeze/cascade requires proximity or penetration of the structural wall.
    # Without proximity, high volume or low DTE alone cannot force an immediate gamma squeeze.
    if call_prox_score == 0.0:
        call_total_score = min(call_total_score * 0.40, 40.0)

    if put_prox_score == 0.0:
        put_total_score = min(put_total_score * 0.40, 40.0)

    return round(float(call_total_score), 2), round(float(put_total_score), 2)
