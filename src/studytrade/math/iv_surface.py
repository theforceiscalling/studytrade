"""
StudyTrade — Implied Volatility Surface, Skew, and Relative Value Analytics
============================================================================

Analytics for IV skew, term structure, IV Rank, IV Percentile,
and the Implied-Realized Volatility Spread (Volatility Risk Premium).

Mathematical Formulations:
--------------------------
1. Parkinson Realized Volatility (High-Low estimator):
   RV_park = sqrt( (252 / N) * sum( (ln(H_i / L_i))^2 / (4 * ln(2)) ) )

2. Close-to-Close Realized Volatility:
   RV_c2c = sqrt( (252 / N) * sum( (ln(C_i / C_{i-1}))^2 ) )

3. IV Rank (IVR):
   IVR = (IV_current - min(IV_lookback)) / (max(IV_lookback) - min(IV_lookback)) * 100

4. IV Percentile (IVP):
   IVP = (Count(IV_i < IV_current) / Total_days) * 100

5. Volatility Risk Premium (IV-RV Spread):
   Spread = IV_ATM - RV_20D
   (When positive: implied vol exceeds realized, options are expensive).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from studytrade.constants import TRADING_DAYS_PER_YEAR


def realized_volatility_close(
    close: pd.Series | np.ndarray,
    window: int = 20,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series | float:
    """
    Calculate annualized Close-to-Close Realized Volatility.
    """
    s = pd.Series(close) if not isinstance(close, pd.Series) else close
    log_ret = np.log(s / s.shift(1))
    rolling_vol = log_ret.rolling(window=window).std(ddof=1) * np.sqrt(annualization_factor)

    if len(s) == window:
        return float(rolling_vol.iloc[-1])
    return rolling_vol


def realized_volatility_parkinson(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    window: int = 20,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series | float:
    """
    Parkinson (1980) High-Low Realized Volatility estimator.
    More statistically efficient than close-to-close (lower variance).
    """
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l_ = pd.Series(low) if not isinstance(low, pd.Series) else low

    factor = 1.0 / (4.0 * np.log(2.0))
    hl_sq = factor * (np.log(h / l_) ** 2)
    rolling_var = hl_sq.rolling(window=window).mean()
    rolling_vol = np.sqrt(rolling_var * annualization_factor)

    if len(h) == window:
        return float(rolling_vol.iloc[-1])
    return rolling_vol


def iv_rank(current_iv: float, iv_history: pd.Series | np.ndarray) -> float:
    """
    Calculate Implied Volatility Rank (IVR):
      IVR = (IV_now - min_IV) / (max_IV - min_IV) * 100
    """
    s = np.asarray(iv_history, dtype=np.float64)
    valid = s[~np.isnan(s)]
    if len(valid) == 0:
        return float("nan")

    min_iv = np.min(valid)
    max_iv = np.max(valid)

    if max_iv <= min_iv:
        return 50.0

    rank = (current_iv - min_iv) / (max_iv - min_iv) * 100.0
    return float(np.clip(rank, 0.0, 100.0))


def iv_percentile(current_iv: float, iv_history: pd.Series | np.ndarray) -> float:
    """
    Calculate Implied Volatility Percentile (IVP):
      Percentage of days in lookback where historical IV was lower than current IV.
    """
    s = np.asarray(iv_history, dtype=np.float64)
    valid = s[~np.isnan(s)]
    if len(valid) == 0:
        return float("nan")

    pct = (np.sum(valid < current_iv) / len(valid)) * 100.0
    return float(pct)


def iv_rv_spread(atm_iv: float, realized_vol: float) -> float:
    """
    Calculate the Volatility Risk Premium (IV-RV spread):
      Spread = IV - RV
    Positive spread indicates options are trading at an implied volatility premium.
    Negative spread indicates options are cheap relative to actual realized volatility.
    """
    return float(atm_iv - realized_vol)


@dataclass(frozen=True)
class SkewMetrics:
    """Volatility skew and slope measurements."""

    atm_iv: float
    otm_put_iv: float  # ~25 Delta Put or strike below ATM
    otm_call_iv: float  # ~25 Delta Call or strike above ATM
    absolute_skew: float  # Put IV - Call IV
    normalized_skew: float  # (Put IV - Call IV) / ATM IV


def calculate_iv_skew(
    chain_df: pd.DataFrame,
    spot_price: float,
    strike_distance: float = 300.0,
) -> SkewMetrics:
    """
    Compute 25-delta / fixed-distance IV skew for a single expiry chain.
    """
    if chain_df.empty:
        return SkewMetrics(np.nan, np.nan, np.nan, np.nan, np.nan)

    # ATM strike
    atm_idx = (chain_df["strike"] - spot_price).abs().idxmin()
    atm_strike = chain_df.loc[atm_idx, "strike"]
    atm_iv = float(chain_df[chain_df["strike"] == atm_strike]["implied_volatility"].mean())

    put_strike = atm_strike - strike_distance
    call_strike = atm_strike + strike_distance

    put_match = chain_df[(chain_df["strike"] == put_strike) & (chain_df["option_type"] == "PE")]
    call_match = chain_df[(chain_df["strike"] == call_strike) & (chain_df["option_type"] == "CE")]

    put_iv = float(put_match["implied_volatility"].iloc[0]) if not put_match.empty else atm_iv
    call_iv = float(call_match["implied_volatility"].iloc[0]) if not call_match.empty else atm_iv

    abs_skew = put_iv - call_iv
    norm_skew = abs_skew / atm_iv if atm_iv > 0 else 0.0

    return SkewMetrics(
        atm_iv=atm_iv,
        otm_put_iv=put_iv,
        otm_call_iv=call_iv,
        absolute_skew=abs_skew,
        normalized_skew=norm_skew,
    )
