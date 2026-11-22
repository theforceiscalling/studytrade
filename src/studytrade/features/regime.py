"""
StudyTrade — Market Regime Detection
======================================

Classifies market regimes (Trending Bull, Trending Bear, Low-Vol Chop, High-Vol Panic)
to filter trades for systematic option buying.

Core Principle for Option Buyers:
  - Option buyers MUST AVOID the 'CHOP_LOW_VOL' regime where theta decay
    inevitably outpaces minimal underlying movement.
  - Option buying thrives in 'BREAKOUT_EXPANSION' and 'HIGH_VOL_MOMENTUM'.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

from studytrade.features.technical import compute_atr, compute_ema


class MarketRegime(StrEnum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    CHOP_LOW_VOL = "CHOP_LOW_VOL"
    HIGH_VOL_EXPANSION = "HIGH_VOL_EXPANSION"


def detect_regimes(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    trend_period: int = 50,
    vol_period: int = 20,
    vol_percentile_cutoff: float = 70.0,
) -> pd.Series:
    """
    Classify market regimes using trend slope and rolling volatility percentiles.

    Rules:
      1. High-Vol Expansion: Current volatility in top 30% of distribution.
      2. Trending Bull: Price > EMA(50) and EMA slope > 0, normal volatility.
      3. Trending Bear: Price < EMA(50) and EMA slope < 0, normal volatility.
      4. Chop / Low Vol: Flat EMA and low volatility.
    """
    c = pd.Series(close) if not isinstance(close, pd.Series) else close
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l_ = pd.Series(low) if not isinstance(low, pd.Series) else low

    # Trend filter
    ema_trend = compute_ema(c, span=trend_period)
    trend_slope = ema_trend.diff(periods=5)

    # Volatility filter via NATR
    atr_df = compute_atr(h, l_, c, period=vol_period)
    natr = atr_df["natr"]
    rolling_vol_pctl = natr.rolling(window=100, min_periods=30).apply(
        lambda x: (np.sum(x < x.iloc[-1]) / len(x)) * 100.0 if len(x) > 0 else 50.0,
        raw=False,
    )

    regimes = pd.Series(MarketRegime.CHOP_LOW_VOL.value, index=c.index)

    # High Volatility Expansion
    high_vol_mask = rolling_vol_pctl >= vol_percentile_cutoff
    regimes[high_vol_mask] = MarketRegime.HIGH_VOL_EXPANSION.value

    # Trending Bull
    bull_mask = (~high_vol_mask) & (c > ema_trend) & (trend_slope > 0)
    regimes[bull_mask] = MarketRegime.TRENDING_BULL.value

    # Trending Bear
    bear_mask = (~high_vol_mask) & (c < ema_trend) & (trend_slope < 0)
    regimes[bear_mask] = MarketRegime.TRENDING_BEAR.value

    return regimes
