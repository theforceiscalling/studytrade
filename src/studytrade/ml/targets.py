"""
StudyTrade — ML Target Generation and Labeling
================================================

Generates leakage-free targets for option buying:
  - Directional breakout targets (binary / ternary).
  - Volatility expansion target (predicting surges in ATR / realized volatility).
  - Triple-barrier labeling (first touch of profit target vs stop loss barrier).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_direction_target(
    close: pd.Series | np.ndarray,
    horizon: int = 5,
    threshold_pct: float = 0.5,
) -> pd.Series:
    """
    Directional target over next `horizon` bars.
      1: Price moves UP by > threshold_pct
     -1: Price moves DOWN by > threshold_pct
      0: Flat / Chop
    """
    s = pd.Series(close) if not isinstance(close, pd.Series) else close
    # Future forward return: (Close_{t+horizon} - Close_t) / Close_t
    future_return = (s.shift(-horizon) - s) / s * 100.0

    target = pd.Series(0, index=s.index)
    target[future_return > threshold_pct] = 1
    target[future_return < -threshold_pct] = -1

    # Invalidate the last `horizon` bars where future return is unknown
    target.iloc[-horizon:] = np.nan
    return target


def create_volatility_expansion_target(
    atr: pd.Series | np.ndarray,
    horizon: int = 5,
    expansion_threshold: float = 1.20,
) -> pd.Series:
    """
    Binary target predicting whether volatility will expand by >= expansion_threshold
    over the next `horizon` bars.
    """
    s = pd.Series(atr) if not isinstance(atr, pd.Series) else atr
    future_atr = s.shift(-horizon)
    expansion_ratio = future_atr / np.maximum(s, 1e-12)

    target = (expansion_ratio >= expansion_threshold).astype(float)
    target.iloc[-horizon:] = np.nan
    return target


def create_triple_barrier_labels(
    close: pd.Series | np.ndarray,
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    horizon: int = 10,
    profit_take_pct: float = 2.0,
    stop_loss_pct: float = 1.0,
) -> pd.Series:
    """
    Triple-Barrier Method (Marcos Lopez de Prado):
      1: Upper profit-taking barrier hit first
     -1: Lower stop-loss barrier hit first
      0: Vertical time barrier (neither touched by horizon end)
    """
    c = pd.Series(close).values
    h = pd.Series(high).values
    l_ = pd.Series(low).values
    n = len(c)

    labels = np.full(n, np.nan)

    for i in range(n - horizon):
        entry_p = c[i]
        upper_barrier = entry_p * (1.0 + profit_take_pct / 100.0)
        lower_barrier = entry_p * (1.0 - stop_loss_pct / 100.0)

        label = 0.0
        for step in range(1, horizon + 1):
            curr_h = h[i + step]
            curr_l = l_[i + step]

            hit_upper = curr_h >= upper_barrier
            hit_lower = curr_l <= lower_barrier

            if hit_upper and not hit_lower:
                label = 1.0
                break
            elif hit_lower and not hit_upper:
                label = -1.0
                break
            elif hit_upper and hit_lower:
                # Ambiguous same-bar breach -> assign 0 or conservative -1
                label = -1.0
                break

        labels[i] = label

    return pd.Series(labels, index=pd.Series(close).index)
