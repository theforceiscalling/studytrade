"""
StudyTrade — Technical Indicators and Trend/Momentum Features
===============================================================

Pure-NumPy / Pandas vectorized implementation of technical indicators
designed for directional option buying.

Features:
  - Trend: EMA (fast, slow, baseline), MACD line, signal, histogram.
  - Momentum: RSI (Wilder's smoothing), Rate of Change (ROC).
  - Volatility & Range: ATR (Average True Range), Normalized ATR (NATR),
    Bollinger Bands and Bollinger Bandwidth (volatility compression/squeeze).
  - Breakouts: Donchian Channels (N-period High/Low breakout).

Guarantees:
  - All rolling windows are strictly backward-looking (shift(1) where appropriate)
    to prevent look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ema(series: pd.Series | np.ndarray, span: int) -> pd.Series:
    """Exponential Moving Average."""
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    return s.ewm(span=span, adjust=False).mean()


def compute_macd(
    close: pd.Series | np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).

    MACD = EMA_fast(Close) - EMA_slow(Close)
    Signal = EMA_signal(MACD)
    Hist = MACD - Signal
    """
    s = pd.Series(close) if not isinstance(close, pd.Series) else close
    ema_fast = compute_ema(s, fast)
    ema_slow = compute_ema(s, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        }
    )


def compute_rsi(close: pd.Series | np.ndarray, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) using Wilder's smoothing.

    RS = AvgGain / AvgLoss
    RSI = 100 - (100 / (1 + RS))
    """
    s = pd.Series(close) if not isinstance(close, pd.Series) else close
    delta = s.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing via ewm(alpha = 1 / period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_atr(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    period: int = 14,
) -> pd.DataFrame:
    """
    Average True Range (ATR) and Normalized ATR (NATR = ATR / Close * 100).

    TR = max(High - Low, |High - Close_{t-1}|, |Low - Close_{t-1}|)
    ATR = Wilder_EMA(TR, period)
    """
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l_ = pd.Series(low) if not isinstance(low, pd.Series) else low
    c = pd.Series(close) if not isinstance(close, pd.Series) else close

    prev_c = c.shift(1)
    tr1 = h - l_
    tr2 = (h - prev_c).abs()
    tr3 = (l_ - prev_c).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    natr = (atr / c) * 100.0

    return pd.DataFrame({"true_range": true_range, "atr": atr, "natr": natr})


def compute_bollinger_bands(
    close: pd.Series | np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands and Bandwidth.

    Mid = SMA(Close, period)
    Upper = Mid + num_std * StdDev
    Lower = Mid - num_std * StdDev
    Bandwidth = (Upper - Lower) / Mid  (Used to detect volatility squeezes)
    """
    s = pd.Series(close) if not isinstance(close, pd.Series) else close
    mid = s.rolling(window=period).mean()
    std = s.rolling(window=period).std(ddof=1)

    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / np.maximum(mid, 1e-12)
    percent_b = (s - lower) / np.maximum(upper - lower, 1e-12)

    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_pct_b": percent_b,
        }
    )


def compute_donchian_breakout(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Donchian Channel breakout signals.

    Upper = max(High_{t-1}, ..., High_{t-lookback})
    Lower = min(Low_{t-1}, ..., Low_{t-lookback})
    Breakout Long  = Close > Upper (prior bar high)
    Breakout Short = Close < Lower (prior bar low)
    """
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l_ = pd.Series(low) if not isinstance(low, pd.Series) else low
    c = pd.Series(close) if not isinstance(close, pd.Series) else close

    # Must shift by 1 to exclude current bar from channel boundary
    donchian_high = h.shift(1).rolling(window=lookback).max()
    donchian_low = l_.shift(1).rolling(window=lookback).min()

    breakout_bull = c > donchian_high
    breakout_bear = c < donchian_low

    return pd.DataFrame(
        {
            "donchian_high": donchian_high,
            "donchian_low": donchian_low,
            "breakout_bull": breakout_bull,
            "breakout_bear": breakout_bear,
        }
    )
