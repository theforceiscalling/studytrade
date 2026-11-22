"""
StudyTrade — Momentum and Trend Alignment Option Buying Strategy
==================================================================

Generates directional option buying signals based on EMA trend alignment,
MACD momentum, and RSI confirmation.
"""

from __future__ import annotations

import pandas as pd

from studytrade.features.technical import compute_ema, compute_macd, compute_rsi
from studytrade.strategy.base import BaseStrategy, SignalDirection, TradeSignal


class MomentumStrategy(BaseStrategy):
    """
    Momentum option buying strategy.
    """

    def __init__(
        self,
        fast_ema: int = 10,
        slow_ema: int = 30,
        target_delta: float = 0.45,
    ) -> None:
        super().__init__(name="Trend_Momentum")
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.target_delta = target_delta

    def generate_signals(
        self,
        spot_df: pd.DataFrame,
        chain_df: pd.DataFrame | None = None,
    ) -> list[TradeSignal]:
        if len(spot_df) < self.slow_ema + 10:
            return []

        c = spot_df["close"]
        ema_fast = compute_ema(c, self.fast_ema)
        ema_slow = compute_ema(c, self.slow_ema)
        macd_df = compute_macd(c)
        rsi = compute_rsi(c, period=14)

        idx = len(spot_df) - 1
        cur_ts = spot_df["timestamp"].iloc[idx]
        sym = spot_df["symbol"].iloc[idx]

        bull_trend = ema_fast.iloc[idx] > ema_slow.iloc[idx]
        bear_trend = ema_fast.iloc[idx] < ema_slow.iloc[idx]
        macd_pos = macd_df["macd_hist"].iloc[idx] > 0
        macd_neg = macd_df["macd_hist"].iloc[idx] < 0
        rsi_bull = 52.0 <= rsi.iloc[idx] <= 72.0
        rsi_bear = 28.0 <= rsi.iloc[idx] <= 48.0

        signals: list[TradeSignal] = []

        if bull_trend and macd_pos and rsi_bull:
            signals.append(
                TradeSignal(
                    timestamp=cur_ts,
                    underlying_symbol=sym,
                    direction=SignalDirection.BUY_CALL,
                    strength=0.75,
                    reason=(
                        f"Bullish momentum: Fast EMA > Slow EMA, "
                        f"MACD hist > 0, RSI={rsi.iloc[idx]:.1f}"
                    ),
                    target_delta=self.target_delta,
                )
            )
        elif bear_trend and macd_neg and rsi_bear:
            signals.append(
                TradeSignal(
                    timestamp=cur_ts,
                    underlying_symbol=sym,
                    direction=SignalDirection.BUY_PUT,
                    strength=0.75,
                    reason=(
                        f"Bearish momentum: Fast EMA < Slow EMA, "
                        f"MACD hist < 0, RSI={rsi.iloc[idx]:.1f}"
                    ),
                    target_delta=self.target_delta,
                )
            )

        return signals
