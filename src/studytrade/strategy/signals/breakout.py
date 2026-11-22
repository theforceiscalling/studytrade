"""
StudyTrade — Donchian Breakout Option Buying Strategy
=======================================================

Generates Call and Put buying signals when underlying breaks out
of an N-period Donchian channel with volume confirmation.
"""

from __future__ import annotations

import pandas as pd

from studytrade.features.technical import compute_donchian_breakout
from studytrade.strategy.base import BaseStrategy, SignalDirection, TradeSignal


class BreakoutStrategy(BaseStrategy):
    """
    Directional breakout option buying strategy.
    """

    def __init__(
        self,
        lookback: int = 20,
        volume_mult: float = 1.15,
        target_delta: float = 0.40,
    ) -> None:
        super().__init__(name="Donchian_Breakout")
        self.lookback = lookback
        self.volume_mult = volume_mult
        self.target_delta = target_delta

    def generate_signals(
        self,
        spot_df: pd.DataFrame,
        chain_df: pd.DataFrame | None = None,
    ) -> list[TradeSignal]:
        if len(spot_df) < self.lookback + 2:
            return []

        h = spot_df["high"]
        l_ = spot_df["low"]
        c = spot_df["close"]
        v = spot_df["volume"]

        breakouts = compute_donchian_breakout(h, l_, c, lookback=self.lookback)
        vol_sma = v.rolling(window=self.lookback).mean()

        latest_idx = len(spot_df) - 1
        is_bull_break = bool(breakouts["breakout_bull"].iloc[latest_idx] or (h.iloc[latest_idx] > breakouts["donchian_high"].iloc[latest_idx]))
        is_bear_break = bool(breakouts["breakout_bear"].iloc[latest_idx] or (l_.iloc[latest_idx] < breakouts["donchian_low"].iloc[latest_idx]))
        vol_surge = bool(v.iloc[latest_idx] >= (vol_sma.iloc[latest_idx] * min(self.volume_mult, 1.05)))

        signals: list[TradeSignal] = []
        cur_ts = spot_df["timestamp"].iloc[latest_idx]
        sym = spot_df["symbol"].iloc[latest_idx]

        if is_bull_break and vol_surge:
            high_val = breakouts["donchian_high"].iloc[latest_idx]
            signals.append(
                TradeSignal(
                    timestamp=cur_ts,
                    underlying_symbol=sym,
                    direction=SignalDirection.BUY_CALL,
                    strength=0.85,
                    reason=f"Bullish Donchian breakout above {high_val:.1f}",
                    target_delta=self.target_delta,
                )
            )
        elif is_bear_break and vol_surge:
            low_val = breakouts["donchian_low"].iloc[latest_idx]
            signals.append(
                TradeSignal(
                    timestamp=cur_ts,
                    underlying_symbol=sym,
                    direction=SignalDirection.BUY_PUT,
                    strength=0.85,
                    reason=f"Bearish Donchian breakdown below {low_val:.1f}",
                    target_delta=self.target_delta,
                )
            )

        return signals
