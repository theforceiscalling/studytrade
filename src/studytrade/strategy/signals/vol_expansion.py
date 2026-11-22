"""
StudyTrade — Volatility Expansion & Squeeze Breakout Strategy
==============================================================

Identifies low-volatility compression (Bollinger Squeeze) followed
by directional expansion to buy options right before a volatility spike.
"""

from __future__ import annotations

import pandas as pd

from studytrade.features.technical import compute_atr, compute_bollinger_bands
from studytrade.features.volatility import compute_volatility_expansion_ratio
from studytrade.strategy.base import BaseStrategy, SignalDirection, TradeSignal


class VolatilityExpansionStrategy(BaseStrategy):
    """
    Buys options when price breaks out of a volatility compression squeeze.
    """

    def __init__(
        self,
        bb_period: int = 20,
        squeeze_lookback: int = 15,
        target_delta: float = 0.35,
    ) -> None:
        super().__init__(name="Vol_Expansion_Squeeze")
        self.bb_period = bb_period
        self.squeeze_lookback = squeeze_lookback
        self.target_delta = target_delta

    def generate_signals(
        self,
        spot_df: pd.DataFrame,
        chain_df: pd.DataFrame | None = None,
    ) -> list[TradeSignal]:
        if len(spot_df) < self.bb_period + self.squeeze_lookback:
            return []

        c = spot_df["close"]
        h = spot_df["high"]
        l_ = spot_df["low"]

        bb = compute_bollinger_bands(c, period=self.bb_period)
        atr_short = compute_atr(h, l_, c, period=5)["atr"]
        atr_long = compute_atr(h, l_, c, period=20)["atr"]
        vr = compute_volatility_expansion_ratio(atr_short, atr_long)

        idx = len(spot_df) - 1
        cur_ts = spot_df["timestamp"].iloc[idx]
        sym = spot_df["symbol"].iloc[idx]

        # Prior bandwidth low indicates squeeze condition
        prior_bw = bb["bb_bandwidth"].iloc[idx - self.squeeze_lookback : idx]
        was_squeezed = (prior_bw <= prior_bw.quantile(0.35)).any() or (bb["bb_bandwidth"].iloc[idx - 1] <= 0.06)

        # Volatility expansion trigger
        is_expanding = vr.iloc[idx] >= 1.02
        broke_upper = (c.iloc[idx] >= bb["bb_upper"].iloc[idx]) or (h.iloc[idx] >= bb["bb_upper"].iloc[idx])
        broke_lower = (c.iloc[idx] <= bb["bb_lower"].iloc[idx]) or (l_.iloc[idx] <= bb["bb_lower"].iloc[idx])

        signals: list[TradeSignal] = []

        if was_squeezed and is_expanding and broke_upper:
            signals.append(
                TradeSignal(
                    timestamp=cur_ts,
                    underlying_symbol=sym,
                    direction=SignalDirection.BUY_CALL,
                    strength=0.90,
                    reason=f"Volatility squeeze breakout above upper BB, VR={vr.iloc[idx]:.2f}",
                    target_delta=self.target_delta,
                )
            )
        elif was_squeezed and is_expanding and broke_lower:
            signals.append(
                TradeSignal(
                    timestamp=cur_ts,
                    underlying_symbol=sym,
                    direction=SignalDirection.BUY_PUT,
                    strength=0.90,
                    reason=f"Volatility squeeze breakdown below lower BB, VR={vr.iloc[idx]:.2f}",
                    target_delta=self.target_delta,
                )
            )

        return signals
