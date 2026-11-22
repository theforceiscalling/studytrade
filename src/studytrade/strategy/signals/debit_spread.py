"""
StudyTrade — Defined-Risk Debit Spread Strategy
=================================================

Transforms directional signals into Bull Call or Bear Put Spreads,
selling a higher OTM strike to partially finance the long option and reduce theta burn.
"""

from __future__ import annotations

import pandas as pd

from studytrade.strategy.base import BaseStrategy, SignalDirection, TradeSignal


class DebitSpreadStrategy(BaseStrategy):
    """
    Wraps an underlying directional strategy and converts single-leg options
    into defined-risk vertical debit spreads (Bull Call or Bear Put).
    """

    def __init__(self, base_directional_strategy: BaseStrategy) -> None:
        super().__init__(name=f"DebitSpread_{base_directional_strategy.name}")
        self.base_strategy = base_directional_strategy

    def generate_signals(
        self,
        spot_df: pd.DataFrame,
        chain_df: pd.DataFrame | None = None,
    ) -> list[TradeSignal]:
        raw_signals = self.base_strategy.generate_signals(spot_df, chain_df)
        spread_signals: list[TradeSignal] = []

        for sig in raw_signals:
            if sig.direction == SignalDirection.BUY_CALL:
                spread_signals.append(
                    TradeSignal(
                        timestamp=sig.timestamp,
                        underlying_symbol=sig.underlying_symbol,
                        direction=SignalDirection.BULL_CALL_SPREAD,
                        strength=sig.strength,
                        reason=f"Bull Call Spread ({sig.reason})",
                        target_delta=0.50,  # Long leg target
                        min_dte=sig.min_dte,
                        max_dte=sig.max_dte,
                    )
                )
            elif sig.direction == SignalDirection.BUY_PUT:
                spread_signals.append(
                    TradeSignal(
                        timestamp=sig.timestamp,
                        underlying_symbol=sig.underlying_symbol,
                        direction=SignalDirection.BEAR_PUT_SPREAD,
                        strength=sig.strength,
                        reason=f"Bear Put Spread ({sig.reason})",
                        target_delta=0.50,
                        min_dte=sig.min_dte,
                        max_dte=sig.max_dte,
                    )
                )

        return spread_signals
