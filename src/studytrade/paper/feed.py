"""
StudyTrade — Live / Paper Market Data Feed
===========================================

Provides real-time / simulated streaming quotes for underlying spot
and active option chains to drive the paper-trading execution engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from studytrade.data.fetchers.synthetic import SyntheticDataGenerator


@dataclass(frozen=True)
class LiveQuote:
    """Real-time top of book quote."""

    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int


class PaperMarketFeed:
    """
    Feeds real-time or simulated market data ticks to the paper trading engine.
    """

    def __init__(self, synthetic_gen: SyntheticDataGenerator | None = None) -> None:
        self.gen = synthetic_gen or SyntheticDataGenerator(seed=42)
        self._current_spot_price: float = 25800.0
        self._last_update_time: datetime = datetime.now()

    def get_spot_quote(self, symbol: str = "NIFTY") -> LiveQuote:
        """Fetch current underlying spot price quote."""
        now = datetime.now()
        # Random small walk if running standalone
        self._current_spot_price += float(np.random.normal(0, 5.0))
        bid = self._current_spot_price - 0.25
        ask = self._current_spot_price + 0.25
        return LiveQuote(
            timestamp=now,
            symbol=symbol,
            bid=bid,
            ask=ask,
            last_price=self._current_spot_price,
            volume=500000,
            open_interest=0,
        )

    def get_option_chain_snapshot(
        self,
        underlying_symbol: str = "NIFTY",
        expiries: list[datetime] | None = None,
    ) -> pd.DataFrame:
        """
        Generate or fetch current live option chain snapshot.
        """
        now = datetime.now()
        if expiries is None:
            # Next Thursday
            days_ahead = (3 - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            exp = now + pd.Timedelta(days=days_ahead)
            expiries = [exp]

        chain_df = self.gen.generate_option_chain_snapshot(
            timestamp=now,
            underlying_symbol=underlying_symbol,
            spot_price=self._current_spot_price,
            expiries=expiries,
        )

        # Overlay real-time market ticks from LiveTickCache
        try:
            from studytrade.live.session_engine import get_live_tick_cache
            cache = get_live_tick_cache()
            for idx, row in chain_df.iterrows():
                sym = str(row["symbol"])
                tick = cache.get_tick(sym)
                if tick:
                    if "last_price" in tick:
                        chain_df.at[idx, "close"] = tick["last_price"]
                    if "bid" in tick:
                        chain_df.at[idx, "bid"] = tick["bid"]
                    if "ask" in tick:
                        chain_df.at[idx, "ask"] = tick["ask"]
                    if "implied_volatility" in tick:
                        chain_df.at[idx, "implied_volatility"] = tick["implied_volatility"]
        except Exception:
            pass

        return chain_df
