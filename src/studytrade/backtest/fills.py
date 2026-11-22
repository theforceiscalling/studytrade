"""
StudyTrade — Realistic Option Fill Simulator
==============================================

Simulates non-ideal market order execution using top-of-book Bid/Ask quotes,
tick slippage, volume participation filters, and cash settlement at expiration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from studytrade.constants import OptionType


@dataclass(frozen=True)
class ExecutionFill:
    """Filled order details."""

    timestamp: datetime
    symbol: str
    action: str  # 'BUY' or 'SELL'
    requested_price: float  # Ask on buy, Bid on sell
    executed_price: float  # Price after slippage
    n_lots: int
    lot_size: int
    slippage_cost: float
    liquidity_warning: bool


class FillSimulator:
    """
    Simulates realistic fills enforcing bid/ask spread crossing and slippage.
    """

    def __init__(
        self,
        slippage_ticks: float = 1.0,
        tick_size: float = 0.05,
        max_volume_pct: float = 0.10,  # Cannot take more than 10% of bar volume
    ) -> None:
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.max_volume_pct = max_volume_pct

    def execute_buy(
        self,
        timestamp: datetime,
        symbol: str,
        ask_price: float,
        lot_size: int,
        n_lots: int = 1,
        bar_volume: int = 1000,
    ) -> ExecutionFill:
        """
        Execute option buy order crossing the spread at Ask + slippage.
        """
        slippage_per_unit = self.slippage_ticks * self.tick_size
        exec_price = ask_price + slippage_per_unit
        units = lot_size * n_lots
        slip_cost = slippage_per_unit * units

        # Check volume participation
        liq_warning = units > (bar_volume * self.max_volume_pct)

        return ExecutionFill(
            timestamp=timestamp,
            symbol=symbol,
            action="BUY",
            requested_price=ask_price,
            executed_price=float(exec_price),
            n_lots=n_lots,
            lot_size=lot_size,
            slippage_cost=float(slip_cost),
            liquidity_warning=liq_warning,
        )

    def execute_sell(
        self,
        timestamp: datetime,
        symbol: str,
        bid_price: float,
        lot_size: int,
        n_lots: int = 1,
        bar_volume: int = 1000,
    ) -> ExecutionFill:
        """
        Execute option sell order crossing the spread at Bid - slippage.
        """
        slippage_per_unit = self.slippage_ticks * self.tick_size
        exec_price = max(self.tick_size, bid_price - slippage_per_unit)
        units = lot_size * n_lots
        slip_cost = slippage_per_unit * units

        liq_warning = units > (bar_volume * self.max_volume_pct)

        return ExecutionFill(
            timestamp=timestamp,
            symbol=symbol,
            action="SELL",
            requested_price=bid_price,
            executed_price=float(exec_price),
            n_lots=n_lots,
            lot_size=lot_size,
            slippage_cost=float(slip_cost),
            liquidity_warning=liq_warning,
        )

    @staticmethod
    def settle_at_expiry(
        strike: float,
        spot_settlement_price: float,
        option_type: OptionType,
    ) -> float:
        """
        Compute European cash settlement value at 15:30 on expiry date.
        """
        if option_type == OptionType.CALL:
            return float(max(0.0, spot_settlement_price - strike))
        else:
            return float(max(0.0, strike - spot_settlement_price))
