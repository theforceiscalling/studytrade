"""
StudyTrade — Portfolio, Position, and Trade Journal Tracker
============================================================

Maintains portfolio cash, open option contracts, equity curve history,
and completed trade records with fee deduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from studytrade.backtest.costs import compute_nse_option_costs
from studytrade.constants import OptionType


@dataclass(frozen=True)
class TradeRecord:
    """Historical completed round-trip trade record."""

    entry_time: datetime
    exit_time: datetime
    symbol: str
    underlying_symbol: str
    strike: float
    expiry: datetime
    option_type: OptionType
    n_lots: int
    lot_size: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    total_fees: float
    return_pct: float
    holding_bars: int
    exit_reason: str


@dataclass
class OpenPosition:
    """Active open option contract."""

    entry_time: datetime
    symbol: str
    underlying_symbol: str
    strike: float
    expiry: datetime
    option_type: OptionType
    n_lots: int
    lot_size: int
    entry_price: float  # Ask + slippage
    peak_bid: float
    bars_held: int = 0


class Portfolio:
    """
    Manages capital balance, active positions, and performance equity time-series.
    """

    def __init__(self, initial_capital: float = 500000.0) -> None:
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.open_positions: dict[str, OpenPosition] = {}
        self.closed_trades: list[TradeRecord] = []
        self.equity_history: list[dict] = []
        self.total_fees_paid: float = 0.0

    @property
    def total_equity(self) -> float:
        """Total current equity = cash + mark-to-market of open positions."""
        position_val = sum(
            pos.entry_price * pos.lot_size * pos.n_lots for pos in self.open_positions.values()
        )
        return self.cash + position_val

    def open_position(
        self,
        timestamp: datetime,
        symbol: str,
        underlying_symbol: str,
        strike: float,
        expiry: datetime,
        option_type: OptionType,
        entry_price: float,
        lot_size: int,
        n_lots: int = 1,
    ) -> bool:
        """
        Deduct cash premium and open position.
        """
        if symbol in self.open_positions:
            return False  # Already open, prevent duplicate clobbering

        premium_required = entry_price * lot_size * n_lots
        if self.cash < premium_required:
            return False  # Insufficient margin/cash

        self.cash -= premium_required
        self.open_positions[symbol] = OpenPosition(
            entry_time=timestamp,
            symbol=symbol,
            underlying_symbol=underlying_symbol,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            n_lots=n_lots,
            lot_size=lot_size,
            entry_price=entry_price,
            peak_bid=entry_price,
            bars_held=0,
        )
        return True

    def close_position(
        self,
        timestamp: datetime,
        symbol: str,
        exit_price: float,
        exit_reason: str,
    ) -> TradeRecord | None:
        """
        Close an open position, compute statutory fees, credit cash, and log record.
        """
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return None

        units = pos.lot_size * pos.n_lots
        gross_proceeds = exit_price * units
        cost_breakdown = compute_nse_option_costs(
            buy_price=pos.entry_price,
            sell_price=exit_price,
            lot_size=pos.lot_size,
            n_lots=pos.n_lots,
        )

        net_proceeds = gross_proceeds - cost_breakdown.total_statutory_fees
        self.cash += net_proceeds
        self.total_fees_paid += cost_breakdown.total_statutory_fees

        gross_pnl = (exit_price - pos.entry_price) * units
        net_pnl = gross_pnl - cost_breakdown.total_statutory_fees
        return_pct = (net_pnl / (pos.entry_price * units)) * 100.0

        record = TradeRecord(
            entry_time=pos.entry_time,
            exit_time=timestamp,
            symbol=pos.symbol,
            underlying_symbol=pos.underlying_symbol,
            strike=pos.strike,
            expiry=pos.expiry,
            option_type=pos.option_type,
            n_lots=pos.n_lots,
            lot_size=pos.lot_size,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            gross_pnl=float(gross_pnl),
            net_pnl=float(net_pnl),
            total_fees=float(cost_breakdown.total_statutory_fees),
            return_pct=float(return_pct),
            holding_bars=pos.bars_held,
            exit_reason=exit_reason,
        )
        self.closed_trades.append(record)
        return record

    def record_bar_snapshot(self, timestamp: datetime) -> None:
        """Record equity timestamp snapshot for equity curve analysis."""
        self.equity_history.append(
            {
                "timestamp": timestamp,
                "cash": self.cash,
                "equity": self.total_equity,
                "open_positions": len(self.open_positions),
                "total_fees": self.total_fees_paid,
            }
        )

    def to_dataframe(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert trade history and equity curve to DataFrames."""
        trades_df = pd.DataFrame([vars(t) for t in self.closed_trades])
        equity_df = pd.DataFrame(self.equity_history)
        return trades_df, equity_df
