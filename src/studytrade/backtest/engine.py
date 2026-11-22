"""
StudyTrade — Event-Driven Systematic Backtesting Engine
=========================================================

Event loop engine simulating realistic option buying strategies:
  - Strict point-in-time data slicing (no future leakage).
  - Bid/Ask fills with slippage.
  - Multi-condition exit rule enforcement (Stop loss, Profit target, Time stop, Theta decay).
  - Full statutory transaction fee schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from studytrade.backtest.fills import FillSimulator
from studytrade.backtest.portfolio import Portfolio
from studytrade.constants import INSTRUMENTS, TIMEZONE, OptionType
from studytrade.data.cleaner import PointInTimeView
from studytrade.strategy.base import BaseStrategy, SignalDirection
from studytrade.strategy.rules import ExitRuleEvaluator
from studytrade.strategy.selector import ContractSelector


@dataclass(frozen=True)
class BacktestResult:
    """Summary of backtest execution."""

    strategy_name: str
    initial_capital: float
    final_equity: float
    net_profit: float
    total_fees: float
    total_trades: int
    trades_df: pd.DataFrame
    equity_df: pd.DataFrame


class BacktestEngine:
    """
    Event-driven backtesting engine for options buying strategies.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 500000.0,
        slippage_ticks: float = 1.0,
        max_concurrent_positions: int = 3,
        exit_evaluator: ExitRuleEvaluator | None = None,
    ) -> None:
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.slippage_ticks = slippage_ticks
        self.max_concurrent_positions = max_concurrent_positions
        self.exit_evaluator = exit_evaluator or ExitRuleEvaluator()
        self.fill_sim = FillSimulator(slippage_ticks=slippage_ticks)

    def run(
        self,
        spot_df: pd.DataFrame,
        options_df: pd.DataFrame,
        symbol: str = "NIFTY",
    ) -> BacktestResult:
        """
        Execute backtest over historical time-series.
        """
        if spot_df.empty or options_df.empty:
            raise ValueError("Spot and Options DataFrames must not be empty")

        spec = INSTRUMENTS.get(symbol, INSTRUMENTS["NIFTY"])
        portfolio = Portfolio(initial_capital=self.initial_capital)
        pit_spot = PointInTimeView(spot_df)

        # Pre-group option chain by timestamp for O(1) instant slice lookup
        chains_by_ts: dict[Any, pd.DataFrame] = {
            ts: df for ts, df in options_df.groupby("timestamp")
        }
        empty_chain = options_df.iloc[:0]

        timestamps = spot_df["timestamp"].sort_values().unique()

        for cur_ts in timestamps:
            cur_dt = pd.to_datetime(cur_ts).tz_convert(TIMEZONE)
            cur_chain = chains_by_ts.get(cur_dt, empty_chain)

            # 1. Update and evaluate existing open positions
            positions_to_close: list[tuple[str, float, str]] = []

            for sym, pos in list(portfolio.open_positions.items()):
                pos.bars_held += 1
                dte = max(0, (pos.expiry.date() - cur_dt.date()).days)

                # Check expiry settlement first if contract reached or passed expiry
                if cur_dt.date() >= pos.expiry.date():
                    spot_val = float(spot_df[spot_df["timestamp"] <= cur_dt]["close"].iloc[-1])
                    intrinsic = (
                        max(0.0, spot_val - pos.strike)
                        if pos.option_type == OptionType.CALL
                        else max(0.0, pos.strike - spot_val)
                    )
                    positions_to_close.append((sym, intrinsic, "EXPIRY_SETTLEMENT"))
                    continue

                # Find latest quote for this specific option contract in current bar
                if not cur_chain.empty:
                    matching_quotes = cur_chain[cur_chain["symbol"] == sym]
                else:
                    matching_quotes = empty_chain

                if matching_quotes.empty:
                    quote_mask = (options_df["timestamp"] <= cur_dt) & (options_df["symbol"] == sym)
                    matching_quotes = options_df[quote_mask]

                if matching_quotes.empty:
                    continue

                latest_quote = matching_quotes.iloc[-1]
                cur_bid = float(latest_quote["bid"])
                pos.peak_bid = max(pos.peak_bid, cur_bid)

                # Check exit rules
                decision = self.exit_evaluator.evaluate(
                    entry_price=pos.entry_price,
                    current_bid=cur_bid,
                    peak_bid=pos.peak_bid,
                    bars_held=pos.bars_held,
                    dte_days=dte,
                )

                if decision.should_exit:
                    # Execute sell at Bid - slippage
                    fill = self.fill_sim.execute_sell(
                        timestamp=cur_dt,
                        symbol=sym,
                        bid_price=cur_bid,
                        lot_size=pos.lot_size,
                        n_lots=pos.n_lots,
                    )
                    positions_to_close.append((sym, fill.executed_price, decision.reason.value))

            # Execute closed trades
            for sym, exit_price, reason in positions_to_close:
                portfolio.close_position(cur_dt, sym, exit_price, reason)

            # 2. Check for new entry signals
            if len(portfolio.open_positions) < self.max_concurrent_positions:
                # Point-in-time slice up to current bar
                historical_spot = pit_spot.as_of(cur_dt)
                signals = self.strategy.generate_signals(historical_spot, chain_df=cur_chain)

                for sig in signals:
                    if len(portfolio.open_positions) >= self.max_concurrent_positions:
                        break

                    is_call = sig.direction in (SignalDirection.BUY_CALL, SignalDirection.BULL_CALL_SPREAD)
                    is_put = sig.direction in (SignalDirection.BUY_PUT, SignalDirection.BEAR_PUT_SPREAD)
                    if not (is_call or is_put):
                        continue

                    opt_type = OptionType.CALL if is_call else OptionType.PUT

                    if cur_chain.empty:
                        continue

                    # Select contract
                    spot_close = historical_spot["close"].iloc[-1]
                    exp = ContractSelector.select_expiry(
                        cur_chain, cur_dt, min_dte=sig.min_dte, max_dte=sig.max_dte
                    )
                    if exp is None:
                        # Fallback to broader DTE window if exact narrow window has no contract
                        exp = ContractSelector.select_expiry(
                            cur_chain, cur_dt, min_dte=max(0, sig.min_dte - 2), max_dte=sig.max_dte + 15
                        )
                    if exp is None:
                        continue

                    contract = ContractSelector.select_strike_by_delta(
                        cur_chain,
                        spot_price=spot_close,
                        expiry=exp,
                        option_type=opt_type,
                        target_delta=sig.target_delta,
                    )
                    if contract is None or contract.symbol in portfolio.open_positions:
                        continue

                    # Simulate fill at Ask + slippage
                    fill = self.fill_sim.execute_buy(
                        timestamp=cur_dt,
                        symbol=contract.symbol,
                        ask_price=contract.ask,
                        lot_size=spec.lot_size,
                        n_lots=1,
                    )

                    # Attempt position open
                    portfolio.open_position(
                        timestamp=cur_dt,
                        symbol=contract.symbol,
                        underlying_symbol=symbol,
                        strike=contract.strike,
                        expiry=contract.expiry,
                        option_type=opt_type,
                        entry_price=fill.executed_price,
                        lot_size=spec.lot_size,
                        n_lots=1,
                    )

            # 3. Record equity snapshot
            portfolio.record_bar_snapshot(cur_dt)

        # Force-settle any positions remaining open on the final simulation bar
        if portfolio.open_positions:
            final_dt = pd.to_datetime(timestamps[-1]).tz_convert(TIMEZONE)
            final_spot = float(spot_df["close"].iloc[-1])
            for sym, pos in list(portfolio.open_positions.items()):
                intrinsic = (
                    max(0.0, final_spot - pos.strike)
                    if pos.option_type == OptionType.CALL
                    else max(0.0, pos.strike - final_spot)
                )
                portfolio.close_position(final_dt, sym, intrinsic, "FINAL_BAR_SETTLEMENT")

        trades_df, equity_df = portfolio.to_dataframe()
        final_eq = portfolio.total_equity
        net_profit = final_eq - self.initial_capital

        return BacktestResult(
            strategy_name=self.strategy.name,
            initial_capital=self.initial_capital,
            final_equity=final_eq,
            net_profit=net_profit,
            total_fees=portfolio.total_fees_paid,
            total_trades=len(trades_df),
            trades_df=trades_df,
            equity_df=equity_df,
        )
