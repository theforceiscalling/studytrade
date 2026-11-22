"""
StudyTrade — Paper Trading Execution Engine and Live Greeks Tracker
=====================================================================

Executes simulated forward trades against real-time quotes, enforces risk management,
and computes live aggregate portfolio Greeks exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from studytrade.backtest.fills import FillSimulator
from studytrade.backtest.portfolio import Portfolio
from studytrade.constants import INSTRUMENTS, OptionType
from studytrade.math.greeks import greeks
from studytrade.paper.journal import PaperJournal
from studytrade.risk.limits import RiskManager
from studytrade.risk.sizing import PositionSizer
from studytrade.strategy.base import BaseStrategy, SignalDirection
from studytrade.strategy.rules import ExitRuleEvaluator
from studytrade.strategy.selector import ContractSelector


@dataclass(frozen=True)
class PortfolioGreeksExposure:
    """Aggregate portfolio Greeks."""

    net_delta_units: float  # Directional exposure in underlying index units
    net_gamma_units: float
    total_daily_theta_inr: float  # Daily time decay in rupees
    net_vega_1pct_inr: float  # Rupee PnL impact per 1% IV shift


class PaperTradingExecutor:
    """
    Coordinates forward paper execution, risk checks, and live Greek monitoring.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 500000.0,
        journal: PaperJournal | None = None,
        risk_manager: RiskManager | None = None,
        exit_evaluator: ExitRuleEvaluator | None = None,
        slippage_ticks: float = 1.0,
    ) -> None:
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.journal = journal or PaperJournal()
        self.risk_manager = risk_manager or RiskManager(initial_capital=initial_capital)
        self.exit_evaluator = exit_evaluator or ExitRuleEvaluator()
        self.fill_sim = FillSimulator(slippage_ticks=slippage_ticks)

    def compute_portfolio_greeks(
        self,
        current_spot: float,
        options_chain_df: pd.DataFrame,
    ) -> PortfolioGreeksExposure:
        """
        Aggregate analytical Greeks across all currently open option positions.
        """
        if not self.portfolio.open_positions:
            return PortfolioGreeksExposure(0.0, 0.0, 0.0, 0.0)

        total_delta = 0.0
        total_gamma = 0.0
        total_theta_daily_inr = 0.0
        total_vega_1pct_inr = 0.0

        for sym, pos in self.portfolio.open_positions.items():
            matching = options_chain_df[options_chain_df["symbol"] == sym]
            if matching.empty:
                continue

            row = matching.iloc[-1]
            cur_iv = float(row.get("implied_volatility", 0.16))
            units = pos.lot_size * pos.n_lots

            cur_ts = pd.to_datetime(row["timestamp"])
            dte_days = max(1, (pos.expiry.date() - cur_ts.date()).days)
            t_years = max(1e-4, dte_days / 365.0)

            g = greeks(
                option_type=pos.option_type,
                s=current_spot,
                k=pos.strike,
                t=t_years,
                r=0.07,
                sigma=cur_iv,
            )

            total_delta += float(g.delta) * units
            total_gamma += float(g.gamma) * units
            total_theta_daily_inr += float(g.theta_daily) * units
            total_vega_1pct_inr += float(g.vega_1pct) * units

        return PortfolioGreeksExposure(
            net_delta_units=total_delta,
            net_gamma_units=total_gamma,
            total_daily_theta_inr=total_theta_daily_inr,
            net_vega_1pct_inr=total_vega_1pct_inr,
        )

    def process_market_tick(
        self,
        spot_history_df: pd.DataFrame,
        current_chain_df: pd.DataFrame,
        symbol: str = "NIFTY",
    ) -> None:
        """
        Ingest live tick/bar, evaluate exits, size and execute new entries.
        """
        spec = INSTRUMENTS.get(symbol, INSTRUMENTS["NIFTY"])
        now = datetime.now()
        spot_close = spot_history_df["close"].iloc[-1]

        # 1. Evaluate exits for existing positions
        for sym, pos in list(self.portfolio.open_positions.items()):
            pos.bars_held += 1
            dte = max(0, (pos.expiry.date() - now.date()).days)

            # Check terminal intrinsic settlement if past expiry
            if now.date() >= pos.expiry.date():
                intrinsic = (
                    max(0.0, spot_close - pos.strike)
                    if pos.option_type == OptionType.CALL
                    else max(0.0, pos.strike - spot_close)
                )
                record = self.portfolio.close_position(
                    timestamp=now,
                    symbol=sym,
                    exit_price=intrinsic,
                    exit_reason="EXPIRY_SETTLEMENT",
                )
                if record:
                    self.journal.log_trade(
                        entry_time=record.entry_time,
                        exit_time=record.exit_time,
                        symbol=record.symbol,
                        underlying_symbol=record.underlying_symbol,
                        strike=record.strike,
                        option_type=record.option_type.value,
                        lot_size=record.lot_size,
                        n_lots=record.n_lots,
                        entry_price=record.entry_price,
                        exit_price=record.exit_price,
                        net_pnl=record.net_pnl,
                        exit_reason=record.exit_reason,
                    )
                continue

            matching = current_chain_df[current_chain_df["symbol"] == sym]
            if matching.empty:
                continue

            cur_bid = float(matching["bid"].iloc[-1])
            pos.peak_bid = max(pos.peak_bid, cur_bid)

            decision = self.exit_evaluator.evaluate(
                entry_price=pos.entry_price,
                current_bid=cur_bid,
                peak_bid=pos.peak_bid,
                bars_held=pos.bars_held,
                dte_days=dte,
            )

            if decision.should_exit:
                fill = self.fill_sim.execute_sell(
                    timestamp=now,
                    symbol=sym,
                    bid_price=cur_bid,
                    lot_size=pos.lot_size,
                    n_lots=pos.n_lots,
                )
                record = self.portfolio.close_position(
                    timestamp=now,
                    symbol=sym,
                    exit_price=fill.executed_price,
                    exit_reason=decision.reason.value,
                )
                if record:
                    self.journal.log_trade(
                        entry_time=record.entry_time,
                        exit_time=record.exit_time,
                        symbol=record.symbol,
                        underlying_symbol=record.underlying_symbol,
                        strike=record.strike,
                        expiry=record.expiry,
                        option_type=record.option_type.value,
                        n_lots=record.n_lots,
                        lot_size=record.lot_size,
                        entry_price=record.entry_price,
                        exit_price=record.exit_price,
                        gross_pnl=record.gross_pnl,
                        net_pnl=record.net_pnl,
                        total_fees=record.total_fees,
                        return_pct=record.return_pct,
                        holding_bars=record.holding_bars,
                        exit_reason=record.exit_reason,
                    )
                    self.risk_manager.update_capital(self.portfolio.total_equity)

        # 2. Check for new strategy entries
        risk_check = self.risk_manager.check_trade_allowed(
            current_open_count=len(self.portfolio.open_positions),
            proposed_premium_cost=2500.0,
        )

        if risk_check.allowed:
            signals = self.strategy.generate_signals(spot_history_df)
            for sig in signals:
                if sig.direction not in (SignalDirection.BUY_CALL, SignalDirection.BUY_PUT):
                    continue

                opt_type = (
                    OptionType.CALL if sig.direction == SignalDirection.BUY_CALL else OptionType.PUT
                )
                exp = ContractSelector.select_expiry(
                    current_chain_df, now, min_dte=sig.min_dte, max_dte=sig.max_dte
                )
                if exp is None:
                    continue

                contract = ContractSelector.select_strike_by_delta(
                    current_chain_df,
                    spot_price=spot_close,
                    expiry=exp,
                    option_type=opt_type,
                    target_delta=sig.target_delta,
                )
                if contract is None:
                    continue

                # Position sizing (1.5% fixed fractional)
                lots = PositionSizer.fixed_fractional(
                    capital=self.portfolio.cash,
                    risk_pct=1.5,
                    option_premium=contract.ask,
                    stop_loss_pct=40.0,
                    lot_size=spec.lot_size,
                    max_lots=3,
                )
                if lots <= 0:
                    continue

                fill = self.fill_sim.execute_buy(
                    timestamp=now,
                    symbol=contract.symbol,
                    ask_price=contract.ask,
                    lot_size=spec.lot_size,
                    n_lots=lots,
                )

                opened = self.portfolio.open_position(
                    timestamp=now,
                    symbol=contract.symbol,
                    underlying_symbol=symbol,
                    strike=contract.strike,
                    expiry=contract.expiry,
                    option_type=opt_type,
                    entry_price=fill.executed_price,
                    lot_size=spec.lot_size,
                    n_lots=lots,
                )
                if opened:
                    break

        # Log snapshot
        self.journal.log_equity_snapshot(
            timestamp=now,
            cash=self.portfolio.cash,
            total_equity=self.portfolio.total_equity,
            open_positions=len(self.portfolio.open_positions),
        )
