"""
StudyTrade — Position Exit and Trade Lifecycle Rules
======================================================

Strict exit rules tailored for systematic option buying:
  - Premium-based Stop Loss
  - Asymmetric Profit Target
  - Time-Stop (exit if breakout fails to materialize)
  - Exponential Theta Decay Cutoff (DTE stop)
  - Profit Trailing Stop
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExitReason(StrEnum):
    STOP_LOSS = "STOP_LOSS"
    PROFIT_TARGET = "PROFIT_TARGET"
    TIME_STOP = "TIME_STOP"
    THETA_DECAY = "THETA_DECAY"
    TRAILING_STOP = "TRAILING_STOP"
    NONE = "NONE"


@dataclass(frozen=True)
class ExitDecision:
    """Result of evaluating exit rules."""

    should_exit: bool
    reason: ExitReason
    exit_price: float
    description: str


class ExitRuleEvaluator:
    """
    Evaluates systematic exit rules for an open option buying position.
    """

    def __init__(
        self,
        stop_loss_pct: float = 40.0,  # Exit if down 40% on premium
        profit_target_pct: float = 100.0,  # Exit if up 100% (2x return)
        max_holding_bars: int = 5,  # Exit after 5 bars without profit
        min_dte_cutoff: int = 2,  # Exit if DTE <= 2 days
        trail_trigger_pct: float = 50.0,  # Activate trailing stop at +50% gain
        trail_giveback_pct: float = 25.0,  # Give back at most 25% of peak gain
    ) -> None:
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        self.max_holding_bars = max_holding_bars
        self.min_dte_cutoff = min_dte_cutoff
        self.trail_trigger_pct = trail_trigger_pct
        self.trail_giveback_pct = trail_giveback_pct

    def evaluate(
        self,
        entry_price: float,
        current_bid: float,
        peak_bid: float,
        bars_held: int,
        dte_days: int,
    ) -> ExitDecision:
        """
        Evaluate all exit rules against current market quote.
        """
        pnl_pct = ((current_bid - entry_price) / entry_price) * 100.0
        peak_pnl_pct = ((peak_bid - entry_price) / entry_price) * 100.0

        # 1. Hard Stop-Loss
        if pnl_pct <= -self.stop_loss_pct:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                exit_price=current_bid,
                description=f"Stop loss triggered: PnL {pnl_pct:.1f}% <= -{self.stop_loss_pct}%",
            )

        # 2. Profit Target (Asymmetric Reward:Risk)
        if pnl_pct >= self.profit_target_pct:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                exit_price=current_bid,
                description=(
                    f"Profit target attained: PnL {pnl_pct:.1f}% "
                    f">= +{self.profit_target_pct}%"
                ),
            )

        # 3. Theta Decay Cutoff (Avoid expiry-week gamma whipsaw / zero value burn)
        if dte_days <= self.min_dte_cutoff:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.THETA_DECAY,
                exit_price=current_bid,
                description=(
                    f"Theta decay cutoff triggered: DTE {dte_days} "
                    f"<= {self.min_dte_cutoff}"
                ),
            )

        # 4. Trailing Stop
        if peak_pnl_pct >= self.trail_trigger_pct:
            drawdown_from_peak = peak_pnl_pct - pnl_pct
            if drawdown_from_peak >= self.trail_giveback_pct:
                return ExitDecision(
                    should_exit=True,
                    reason=ExitReason.TRAILING_STOP,
                    exit_price=current_bid,
                    description=(
                        f"Trailing stop hit: {drawdown_from_peak:.1f}% giveback "
                        f"from peak +{peak_pnl_pct:.1f}%"
                    ),
                )

        # 5. Time Stop (If trade hasn't worked in N bars, cut it to protect against theta)
        if bars_held >= self.max_holding_bars and pnl_pct < 15.0:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.TIME_STOP,
                exit_price=current_bid,
                description=f"Time stop triggered: held {bars_held} bars without breakout",
            )

        return ExitDecision(
            should_exit=False,
            reason=ExitReason.NONE,
            exit_price=current_bid,
            description="Hold position",
        )
