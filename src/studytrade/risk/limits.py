"""
StudyTrade — Risk Limits and Emergency Kill Switch
===================================================

Enforces portfolio-level risk limits and circuit breakers:
  - Max concurrent open positions
  - Max daily loss limit (% of starting day capital)
  - Max weekly loss limit
  - Hard Emergency Kill Switch
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskCheckResult:
    """Approval or rejection of a proposed trade."""

    allowed: bool
    reason: str


class RiskManager:
    """
    Guards account survival by enforcing hard risk constraints.
    """

    def __init__(
        self,
        initial_capital: float = 500000.0,
        max_concurrent_positions: int = 3,
        max_daily_loss_pct: float = 5.0,
        max_weekly_loss_pct: float = 10.0,
        kill_switch_loss_pct: float = 15.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.max_concurrent_positions = max_concurrent_positions
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.kill_switch_loss_pct = kill_switch_loss_pct

        self.current_capital = initial_capital
        self.day_start_capital = initial_capital
        self.week_start_capital = initial_capital
        self.is_kill_switch_triggered = False

    def check_trade_allowed(
        self,
        current_open_count: int,
        proposed_premium_cost: float,
    ) -> RiskCheckResult:
        """
        Verify if a new trade is permitted under current risk limits.
        """
        if self.is_kill_switch_triggered:
            return RiskCheckResult(
                allowed=False,
                reason="KILL SWITCH ACTIVATED: Trading halted due to maximum loss breach.",
            )

        # 1. Concurrent position limit
        if current_open_count >= self.max_concurrent_positions:
            return RiskCheckResult(
                allowed=False,
                reason=(
                    f"Max concurrent positions reached "
                    f"({current_open_count}/{self.max_concurrent_positions})"
                ),
            )

        # 2. Daily loss limit check
        daily_loss_inr = self.day_start_capital - self.current_capital
        daily_loss_pct = (daily_loss_inr / self.day_start_capital) * 100.0
        if daily_loss_pct >= self.max_daily_loss_pct:
            return RiskCheckResult(
                allowed=False,
                reason=f"Daily loss limit hit: {daily_loss_pct:.1f}% >= {self.max_daily_loss_pct}%",
            )

        # 3. Weekly loss limit check
        weekly_loss_inr = self.week_start_capital - self.current_capital
        weekly_loss_pct = (weekly_loss_inr / self.week_start_capital) * 100.0
        if weekly_loss_pct >= self.max_weekly_loss_pct:
            return RiskCheckResult(
                allowed=False,
                reason=(
                    f"Weekly loss limit hit: {weekly_loss_pct:.1f}% "
                    f">= {self.max_weekly_loss_pct}%"
                ),
            )

        # 4. Solvency check
        if proposed_premium_cost > self.current_capital * 0.90:
            return RiskCheckResult(
                allowed=False, reason="Insufficient cash margin to purchase contract"
            )

        return RiskCheckResult(allowed=True, reason="Risk checks passed")

    def update_capital(self, new_capital: float) -> None:
        """Update current portfolio equity and check kill switch."""
        self.current_capital = new_capital
        total_loss_pct = ((self.initial_capital - new_capital) / self.initial_capital) * 100.0

        if total_loss_pct >= self.kill_switch_loss_pct:
            self.is_kill_switch_triggered = True

    def reset_day(self, capital: float) -> None:
        """Reset starting daily capital benchmark at market open."""
        self.day_start_capital = capital
        self.current_capital = capital

    def reset_week(self, capital: float) -> None:
        """Reset starting weekly capital benchmark at Monday market open."""
        self.week_start_capital = capital
        self.day_start_capital = capital
        self.current_capital = capital
