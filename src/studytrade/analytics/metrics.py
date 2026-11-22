"""
StudyTrade — Performance and Risk Metrics
===========================================

Comprehensive performance and risk analytics:
  - Win Rate, Expectancy, Profit Factor, Payoff Ratio
  - Annualized Sharpe Ratio, Sortino Ratio, Calmar Ratio
  - Max Drawdown (MDD), Drawdown Duration, Equity Curve metrics
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from studytrade.constants import TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class PerformanceSummary:
    """Comprehensive performance statistics container."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_inr: float
    avg_win_inr: float
    avg_loss_inr: float
    payoff_ratio: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    avg_holding_bars: float


def compute_drawdowns(equity_series: pd.Series | np.ndarray) -> tuple[pd.Series, float, int]:
    """
    Calculate underwater drawdown series, max drawdown percentage, and max duration.

    Returns (drawdown_series, max_drawdown_pct, max_duration_bars).
    """
    eq = pd.Series(equity_series) if not isinstance(equity_series, pd.Series) else equity_series
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max * 100.0
    max_dd = float(drawdown.min())

    # Duration calculation
    is_zero = drawdown == 0.0
    duration = 0
    max_duration = 0
    for z in is_zero:
        if z:
            duration = 0
        else:
            duration += 1
            max_duration = max(max_duration, duration)

    return drawdown, abs(max_dd), max_duration


def compute_performance_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float = 500000.0,
    risk_free_rate: float = 0.07,
) -> PerformanceSummary:
    """
    Calculate performance and risk metrics from trade records and equity curve.
    """
    if trades_df.empty or equity_df.empty:
        return PerformanceSummary(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            expectancy_inr=0.0,
            avg_win_inr=0.0,
            avg_loss_inr=0.0,
            payoff_ratio=0.0,
            cagr_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            calmar_ratio=0.0,
            avg_holding_bars=0.0,
        )

    pnls = trades_df["net_pnl"].values
    n_trades = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = (n_wins / n_trades) * 100.0

    sum_wins = float(np.sum(wins)) if n_wins > 0 else 0.0
    sum_losses = float(np.sum(np.abs(losses))) if n_losses > 0 else 0.0
    profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else (99.0 if sum_wins > 0 else 0.0)

    avg_win = float(np.mean(wins)) if n_wins > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losses))) if n_losses > 0 else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    # Expectancy = (Win_Rate * Avg_Win) - (Loss_Rate * Avg_Loss)
    p_win = n_wins / n_trades
    p_loss = n_losses / n_trades
    expectancy = (p_win * avg_win) - (p_loss * avg_loss)

    avg_holding = float(trades_df["holding_bars"].mean())

    # Daily returns on equity curve
    equity_series = equity_df["equity"].values
    daily_returns = np.diff(equity_series) / equity_series[:-1]
    n_days = max(1, len(daily_returns))

    total_return = (equity_series[-1] - initial_capital) / initial_capital
    cagr = ((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0) * 100.0

    # Sharpe ratio
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_ret = daily_returns - rf_daily
    std_ret = float(np.std(daily_returns, ddof=1)) if n_days > 1 else 1e-6
    sharpe = (
        float(np.mean(excess_ret) / std_ret * np.sqrt(TRADING_DAYS_PER_YEAR))
        if std_ret > 1e-12
        else 0.0
    )

    # Sortino ratio
    downside_returns = daily_returns[daily_returns < rf_daily] - rf_daily
    downside_std = (
        float(np.sqrt(np.mean(downside_returns**2))) if len(downside_returns) > 0 else 1e-6
    )
    sortino = (
        float(np.mean(excess_ret) / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))
        if downside_std > 1e-12
        else 0.0
    )

    # Drawdown
    _, max_dd, _ = compute_drawdowns(equity_series)
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0

    return PerformanceSummary(
        total_trades=n_trades,
        winning_trades=n_wins,
        losing_trades=n_losses,
        win_rate_pct=float(win_rate),
        profit_factor=float(profit_factor),
        expectancy_inr=float(expectancy),
        avg_win_inr=float(avg_win),
        avg_loss_inr=float(avg_loss),
        payoff_ratio=float(payoff_ratio),
        cagr_pct=float(cagr),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown_pct=float(max_dd),
        calmar_ratio=float(calmar),
        avg_holding_bars=float(avg_holding),
    )
