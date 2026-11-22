"""
StudyTrade — Position Sizing Algorithms
=========================================

Systematic position sizing algorithms for option buying:
  - Fixed Fractional Risk (1-2% capital risk per trade)
  - Volatility-Adjusted (ATR-scaled) Sizing
  - Half-Kelly Criterion for Option Buying
"""

from __future__ import annotations

import numpy as np


class PositionSizer:
    """
    Computes integer lot size for option orders while strictly respecting capital constraints.
    """

    @staticmethod
    def fixed_fractional(
        capital: float,
        risk_pct: float,
        option_premium: float,
        stop_loss_pct: float,
        lot_size: int,
        max_lots: int = 10,
    ) -> int:
        """
        Fixed Fractional Position Sizing.

        Formulation:
          Max_Risk_INR = Capital * (risk_pct / 100)
          Risk_Per_Unit = Option_Premium * (stop_loss_pct / 100)
          Risk_Per_Lot = Risk_Per_Unit * Lot_Size
          Lots = floor(Max_Risk_INR / Risk_Per_Lot)

        Bounded by max_lots and capital limits.
        """
        if capital <= 0 or option_premium <= 0 or stop_loss_pct <= 0 or lot_size <= 0:
            return 0

        max_risk_inr = capital * (risk_pct / 100.0)
        risk_per_unit = option_premium * (stop_loss_pct / 100.0)
        risk_per_lot = risk_per_unit * lot_size

        if risk_per_lot <= 0:
            return 0

        raw_lots = int(np.floor(max_risk_inr / risk_per_lot))

        # Check total cash requirement (cannot exceed 80% of cash on 1 trade)
        max_cash_allowed = capital * 0.80
        cost_per_lot = option_premium * lot_size
        cash_bounded_lots = (
            int(np.floor(max_cash_allowed / cost_per_lot)) if cost_per_lot > 0 else 0
        )

        final_lots = min(raw_lots, cash_bounded_lots, max_lots)
        return max(1, final_lots) if (final_lots >= 1 and cost_per_lot <= capital) else 0

    @staticmethod
    def volatility_adjusted(
        capital: float,
        base_risk_pct: float,
        current_natr: float,
        baseline_natr: float,
        option_premium: float,
        stop_loss_pct: float,
        lot_size: int,
        max_lots: int = 10,
    ) -> int:
        """
        Volatility-Adjusted Position Sizing.
        Scales position size inversely with normalized volatility.
        When NATR spikes, position size decreases automatically.
        """
        if current_natr <= 0 or baseline_natr <= 0:
            return PositionSizer.fixed_fractional(
                capital, base_risk_pct, option_premium, stop_loss_pct, lot_size, max_lots
            )

        vol_scale = baseline_natr / current_natr
        adjusted_risk_pct = float(np.clip(base_risk_pct * vol_scale, 0.5, base_risk_pct * 1.5))

        return PositionSizer.fixed_fractional(
            capital, adjusted_risk_pct, option_premium, stop_loss_pct, lot_size, max_lots
        )

    @staticmethod
    def half_kelly(
        win_rate: float,
        payoff_ratio: float,
        capital: float,
        option_premium: float,
        stop_loss_pct: float,
        lot_size: int,
        max_lots: int = 10,
    ) -> int:
        """
        Half-Kelly Criterion Sizing.
          f* = (p * b - q) / b
          f_half = 0.5 * f*
        """
        p = win_rate if win_rate <= 1.0 else win_rate / 100.0
        q = 1.0 - p
        b = payoff_ratio

        if b <= 0 or (p * b - q) <= 0:
            return 0  # Negative expectancy -> zero lots

        f_star = (p * b - q) / b
        f_half = 0.5 * f_star
        kelly_risk_pct = float(np.clip(f_half * 100.0, 0.5, 3.0))

        return PositionSizer.fixed_fractional(
            capital, kelly_risk_pct, option_premium, stop_loss_pct, lot_size, max_lots
        )
