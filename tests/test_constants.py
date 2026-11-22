"""
Tests for studytrade.constants — Market constants and cost model.
"""

from __future__ import annotations

from studytrade.constants import (
    BANKNIFTY,
    DEFAULT_COSTS,
    INSTRUMENTS,
    NIFTY,
    OptionType,
)


class TestInstrumentSpecs:
    """Verify instrument specifications."""

    def test_nifty_lot_size(self):
        assert NIFTY.lot_size == 25

    def test_banknifty_lot_size(self):
        assert BANKNIFTY.lot_size == 15

    def test_nifty_strike_interval(self):
        assert NIFTY.strike_interval == 50.0

    def test_banknifty_strike_interval(self):
        assert BANKNIFTY.strike_interval == 100.0

    def test_registry_lookup(self):
        assert INSTRUMENTS["NIFTY"] is NIFTY
        assert INSTRUMENTS["BANKNIFTY"] is BANKNIFTY

    def test_instruments_are_frozen(self):
        """InstrumentSpec is a frozen dataclass — should not be mutable."""
        import dataclasses

        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            NIFTY.lot_size = 100  # type: ignore


class TestCostSchedule:
    """Verify the NSE cost model math."""

    def test_round_trip_cost_calculation(self):
        """
        Example: Buy 1 lot of NIFTY CE at ₹150, sell at ₹200.
        Verify total cost calculation is realistic.
        """
        costs = DEFAULT_COSTS
        lot_size = NIFTY.lot_size  # 25
        n_lots = 1

        buy_premium = 150.0
        sell_premium = 200.0

        buy_turnover = buy_premium * lot_size * n_lots  # 3,750
        sell_turnover = sell_premium * lot_size * n_lots  # 5,000
        total_turnover = buy_turnover + sell_turnover  # 8,750

        brokerage = 2 * costs.brokerage_per_order  # 40
        stt = sell_turnover * costs.stt_sell_rate  # 3.125
        exchange_txn = total_turnover * costs.exchange_txn_rate  # 4.33125
        sebi_fee = total_turnover * costs.sebi_rate  # 0.00875
        gst = (brokerage + exchange_txn) * costs.gst_rate  # 7.9796
        stamp_duty = buy_turnover * costs.stamp_duty_buy_rate  # 0.1125

        total_cost = brokerage + stt + exchange_txn + sebi_fee + gst + stamp_duty

        # Gross P&L: (200 - 150) × 25 = ₹1,250
        gross_pnl = (sell_premium - buy_premium) * lot_size * n_lots

        # Net P&L should be meaningfully less
        net_pnl = gross_pnl - total_cost

        assert gross_pnl == 1250.0
        assert total_cost > 50  # Costs are non-trivial
        assert total_cost < 100  # But not absurd for this size
        assert net_pnl < gross_pnl  # Costs eat into profits
        assert net_pnl > 0  # Still profitable on a 33% move

    def test_costs_all_non_negative(self):
        costs = DEFAULT_COSTS
        assert costs.brokerage_per_order >= 0
        assert costs.stt_sell_rate >= 0
        assert costs.exchange_txn_rate >= 0
        assert costs.sebi_rate >= 0
        assert costs.gst_rate >= 0
        assert costs.stamp_duty_buy_rate >= 0


class TestOptionType:
    def test_call_value(self):
        assert OptionType.CALL == "CE"

    def test_put_value(self):
        assert OptionType.PUT == "PE"
