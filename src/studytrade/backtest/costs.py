"""
StudyTrade — NSE India Options Transaction Cost Engine
========================================================

Precise, circular-compliant calculation of all exchange, broker,
and government statutory charges for NSE equity derivatives (options).

Formulas:
---------
Turnover_buy  = Buy_Price * Lot_Size * Lots
Turnover_sell = Sell_Price * Lot_Size * Lots
Total_turnover = Turnover_buy + Turnover_sell

Brokerage    = 2 * brokerage_per_order (buy + sell orders)
STT          = Turnover_sell * 0.0625% (on sell side premium only)
Exchange_Txn = Total_turnover * 0.0495%
SEBI_Charges = Total_turnover * (10 / 10,000,000) (Rs 10 / crore)
GST          = (Brokerage + Exchange_Txn) * 18%
Stamp_Duty   = Turnover_buy * 0.003% (on buy side premium only)
Slippage     = 2 * (slippage_ticks * tick_size) * Lot_Size * Lots
"""

from __future__ import annotations

from dataclasses import dataclass

from studytrade.constants import DEFAULT_COSTS, NSECostSchedule


@dataclass(frozen=True)
class TradeCostBreakdown:
    """Detailed fee breakdown for a complete round-trip option trade."""

    turnover_buy: float
    turnover_sell: float
    total_turnover: float
    brokerage: float
    stt: float
    exchange_txn: float
    sebi_charges: float
    gst: float
    stamp_duty: float
    slippage_cost: float
    total_statutory_fees: float  # Brokerage + STT + Exch + SEBI + GST + Stamp
    total_costs_with_slippage: float


def compute_nse_option_costs(
    buy_price: float,
    sell_price: float,
    lot_size: int,
    n_lots: int = 1,
    slippage_ticks: float = 1.0,
    tick_size: float = 0.05,
    schedule: NSECostSchedule = DEFAULT_COSTS,
) -> TradeCostBreakdown:
    """
    Calculate statutory fees, brokerage, and slippage for a round-trip option trade.
    """
    units = lot_size * n_lots
    turnover_buy = float(buy_price * units)
    turnover_sell = float(sell_price * units)
    total_turnover = turnover_buy + turnover_sell

    # 1. Brokerage (flat per executed order)
    brokerage = float(2 * schedule.brokerage_per_order)

    # 2. STT (Securities Transaction Tax: 0.0625% on sell turnover)
    stt = float(turnover_sell * schedule.stt_sell_rate)

    # 3. NSE Exchange Transaction Charge (0.0495% on turnover)
    exchange_txn = float(total_turnover * schedule.exchange_txn_rate)

    # 4. SEBI Turnover Fee (Rs 10 per crore)
    sebi_charges = float(total_turnover * schedule.sebi_rate)

    # 5. GST (18% on Brokerage + Exchange Txn)
    gst = float((brokerage + exchange_txn) * schedule.gst_rate)

    # 6. Stamp Duty (0.003% on buy turnover)
    stamp_duty = float(turnover_buy * schedule.stamp_duty_buy_rate)

    total_statutory = brokerage + stt + exchange_txn + sebi_charges + gst + stamp_duty

    # 7. Slippage cost (buy slippage + sell slippage)
    slippage_per_unit = slippage_ticks * tick_size
    slippage_cost = float(2.0 * slippage_per_unit * units)

    total_with_slippage = total_statutory + slippage_cost

    return TradeCostBreakdown(
        turnover_buy=turnover_buy,
        turnover_sell=turnover_sell,
        total_turnover=total_turnover,
        brokerage=brokerage,
        stt=stt,
        exchange_txn=exchange_txn,
        sebi_charges=sebi_charges,
        gst=gst,
        stamp_duty=stamp_duty,
        slippage_cost=slippage_cost,
        total_statutory_fees=total_statutory,
        total_costs_with_slippage=total_with_slippage,
    )


def compute_multi_broker_comparison(trades_df) -> dict[str, Any]:
    """
    Compare exact transaction costs and net realized PnL across:
    1. Zerodha: Flat Rs 20/order (Rs 40 roundtrip) + statutory.
    2. Angel One: Flat Rs 20/order (Rs 40 roundtrip) + statutory.
    3. Stoxkart: Zero brokerage on losing trades, Flat Rs 15/order (Rs 30 roundtrip) on profitable trades only.
    """
    if trades_df is None or len(trades_df) == 0:
        empty_broker = {
            "brokerage": 0.0,
            "stt": 0.0,
            "exchange_txn": 0.0,
            "sebi": 0.0,
            "stamp_duty": 0.0,
            "gst": 0.0,
            "total_charges": 0.0,
            "net_pnl": 0.0,
        }
        return {
            "gross_realized_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "zerodha": empty_broker.copy(),
            "angel_one": empty_broker.copy(),
            "stoxkart": empty_broker.copy(),
            "stoxkart_savings": 0.0,
        }

    gross_realized = float(trades_df["gross_pnl"].sum())
    total_trades = len(trades_df)
    winning_trades = int((trades_df["gross_pnl"] > 0).sum())
    losing_trades = int((trades_df["gross_pnl"] <= 0).sum())

    # Shared statutory components
    stt_total = 0.0
    exchange_total = 0.0
    sebi_total = 0.0
    stamp_total = 0.0

    # Broker-specific brokerage accumulators
    zerodha_brokerage = 0.0
    angel_brokerage = 0.0
    stoxkart_brokerage = 0.0

    stoxkart_gst_base = 0.0
    standard_gst_base = 0.0

    for _, row in trades_df.iterrows():
        n_lots = row.get("n_lots", 1)
        lot_size = row.get("lot_size", 50)
        units = n_lots * lot_size
        buy_p = float(row.get("entry_price", 0.0))
        sell_p = float(row.get("exit_price", 0.0))
        gross_pnl = float(row.get("gross_pnl", 0.0))

        tb = buy_p * units
        ts = sell_p * units
        tot_turnover = tb + ts

        # Statutory calculations
        stt = ts * 0.000625  # 0.0625% on sell side premium
        exch = tot_turnover * 0.000495  # 0.0495%
        sebi = tot_turnover * 0.000001  # Rs 10 / crore
        stamp = tb * 0.00003  # 0.003% on buy side premium

        stt_total += stt
        exchange_total += exch
        sebi_total += sebi
        stamp_total += stamp

        # Zerodha: Rs 20 buy + Rs 20 sell = Rs 40 flat
        z_brok = 40.0
        zerodha_brokerage += z_brok
        standard_gst_base += (z_brok + exch)

        # Angel One: Rs 20 buy + Rs 20 sell = Rs 40 flat
        a_brok = 40.0
        angel_brokerage += a_brok

        # Stoxkart: Pay brokerage ONLY on profit! Rs 15 buy + Rs 15 sell = Rs 30 if profit, Rs 0 if loss
        if gross_pnl > 0:
            s_brok = 30.0
        else:
            s_brok = 0.0
        stoxkart_brokerage += s_brok
        stoxkart_gst_base += (s_brok + exch)

    zerodha_gst = standard_gst_base * 0.18
    angel_gst = standard_gst_base * 0.18
    stoxkart_gst = stoxkart_gst_base * 0.18

    zerodha_total = zerodha_brokerage + stt_total + exchange_total + sebi_total + stamp_total + zerodha_gst
    angel_total = angel_brokerage + stt_total + exchange_total + sebi_total + stamp_total + angel_gst
    stoxkart_total = stoxkart_brokerage + stt_total + exchange_total + sebi_total + stamp_total + stoxkart_gst

    zerodha_net = gross_realized - zerodha_total
    angel_net = gross_realized - angel_total
    stoxkart_net = gross_realized - stoxkart_total

    return {
        "gross_realized_pnl": round(gross_realized, 2),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "zerodha": {
            "brokerage": round(zerodha_brokerage, 2),
            "stt": round(stt_total, 2),
            "exchange_txn": round(exchange_total, 2),
            "sebi": round(sebi_total, 2),
            "stamp_duty": round(stamp_total, 2),
            "gst": round(zerodha_gst, 2),
            "total_charges": round(zerodha_total, 2),
            "net_pnl": round(zerodha_net, 2),
        },
        "angel_one": {
            "brokerage": round(angel_brokerage, 2),
            "stt": round(stt_total, 2),
            "exchange_txn": round(exchange_total, 2),
            "sebi": round(sebi_total, 2),
            "stamp_duty": round(stamp_total, 2),
            "gst": round(angel_gst, 2),
            "total_charges": round(angel_total, 2),
            "net_pnl": round(angel_net, 2),
        },
        "stoxkart": {
            "brokerage": round(stoxkart_brokerage, 2),
            "stt": round(stt_total, 2),
            "exchange_txn": round(exchange_total, 2),
            "sebi": round(sebi_total, 2),
            "stamp_duty": round(stamp_total, 2),
            "gst": round(stoxkart_gst, 2),
            "total_charges": round(stoxkart_total, 2),
            "net_pnl": round(stoxkart_net, 2),
        },
        "stoxkart_savings": round(zerodha_total - stoxkart_total, 2),
    }
