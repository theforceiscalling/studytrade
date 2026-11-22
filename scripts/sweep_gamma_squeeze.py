"""
StudyTrade — 10-Year Gamma Squeeze & Dealer Cascade Parameter Sweep
===================================================================
Sweeps target deltas (0.25 Far OTM, 0.35 Directional OTM, 0.50 ATM)
across profit targets (50%, 60%, 70%, 80%, 90%, 100%)
for 10-year historical data (2016 - 2026).
Evaluates capital at ₹2,000 and ₹20,000, ranks performance,
and persists every run to data/backtest_runs.db.
"""

import sys
import os
import time
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, "src")

from studytrade.constants import TIMEZONE, INSTRUMENTS, OptionType
from studytrade.data.store import DataStore
from studytrade.strategy.signals.gamma_squeeze import GammaSqueezeStrategy
from studytrade.strategy.rules import ExitRuleEvaluator
from studytrade.strategy.selector import ContractSelector
from studytrade.strategy.base import SignalDirection
from studytrade.backtest.engine import BacktestEngine
from studytrade.backtest.portfolio import Portfolio
from studytrade.backtest.fills import FillSimulator
from studytrade.backtest.costs import compute_multi_broker_comparison
from studytrade.analytics.metrics import compute_performance_metrics
from studytrade.data.backtest_db import save_backtest_run


def get_or_compute_signals(spot_df, opt_df, cache_path="data/cache_10y_gamma_signals.pkl"):
    """Compute 10Y raw Gamma Squeeze & Cascade signals once and cache to disk."""
    if os.path.exists(cache_path):
        print(f"[CACHE] Loading precomputed signals from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("[COMPUTE] Precomputing 10-year Gamma Squeeze signals across all timestamps...")
    t0 = time.time()
    strat = GammaSqueezeStrategy(target_delta=0.35, min_dte=2, max_dte=20)
    chains_by_ts = dict(tuple(opt_df.groupby("timestamp")))

    # Pre-slice spot timestamps for fast lookup
    signals_dict = {}
    timestamps = spot_df["timestamp"].sort_values().unique()

    for i, cur_ts in enumerate(timestamps):
        cur_dt = pd.to_datetime(cur_ts).tz_convert(TIMEZONE)
        chain = chains_by_ts.get(cur_dt)
        if chain is None or chain.empty:
            continue

        spot_slice = spot_df.iloc[: i + 1]
        sigs = strat.generate_signals(spot_slice, chain)
        if sigs:
            signals_dict[cur_dt] = sigs

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(timestamps)} bars ({len(signals_dict)} signals found)...")

    print(f"[DONE] Precomputed {len(signals_dict)} active signal bars in {time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(signals_dict, f)
    return signals_dict


class FastBacktestEngine:
    """High-performance backtest engine using precomputed signals and indexed option quotes."""

    def __init__(
        self,
        target_delta: float,
        profit_target_pct: float,
        stop_loss_pct: float = 30.0,
        min_dte: int = 2,
        max_dte: int = 20,
        initial_capital: float = 2000.0,
        slippage_ticks: float = 1.0,
        max_concurrent_positions: int = 3,
    ):
        self.target_delta = target_delta
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.initial_capital = initial_capital
        self.slippage_ticks = slippage_ticks
        self.max_concurrent_positions = max_concurrent_positions

        self.exit_evaluator = ExitRuleEvaluator(
            stop_loss_pct=stop_loss_pct,
            profit_target_pct=profit_target_pct,
            min_dte_cutoff=min_dte,
        )
        self.fill_sim = FillSimulator(slippage_ticks=slippage_ticks)

    def run(self, spot_df, opt_df, signals_by_ts, symbol="NIFTY"):
        spec = INSTRUMENTS.get(symbol, INSTRUMENTS["NIFTY"])
        portfolio = Portfolio(initial_capital=self.initial_capital)

        chains_by_ts = dict(tuple(opt_df.groupby("timestamp")))
        empty_chain = opt_df.iloc[:0]
        timestamps = spot_df["timestamp"].sort_values().unique()

        for cur_ts in timestamps:
            cur_dt = pd.to_datetime(cur_ts).tz_convert(TIMEZONE)
            cur_chain = chains_by_ts.get(cur_dt, empty_chain)

            # 1. Evaluate open positions
            positions_to_close = []
            for sym, pos in list(portfolio.open_positions.items()):
                pos.bars_held += 1

                dte = max(0, (pos.expiry.date() - cur_dt.date()).days)

                if not cur_chain.empty:
                    m = cur_chain[cur_chain["symbol"] == sym]
                else:
                    m = empty_chain

                if m.empty:
                    if cur_dt.date() >= pos.expiry.date() or dte <= 0:
                        cur_spot = (
                            float(cur_chain["spot"].iloc[0])
                            if "spot" in cur_chain.columns and not cur_chain.empty
                            else float(spot_df[spot_df["timestamp"] <= cur_dt]["close"].iloc[-1])
                        )
                        intrinsic = (
                            max(0.0, cur_spot - pos.strike)
                            if pos.option_type == OptionType.CALL
                            else max(0.0, pos.strike - cur_spot)
                        )
                        positions_to_close.append((sym, intrinsic, "EXPIRED_SETTLED"))
                    continue

                latest_quote = m.iloc[-1]
                cur_bid = float(latest_quote["bid"])
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
                        timestamp=cur_dt,
                        symbol=sym,
                        bid_price=cur_bid,
                        lot_size=pos.lot_size,
                        n_lots=pos.n_lots,
                    )
                    positions_to_close.append((sym, fill.executed_price, decision.reason.value))

            for sym, exit_price, reason in positions_to_close:
                portfolio.close_position(cur_dt, sym, exit_price, reason)

            # 2. Check entry signals
            if len(portfolio.open_positions) < self.max_concurrent_positions:
                sigs = signals_by_ts.get(cur_dt, [])
                for sig in sigs:
                    if len(portfolio.open_positions) >= self.max_concurrent_positions:
                        break

                    is_call = sig.direction in (SignalDirection.BUY_CALL, SignalDirection.BULL_CALL_SPREAD)
                    is_put = sig.direction in (SignalDirection.BUY_PUT, SignalDirection.BEAR_PUT_SPREAD)
                    if not (is_call or is_put):
                        continue

                    opt_type = OptionType.CALL if is_call else OptionType.PUT
                    if cur_chain.empty:
                        continue

                    exp = ContractSelector.select_expiry(cur_chain, cur_dt, min_dte=self.min_dte, max_dte=self.max_dte)
                    if exp is None:
                        exp = ContractSelector.select_expiry(cur_chain, cur_dt, min_dte=max(0, self.min_dte - 2), max_dte=self.max_dte + 15)
                    if exp is None:
                        continue

                    contract = ContractSelector.select_strike_by_delta(
                        cur_chain,
                        spot_price=float(cur_chain["spot"].iloc[0]) if "spot" in cur_chain.columns else float(spot_df[spot_df["timestamp"] <= cur_dt]["close"].iloc[-1]),
                        expiry=exp,
                        option_type=opt_type,
                        target_delta=self.target_delta,
                    )
                    if contract is None:
                        continue

                    fill = self.fill_sim.execute_buy(
                        timestamp=cur_dt,
                        symbol=contract.symbol,
                        ask_price=contract.ask,
                        lot_size=spec.lot_size,
                        n_lots=1,
                    )

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

            portfolio.record_bar_snapshot(cur_dt)

        trades_df, equity_df = portfolio.to_dataframe()
        return trades_df, equity_df, portfolio


def main():
    store = DataStore("data")
    print("[LOAD] Loading 10-year spot and options for NIFTY...")
    spot_df = store.load_spot("NIFTY")
    opt_df = store.load_options("NIFTY")

    signals_by_ts = get_or_compute_signals(spot_df, opt_df)

    deltas = [
        (0.25, "Far OTM (Δ=0.25)"),
        (0.35, "Directional OTM (Δ=0.35)"),
        (0.50, "ATM Strike (Δ=0.50)"),
    ]
    profit_targets = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    # We will test ₹2,000 capital as requested, plus ₹20,000 to demonstrate full capacity
    capitals = [2000.0, 20000.0]

    all_results = []

    print("\n" + "=" * 90)
    print("RUNNING 10-YEAR PARAMETER SWEEP: GAMMA SQUEEZE & DEALER CASCADE (2016-2026)")
    print("=" * 90)

    for capital in capitals:
        print(f"\n--- TESTING INITIAL CAPITAL: ₹{capital:,.0f} ---")
        for delta_val, delta_label in deltas:
            for tp in profit_targets:
                engine = FastBacktestEngine(
                    target_delta=delta_val,
                    profit_target_pct=tp,
                    stop_loss_pct=30.0,
                    min_dte=2,
                    max_dte=20,
                    initial_capital=capital,
                )

                t0 = time.time()
                trades_df, equity_df, portfolio = engine.run(spot_df, opt_df, signals_by_ts)
                dur = time.time() - t0

                if trades_df.empty:
                    print(f"  {delta_label} | TP {tp:3.0f}% | 0 Trades taken (insufficient capital/no fill)")
                    continue

                perf = compute_performance_metrics(trades_df, equity_df, initial_capital=capital)
                broker_comp = compute_multi_broker_comparison(trades_df)

                params = {
                    "strategy": "gamma_squeeze",
                    "symbol": "NIFTY",
                    "start_date": "2016-01-01",
                    "end_date": "2026-11-20",
                    "initial_capital": capital,
                    "target_delta": delta_val,
                    "profit_target_pct": tp,
                    "stop_loss_pct": 30.0,
                    "min_dte": 2,
                    "max_dte": 20,
                }

                run_id = save_backtest_run(
                    symbol="NIFTY",
                    strategy="gamma_squeeze",
                    initial_capital=capital,
                    total_trades=perf.total_trades,
                    win_rate_pct=perf.win_rate_pct,
                    gross_pnl=broker_comp["gross_realized_pnl"],
                    broker_comp=broker_comp,
                    sharpe_ratio=perf.sharpe_ratio,
                    max_drawdown_pct=perf.max_drawdown_pct,
                    params=params,
                )

                res_dict = {
                    "run_id": run_id,
                    "capital": capital,
                    "delta": delta_val,
                    "delta_label": delta_label,
                    "profit_target": tp,
                    "total_trades": perf.total_trades,
                    "win_rate_pct": perf.win_rate_pct,
                    "gross_pnl": broker_comp["gross_realized_pnl"],
                    "zerodha_net": broker_comp["zerodha"]["net_pnl"],
                    "stoxkart_net": broker_comp["stoxkart"]["net_pnl"],
                    "savings": broker_comp["stoxkart_savings"],
                    "sharpe": perf.sharpe_ratio,
                    "profit_factor": perf.profit_factor,
                    "max_dd_pct": perf.max_drawdown_pct,
                }
                all_results.append(res_dict)

                print(
                    f"  [Run #{run_id:02d}] {delta_label:<24} | TP {tp:3.0f}% | "
                    f"Trades: {perf.total_trades:4d} | Win: {perf.win_rate_pct:5.1f}% | "
                    f"Sharpe: {perf.sharpe_ratio:5.2f} | "
                    f"Gross: ₹{broker_comp['gross_realized_pnl']:>10,.0f} | "
                    f"Stox Net: ₹{broker_comp['stoxkart']['net_pnl']:>10,.0f} ({dur:.1f}s)"
                )

    # Save summary dataframe
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("data/gamma_squeeze_sweep_results.csv", index=False)
    print("\nSweep complete! Saved summary to data/gamma_squeeze_sweep_results.csv")


if __name__ == "__main__":
    main()
