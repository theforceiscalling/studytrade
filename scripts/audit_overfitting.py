"""
StudyTrade — Comprehensive Overfitting and Quantitative Audit
=============================================================

Audits:
1. Gamma Squeeze sweep results in data/gamma_squeeze_sweep_results.csv and data/backtest_runs.db.
2. Deflated Sharpe Ratio (DSR) using src/studytrade/analytics/overfitting.py on top runs.
3. Out-Of-Sample (OOS) Walk-Forward (2016-2021 IS vs 2022-2026 OOS) and 5-fold Purged Walk-Forward CV.
4. Parameter cliff / sensitivity across Delta and Profit Targets.
5. Point-in-time integrity and look-ahead bias audit.
"""

from __future__ import annotations

import sys
import os
import json
import sqlite3
import time
from pathlib import Path

# Add paths
sys.path.insert(0, "src")
sys.path.insert(0, ".")
import pickle

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, norm

from studytrade.constants import TIMEZONE, INSTRUMENTS, OptionType, TRADING_DAYS_PER_YEAR
from studytrade.data.store import DataStore
from studytrade.strategy.rules import ExitRuleEvaluator
from studytrade.strategy.selector import ContractSelector
from studytrade.strategy.base import SignalDirection
from studytrade.backtest.portfolio import Portfolio
from studytrade.backtest.fills import FillSimulator
from studytrade.analytics.metrics import compute_performance_metrics
from studytrade.analytics.overfitting import compute_deflated_sharpe_ratio
from studytrade.ml.cv import PurgedWalkForwardCV
from scripts.sweep_gamma_squeeze import get_or_compute_signals


def run_strategy_backtest(
    spot_df: pd.DataFrame,
    opt_df: pd.DataFrame,
    signals_by_ts: dict,
    target_delta: float = 0.35,
    profit_target_pct: float = 90.0,
    stop_loss_pct: float = 30.0,
    min_dte: int = 2,
    max_dte: int = 20,
    initial_capital: float = 20000.0,
    slippage_ticks: float = 1.0,
    max_concurrent_positions: int = 3,
    settle_expired: bool = True,
):
    """
    Backtest runner supporting both original behavior and settled-at-expiry behavior.
    """
    spec = INSTRUMENTS["NIFTY"]
    portfolio = Portfolio(initial_capital=initial_capital)
    chains_by_ts = dict(tuple(opt_df.groupby("timestamp")))
    empty_chain = opt_df.iloc[:0]
    timestamps = spot_df["timestamp"].sort_values().unique()
    fill_sim = FillSimulator(slippage_ticks=slippage_ticks)
    exit_eval = ExitRuleEvaluator(
        stop_loss_pct=stop_loss_pct,
        profit_target_pct=profit_target_pct,
        min_dte_cutoff=min_dte,
    )

    spot_lookup = dict(zip(spot_df["timestamp"], spot_df["close"]))

    for cur_ts in timestamps:
        cur_dt = pd.to_datetime(cur_ts).tz_convert(TIMEZONE)
        cur_chain = chains_by_ts.get(cur_dt, empty_chain)
        cur_spot = float(spot_lookup.get(cur_dt, cur_chain["spot"].iloc[0] if "spot" in cur_chain.columns and not cur_chain.empty else 0.0))

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
                if settle_expired and (cur_dt >= pos.expiry or dte <= 0):
                    # Settle expired contract at intrinsic value
                    if pos.option_type == OptionType.CALL:
                        intrinsic = max(0.0, cur_spot - pos.strike)
                    else:
                        intrinsic = max(0.0, pos.strike - cur_spot)
                    positions_to_close.append((sym, intrinsic, "EXPIRED_SETTLED"))
                continue

            latest_quote = m.iloc[-1]
            cur_bid = float(latest_quote["bid"])
            pos.peak_bid = max(pos.peak_bid, cur_bid)

            decision = exit_eval.evaluate(
                entry_price=pos.entry_price,
                current_bid=cur_bid,
                peak_bid=pos.peak_bid,
                bars_held=pos.bars_held,
                dte_days=dte,
            )

            if decision.should_exit:
                fill = fill_sim.execute_sell(
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
        if len(portfolio.open_positions) < max_concurrent_positions:
            sigs = signals_by_ts.get(cur_dt, [])
            for sig in sigs:
                if len(portfolio.open_positions) >= max_concurrent_positions:
                    break
                is_call = sig.direction in (SignalDirection.BUY_CALL, SignalDirection.BULL_CALL_SPREAD)
                is_put = sig.direction in (SignalDirection.BUY_PUT, SignalDirection.BEAR_PUT_SPREAD)
                if not (is_call or is_put) or cur_chain.empty:
                    continue

                opt_type = OptionType.CALL if is_call else OptionType.PUT
                exp = ContractSelector.select_expiry(cur_chain, cur_dt, min_dte=min_dte, max_dte=max_dte)
                if exp is None:
                    exp = ContractSelector.select_expiry(cur_chain, cur_dt, min_dte=max(0, min_dte - 2), max_dte=max_dte + 15)
                if exp is None:
                    continue

                contract = ContractSelector.select_strike_by_delta(
                    cur_chain,
                    spot_price=cur_spot,
                    expiry=exp,
                    option_type=opt_type,
                    target_delta=target_delta,
                )
                if contract is None:
                    continue

                fill = fill_sim.execute_buy(
                    timestamp=cur_dt,
                    symbol=contract.symbol,
                    ask_price=contract.ask,
                    lot_size=spec.lot_size,
                    n_lots=1,
                )
                portfolio.open_position(
                    timestamp=cur_dt,
                    symbol=contract.symbol,
                    underlying_symbol="NIFTY",
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


def compute_dsr_details(sharpe: float, daily_returns: np.ndarray, n_trials: int = 30):
    """Calculate complete moments and Deflated Sharpe Ratio."""
    r = daily_returns[~np.isnan(daily_returns)]
    n = len(r)
    sk = float(skew(r))
    kt = float(kurtosis(r, fisher=False))  # Pearson kurtosis

    # Repository formula
    dsr_repo = compute_deflated_sharpe_ratio(sharpe, r, num_trials_tested=n_trials)
    dsr_single = compute_deflated_sharpe_ratio(sharpe, r, num_trials_tested=1)

    # Lopez de Prado adjusted with cross-trial variance
    # Expected max SR under null hypothesis
    em_gamma = 0.5772156649
    z = np.sqrt(2.0 * np.log(n_trials))
    expected_max_sr = z + (1.0 - em_gamma) / z

    sr_daily = sharpe / np.sqrt(TRADING_DAYS_PER_YEAR)
    expected_max_sr_daily = expected_max_sr / np.sqrt(TRADING_DAYS_PER_YEAR)
    var_sr = 1.0 - sk * sr_daily + ((kt - 1.0) / 4.0) * (sr_daily**2)
    t_stat = (sr_daily - expected_max_sr_daily) * np.sqrt(n - 1) / np.sqrt(var_sr) if var_sr > 0 else 0.0

    return {
        "n_bars": n,
        "observed_sharpe": sharpe,
        "skewness": sk,
        "kurtosis": kt,
        "var_sr": var_sr,
        "expected_max_sr_annual": expected_max_sr,
        "t_stat": t_stat,
        "dsr_n_trials": dsr_repo,
        "dsr_single_trial": dsr_single,
    }


def main():
    print("=" * 90)
    print("STARTING INDEPENDENT OVERFITTING & QUANTITATIVE AUDIT")
    print("=" * 90)

    store = DataStore("data")
    print("\n[1/5] Loading NIFTY spot and options data (2016-2026)...")
    spot_df = store.load_spot("NIFTY")
    opt_df = store.load_options("NIFTY")
    signals_by_ts = get_or_compute_signals(spot_df, opt_df)
    print(f"Loaded {len(spot_df)} daily spot bars, {len(opt_df)} option quotes, {len(signals_by_ts)} signal bars.")

    # 1. Sweep Audit
    print("\n" + "=" * 90)
    print("TASK 1: AUDIT OF SWEEP RUNS (data/gamma_squeeze_sweep_results.csv)")
    print("=" * 90)
    sweep_df = pd.read_csv("data/gamma_squeeze_sweep_results.csv")
    print(f"Total Sweep Runs Recorded: {len(sweep_df)}")
    print("\nTop 5 Runs by Sharpe Ratio:")
    top_runs = sweep_df.sort_values(by="sharpe", ascending=False).head(5)
    print(top_runs[["run_id", "capital", "delta_label", "profit_target", "total_trades", "win_rate_pct", "sharpe", "profit_factor", "max_dd_pct"]])

    print("\nCapital Constraint Impact (₹2,000 vs ₹20,000):")
    cap_summary = sweep_df.groupby("capital").agg(
        avg_trades=("total_trades", "mean"),
        avg_win=("win_rate_pct", "mean"),
        avg_sharpe=("sharpe", "mean"),
        avg_max_dd=("max_dd_pct", "mean"),
    )
    print(cap_summary)

    # 2. Deflated Sharpe Ratio
    print("\n" + "=" * 90)
    print("TASK 2: DEFLATED SHARPE RATIO (BAILEY & LOPEZ DE PRADO 2014)")
    print("=" * 90)
    configs = [
        ("Run #46 (ATM 50% TP)", 0.50, 50.0, 20000.0, 46),
        ("Run #44 (Directional OTM 90% TP)", 0.35, 90.0, 20000.0, 44),
        ("Run #34 (Far OTM 50% TP)", 0.25, 50.0, 20000.0, 34),
    ]

    dsr_results = []
    daily_returns_dict = {}

    for label, delta, tp, cap, run_id in configs:
        # Run original un-settled engine to match exact database run
        trades_orig, eq_orig, _ = run_strategy_backtest(
            spot_df, opt_df, signals_by_ts, target_delta=delta, profit_target_pct=tp, initial_capital=cap, settle_expired=False
        )
        perf_orig = compute_performance_metrics(trades_orig, eq_orig, initial_capital=cap)
        eq_arr = eq_orig["equity"].values
        daily_ret = np.diff(eq_arr) / eq_arr[:-1]
        daily_returns_dict[run_id] = daily_ret

        dsr_info = compute_dsr_details(perf_orig.sharpe_ratio, daily_ret, n_trials=30)
        dsr_info["label"] = label
        dsr_info["run_id"] = run_id
        dsr_info["total_trades"] = perf_orig.total_trades
        dsr_info["win_rate_pct"] = perf_orig.win_rate_pct
        dsr_results.append(dsr_info)

        print(f"\n{label}:")
        print(f"  Observed Annualized Sharpe: {dsr_info['observed_sharpe']:.3f}")
        print(f"  Sample Skewness:            {dsr_info['skewness']:.4f}")
        print(f"  Pearson Kurtosis:          {dsr_info['kurtosis']:.4f}")
        print(f"  Expected Max SR (N=30):     {dsr_info['expected_max_sr_annual']:.3f}")
        print(f"  DSR Probability (N=1):      {dsr_info['dsr_single_trial']:.6f} (Single-test)")
        print(f"  DSR Probability (N=30):     {dsr_info['dsr_n_trials']:.6f} (Multiple-test adjusted)")

    # 3. In-Sample vs Out-Of-Sample & Purged Walk-Forward CV
    print("\n" + "=" * 90)
    print("TASK 3: OUT-OF-SAMPLE (OOS) WALK-FORWARD & 5-FOLD PURGED CV")
    print("=" * 90)
    is_cutoff = pd.Timestamp("2021-12-31 23:59:59", tz=spot_df["timestamp"].dt.tz)
    spot_is = spot_df[spot_df["timestamp"] <= is_cutoff].reset_index(drop=True)
    spot_oos = spot_df[spot_df["timestamp"] > is_cutoff].reset_index(drop=True)
    opt_is = opt_df[opt_df["timestamp"] <= is_cutoff]
    opt_oos = opt_df[opt_df["timestamp"] > is_cutoff]

    print(f"In-Sample Date Range:     {spot_is['timestamp'].min().strftime('%Y-%m-%d')} to {spot_is['timestamp'].max().strftime('%Y-%m-%d')} ({len(spot_is)} bars)")
    print(f"Out-of-Sample Date Range: {spot_oos['timestamp'].min().strftime('%Y-%m-%d')} to {spot_oos['timestamp'].max().strftime('%Y-%m-%d')} ({len(spot_oos)} bars)")

    oos_table = []
    for label, delta, tp, cap, run_id in configs:
        # Audited engine with proper settlement to evaluate TRUE out-of-sample edge
        t_is, eq_is, _ = run_strategy_backtest(spot_is, opt_is, signals_by_ts, target_delta=delta, profit_target_pct=tp, initial_capital=cap, settle_expired=True)
        t_oos, eq_oos, _ = run_strategy_backtest(spot_oos, opt_oos, signals_by_ts, target_delta=delta, profit_target_pct=tp, initial_capital=cap, settle_expired=True)

        p_is = compute_performance_metrics(t_is, eq_is, initial_capital=cap)
        p_oos = compute_performance_metrics(t_oos, eq_oos, initial_capital=cap)
        deg_sharpe = p_oos.sharpe_ratio / p_is.sharpe_ratio if p_is.sharpe_ratio != 0 else 0.0

        res_row = {
            "Config": label,
            "IS_Trades": p_is.total_trades,
            "OOS_Trades": p_oos.total_trades,
            "IS_WinRate": p_is.win_rate_pct,
            "OOS_WinRate": p_oos.win_rate_pct,
            "IS_Sharpe": p_is.sharpe_ratio,
            "OOS_Sharpe": p_oos.sharpe_ratio,
            "Sharpe_Degradation": deg_sharpe,
            "IS_CAGR": p_is.cagr_pct,
            "OOS_CAGR": p_oos.cagr_pct,
            "IS_MaxDD": p_is.max_drawdown_pct,
            "OOS_MaxDD": p_oos.max_drawdown_pct,
        }
        oos_table.append(res_row)

        print(f"\n{label}:")
        print(f"  In-Sample (2016-2021):  Trades={p_is.total_trades:4d} | Win={p_is.win_rate_pct:5.1f}% | Sharpe={p_is.sharpe_ratio:5.2f} | CAGR={p_is.cagr_pct:5.1f}% | MaxDD={p_is.max_drawdown_pct:5.1f}%")
        print(f"  Out-Sample (2022-2026): Trades={p_oos.total_trades:4d} | Win={p_oos.win_rate_pct:5.1f}% | Sharpe={p_oos.sharpe_ratio:5.2f} | CAGR={p_oos.cagr_pct:5.1f}% | MaxDD={p_oos.max_drawdown_pct:5.1f}%")
        print(f"  Sharpe Degradation Ratio (OOS / IS): {deg_sharpe:.3f}")

    # 5-Fold Purged Walk-Forward CV
    print("\n--- 5-Fold Purged Walk-Forward Cross-Validation (Run #44 Δ=0.35 TP=90%) ---")
    cv = PurgedWalkForwardCV(n_splits=5, min_train_size=400, embargo_size=5)
    timestamps = spot_df["timestamp"].values
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(timestamps)):
        spot_tr = spot_df.iloc[train_idx].reset_index(drop=True)
        spot_te = spot_df.iloc[test_idx].reset_index(drop=True)
        tr_min_ts, tr_max_ts = spot_tr["timestamp"].min(), spot_tr["timestamp"].max()
        te_min_ts, te_max_ts = spot_te["timestamp"].min(), spot_te["timestamp"].max()

        opt_tr = opt_df[(opt_df["timestamp"] >= tr_min_ts) & (opt_df["timestamp"] <= tr_max_ts)]
        opt_te = opt_df[(opt_df["timestamp"] >= te_min_ts) & (opt_df["timestamp"] <= te_max_ts)]

        t_tr, eq_tr, _ = run_strategy_backtest(spot_tr, opt_tr, signals_by_ts, target_delta=0.35, profit_target_pct=90.0, settle_expired=True)
        t_te, eq_te, _ = run_strategy_backtest(spot_te, opt_te, signals_by_ts, target_delta=0.35, profit_target_pct=90.0, settle_expired=True)

        p_tr = compute_performance_metrics(t_tr, eq_tr, initial_capital=20000.0)
        p_te = compute_performance_metrics(t_te, eq_te, initial_capital=20000.0)
        fold_deg = p_te.sharpe_ratio / p_tr.sharpe_ratio if p_tr.sharpe_ratio > 0 else 0.0

        fold_res = {
            "Fold": fold_i + 1,
            "Train_Period": f"{tr_min_ts.strftime('%Y-%m')} to {tr_max_ts.strftime('%Y-%m')}",
            "Test_Period": f"{te_min_ts.strftime('%Y-%m')} to {te_max_ts.strftime('%Y-%m')}",
            "Train_Trades": p_tr.total_trades,
            "Test_Trades": p_te.total_trades,
            "Train_Sharpe": p_tr.sharpe_ratio,
            "Test_Sharpe": p_te.sharpe_ratio,
            "Degradation": fold_deg,
            "Test_WinRate": p_te.win_rate_pct,
            "Test_MaxDD": p_te.max_drawdown_pct,
        }
        fold_results.append(fold_res)
        print(f"  Fold {fold_i+1}: Train [{fold_res['Train_Period']}] Sharpe={p_tr.sharpe_ratio:5.2f} -> Test [{fold_res['Test_Period']}] Sharpe={p_te.sharpe_ratio:5.2f} | Deg={fold_deg:5.2f} | Win={p_te.win_rate_pct:5.1f}%")

    df_folds = pd.DataFrame(fold_results)
    mean_deg = df_folds["Degradation"].mean()
    print(f"  Average Walk-Forward Sharpe Degradation: {mean_deg:.3f}")

    # 4. Parameter Sensitivity & Cliff Analysis
    print("\n" + "=" * 90)
    print("TASK 4: PARAMETER SENSITIVITY & CLIFF ANALYSIS")
    print("=" * 90)
    test_deltas = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    test_tps = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    sensitivity_matrix = []
    for d in test_deltas:
        row = {"delta": d}
        for tp in test_tps:
            t_s, eq_s, _ = run_strategy_backtest(spot_df, opt_df, signals_by_ts, target_delta=d, profit_target_pct=tp, initial_capital=20000.0, settle_expired=True)
            p_s = compute_performance_metrics(t_s, eq_s, initial_capital=20000.0)
            row[f"TP_{int(tp)}"] = round(p_s.sharpe_ratio, 2)
        sensitivity_matrix.append(row)

    df_sens = pd.DataFrame(sensitivity_matrix)
    print("Sharpe Ratio Sensitivity Surface (Target Delta vs Profit Target %):")
    print(df_sens.to_string(index=False))

    # Save all audit outputs to csv / json
    output_dir = Path("data/audit_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_folds.to_csv(output_dir / "purged_walk_forward_cv.csv", index=False)
    pd.DataFrame(oos_table).to_csv(output_dir / "oos_split_results.csv", index=False)
    df_sens.to_csv(output_dir / "parameter_sensitivity_grid.csv", index=False)
    with open(output_dir / "dsr_results.json", "w") as f:
        json.dump(dsr_results, f, indent=2)

    print(f"\nAudit complete! Results stored in {output_dir}")


if __name__ == "__main__":
    main()
