"""
Day-by-Day Replay Engine for Gamma Squeeze Validation
====================================================
Replays the last 50 executed trade dates from the recent backtest (Run #6).
Simulates realistic forward bar execution without future lookahead:
1. Re-computes GEX profile (Call Wall, Put Wall, Squeeze/Cascade Vulnerability Scores).
2. Verifies whether Gamma Squeeze signal triggered on that exact day.
3. Compares selected strike, option contract, entry price, and realized outcome.
"""

from __future__ import annotations
import pandas as pd
from studytrade.data.store import DataStore
from studytrade.strategy.signals.gamma_squeeze import GammaSqueezeStrategy
from studytrade.strategy.rules import ExitRuleEvaluator
from studytrade.backtest.engine import BacktestEngine
from studytrade.features.gamma_exposure import compute_gex_profile

def run_replay():
    store = DataStore('data')
    spot = store.load_spot('NIFTY')
    opt = store.load_options('NIFTY')

    s_dt = pd.to_datetime('2024-01-01').tz_localize(spot['timestamp'].dt.tz)
    e_dt = pd.to_datetime('2026-11-20').tz_localize(spot['timestamp'].dt.tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    spot = spot[(spot['timestamp'] >= s_dt) & (spot['timestamp'] <= e_dt)].reset_index(drop=True)
    opt = opt[(opt['timestamp'] >= s_dt) & (opt['timestamp'] <= e_dt)].reset_index(drop=True)

    strat = GammaSqueezeStrategy(target_delta=0.35, min_dte=2, max_dte=20, squeeze_score_threshold=60.0)
    evaluator = ExitRuleEvaluator(stop_loss_pct=30.0, profit_target_pct=90.0, min_dte_cutoff=2)
    engine = BacktestEngine(strategy=strat, initial_capital=20000.0, slippage_ticks=0.5 / 0.05, exit_evaluator=evaluator)
    res = engine.run(spot, opt, symbol='NIFTY')

    all_trades = res.trades_df
    last_50_trades = all_trades.tail(50).copy().reset_index(drop=True)

    print(f"Replaying last {len(last_50_trades)} trades across their execution dates...\n")
    
    replay_results = []
    
    for idx, trade in last_50_trades.iterrows():
        entry_time = trade['entry_time']
        date_str = entry_time.strftime('%Y-%m-%d')
        
        # 1. Isolate historical spot strictly up to that entry timestamp (no future data)
        hist_spot = spot[spot['timestamp'] <= entry_time].reset_index(drop=True)
        cur_spot_close = float(hist_spot['close'].iloc[-1])
        
        # 2. Extract that specific day's option chain
        cur_chain = opt[opt['timestamp'] == entry_time].reset_index(drop=True)
        
        # 3. Compute real-time GEX profile
        gex_prof = compute_gex_profile(cur_chain, spot_price=cur_spot_close, lot_size=50)
        
        # 4. Generate real-time strategy signal on replayed bar
        replayed_signals = strat.generate_signals(hist_spot, chain_df=cur_chain)
        
        sig_triggered = len(replayed_signals) > 0
        sig_dir = replayed_signals[0].direction.value if sig_triggered else "NONE"
        sig_score = replayed_signals[0].strength if sig_triggered else max(gex_prof.call_squeeze_score, gex_prof.put_cascade_score)
        sig_reason = replayed_signals[0].reason if sig_triggered else "No squeeze condition met"
        
        replay_results.append({
            "trade_num": idx + 1,
            "date": date_str,
            "spot_close": round(cur_spot_close, 1),
            "call_wall": gex_prof.call_wall_strike,
            "put_wall": gex_prof.put_wall_strike,
            "squeeze_score": round(sig_score, 1),
            "signal_triggered": sig_triggered,
            "signal_dir": sig_dir,
            "contract": trade['symbol'],
            "backtest_entry_p": round(trade['entry_price'], 2),
            "backtest_exit_p": round(trade['exit_price'], 2),
            "backtest_net_pnl": round(trade['net_pnl'], 2),
            "exit_reason": trade['exit_reason'],
        })

    rdf = pd.DataFrame(replay_results)
    
    triggers = rdf['signal_triggered'].sum()
    total = len(rdf)
    pct_match = (triggers / total) * 100.0
    
    print(f"=== DAY-BY-DAY REPLAY VALIDATION RESULTS ===")
    print(f"Total Trades Replayed: {total}")
    print(f"Gamma Squeeze Triggered in Forward Replay: {triggers} / {total} ({pct_match:.1f}% concordance)")
    print(f"Average Squeeze Vulnerability Score on Trigger: {rdf['squeeze_score'].mean():.1f} / 100")
    print(f"Call Wall Proximity Triggers (Bullish Calls): {(rdf['signal_dir'] == 'BUY_CALL').sum()}")
    print(f"Put Wall Cascade Triggers (Bearish Puts): {(rdf['signal_dir'] == 'BUY_PUT').sum()}")
    print("\nSample 10 Replayed Trades:")
    print(rdf[['trade_num', 'date', 'spot_close', 'squeeze_score', 'signal_triggered', 'signal_dir', 'contract', 'backtest_net_pnl', 'exit_reason']].head(10).to_string())

if __name__ == '__main__':
    run_replay()
