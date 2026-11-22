import sqlite3
import json
import pandas as pd

with sqlite3.connect('data/backtest_runs.db') as conn:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute('SELECT * FROM backtest_runs ORDER BY id ASC')
    rows = [dict(r) for r in cur.fetchall()]

results = []
for r in rows:
    p = json.loads(r.get('params_json', '{}'))
    ret_pct = (r['gross_pnl'] / r['initial_capital']) * 100.0
    results.append({
        'run_id': r['id'],
        'strategy': r['strategy'],
        'symbol': r['symbol'],
        'capital': r['initial_capital'],
        'trades': r['total_trades'],
        'win_rate': round(r['win_rate_pct'], 1),
        'gross_pnl': r['gross_pnl'],
        'net_pnl_stoxkart': r['stoxkart_net_pnl'],
        'return_pct': round(ret_pct, 1),
        'sharpe': r['sharpe_ratio'],
        'max_dd': r['max_drawdown_pct'],
        'start_date': p.get('start_date', '2016-01-01'),
        'end_date': p.get('end_date', '2026-11-20'),
        'target_delta': p.get('target_delta', 0.35),
        'stop_loss_pct': p.get('stop_loss_pct', 30.0),
        'profit_target_pct': p.get('profit_target_pct', 60.0),
        'min_dte': p.get('min_dte', 2),
        'max_dte': p.get('max_dte', 20),
    })

df = pd.DataFrame(results)
print("=== ALL RUNS PERFORMANCE SUMMARY ===")
print(df[['run_id', 'strategy', 'capital', 'trades', 'win_rate', 'return_pct', 'gross_pnl', 'net_pnl_stoxkart', 'sharpe', 'max_dd', 'target_delta', 'stop_loss_pct', 'profit_target_pct']].to_string())

print("\n=== STRATEGY-LEVEL COMPARISON ===")
grouped = df.groupby('strategy').agg({
    'win_rate': 'mean',
    'return_pct': 'mean',
    'gross_pnl': 'mean',
    'sharpe': 'mean',
    'max_dd': 'mean',
    'trades': 'sum'
})
print(grouped.to_string())
