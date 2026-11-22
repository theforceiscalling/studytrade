from studytrade.data.store import DataStore
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath("web"))
from dashboard.views import _get_manual_trade_recommendation

store = DataStore('data')
df = store.load_options('NIFTY')
df['ts'] = pd.to_datetime(df['timestamp'])
latest_date = df['ts'].max()
target_chain = df[df['ts'] == latest_date]
rec = _get_manual_trade_recommendation(target_chain, "BUY_CALL", 25800.0, target_delta=0.35, symbol="NIFTY")
print("Recommendation result:")
for k, v in rec.items():
    print(f"  {k}: {v}")
