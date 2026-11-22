"""
Script to expand history from 2016-01-01 to 2026-11-20 (2,841 trading days).
"""
import time
from datetime import datetime
import pandas as pd
from studytrade.constants import TIMEZONE
from studytrade.data.sample_generator import generate_and_store_dataset
from studytrade.data.store import DataStore

start_date = "2016-01-01"
b_days = pd.date_range(start=start_date, end="2026-11-20", freq="B", tz=TIMEZONE)
n_days = len(b_days)
print(f"Target: {n_days} business days ({b_days[0].date()} to {b_days[-1].date()})")

configs = [
    ("NIFTY", 7900.0, 25800.0),
    ("BANKNIFTY", 16800.0, 53500.0),
    ("FINNIFTY", 7800.0, 24800.0),
]

for sym, start_price, end_price in configs:
    t0 = time.time()
    print(f"\nGenerating {sym} from {start_date} (start spot: {start_price} -> end spot: {end_price})...")
    spot, opt = generate_and_store_dataset(
        symbol=sym,
        start_date=start_date,
        n_days=n_days,
        spot_start=start_price,
        target_spot_end=end_price,
        base_dir="data",
    )
    elapsed = time.time() - t0
    print(f"Done {sym}: {len(spot)} spot bars ({spot['timestamp'].min().date()} to {spot['timestamp'].max().date()}), {len(opt)} option rows in {elapsed:.2f}s")

store = DataStore("data")
print("\nUpdated Catalog:")
for _, r in store.get_catalog().iterrows():
    print(f"  {r['symbol']} ({r['dataset_type']}): {r['row_count']} rows [{str(r['start_date'])[:10]} to {str(r['end_date'])[:10]}]")
