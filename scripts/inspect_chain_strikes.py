from studytrade.data.store import DataStore
import pandas as pd

store = DataStore('data')
df = store.load_options('NIFTY')
df['ts'] = pd.to_datetime(df['timestamp'])
latest_date = df['ts'].max()
day_df = df[df['ts'] == latest_date]
print("Unique expiries on latest date:", day_df['expiry'].unique())
nov26 = day_df[day_df['expiry'].astype(str).str.contains('2026-11-26')]
print("Count for 2026-11-26 expiry:", len(nov26))
calls = nov26[nov26['option_type'] == 'CE'].sort_values('strike')
print(calls[['symbol', 'strike', 'close', 'implied_volatility', 'bid', 'ask']].to_string())
