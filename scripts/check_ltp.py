from studytrade.data.store import DataStore
import pandas as pd

store = DataStore('data')
df = store.load_options('NIFTY')
print("Columns:", df.columns.tolist())
df['ts'] = pd.to_datetime(df['timestamp'])
max_ts = df['ts'].max()
print("Max ts:", max_ts)
latest = df[df['ts'] == max_ts]
print("Rows at max ts:", len(latest))
match = latest[latest['symbol'].str.contains('26100')]
print(match[['timestamp', 'symbol', 'strike', 'option_type', 'close', 'implied_volatility', 'bid', 'ask']].to_string())
