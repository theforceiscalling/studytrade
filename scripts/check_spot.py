from studytrade.data.store import DataStore
import pandas as pd

store = DataStore('data')
spot_df = store.load_spot('NIFTY')
print("Last spot rows:")
print(spot_df.tail(5)[['timestamp', 'open', 'high', 'low', 'close']])
