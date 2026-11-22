import pandas as pd
import numpy as np
from scipy.stats import norm
from studytrade.data.store import DataStore

store = DataStore('data')

def recalibrate_symbol(symbol: str, spot: float, base_iv: float, dte_days: float = 3.5, r: float = 0.07):
    df = store.load_options(symbol)
    df['ts'] = pd.to_datetime(df['timestamp'])
    max_ts = df['ts'].max()
    t = dte_days / 365.0

    mask = (df['ts'] == max_ts) & (df['expiry'].astype(str).str.contains('2026-11-26'))
    print(f"Recalibrating {symbol} at {max_ts}, spot={spot}, total matching rows={mask.sum()}")

    for idx in df[mask].index:
        k = float(df.at[idx, 'strike'])
        opt_type = str(df.at[idx, 'option_type'])
        m = np.log(k / spot)

        # Realistic smile/skew
        skew = -0.16 * m
        smile = 0.25 * (m**2)
        iv = max(0.08, base_iv + skew + smile)

        d1 = (np.log(spot / k) + (r + 0.5 * iv**2) * t) / (iv * np.sqrt(t))
        d2 = d1 - iv * np.sqrt(t)

        if opt_type == 'CE':
            theo = spot * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
        else:
            # Puts have slightly higher IV
            put_iv = iv + 0.015
            d1_p = (np.log(spot / k) + (r + 0.5 * put_iv**2) * t) / (put_iv * np.sqrt(t))
            d2_p = d1_p - put_iv * np.sqrt(t)
            theo = k * np.exp(-r * t) * norm.cdf(-d2_p) - spot * norm.cdf(-d1_p)
            iv = put_iv

        price = max(0.05, round(round(theo / 0.05) * 0.05, 2))
        df.at[idx, 'close'] = price
        df.at[idx, 'bid'] = max(0.05, round(price - 0.25, 2))
        df.at[idx, 'ask'] = round(price + 0.25, 2)
        df.at[idx, 'implied_volatility'] = round(iv, 4)

    df = df.drop(columns=['ts'])
    store.save_options(df, symbol)
    print(f"Saved recalibrated {symbol} options!")

# Recalibrate BANKNIFTY (spot = 53,500, base_iv = 0.135)
recalibrate_symbol('BANKNIFTY', spot=53500.0, base_iv=0.135)

# Recalibrate FINNIFTY (spot = 24,800, base_iv = 0.115)
recalibrate_symbol('FINNIFTY', spot=24800.0, base_iv=0.115)

# Print verification
for sym in ['BANKNIFTY', 'FINNIFTY']:
    df = store.load_options(sym)
    df['ts'] = pd.to_datetime(df['timestamp'])
    latest = df[(df['ts'] == df['ts'].max()) & (df['expiry'].astype(str).str.contains('2026-11-26')) & (df['option_type'] == 'CE')].sort_values('strike')
    print(f"=== {sym} CE Sample Strikes ===")
    print(latest[['symbol', 'strike', 'close', 'bid', 'ask', 'implied_volatility']].iloc[::3].to_string())
