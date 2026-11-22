import numpy as np
from scipy.stats import norm

spot_price = 25800.0
k = 26100.0
r = 0.07

# In synthetic.py currently:
dte_days = 6 # calendar days
# t_years in synthetic: 6 / 252 = 0.0238095
t_years_trading = 6 / 252.0
# calendar t_years:
# Note: In option markets, DTE for weekly expiry on Friday -> Thursday is 4 trading days or calendar 6 days / 365
iv_synthetic = 0.2527
d1 = (np.log(spot_price / k) + (r + 0.5 * (iv_synthetic**2)) * t_years_trading) / (iv_synthetic * np.sqrt(t_years_trading))
d2 = d1 - iv_synthetic * np.sqrt(t_years_trading)
call_synthetic = spot_price * norm.cdf(d1) - k * np.exp(-r * t_years_trading) * norm.cdf(d2)
print("Synthetic formula result:", call_synthetic)

# Now, why is the real market LTP 14.80?
# In real market:
# 1) NIFTY real ATM IV / OTM IV is ~11.5% - 12.5% (India VIX is ~12-13%).
# 2) DTE: Friday to Thursday expiry is 4 trading days (or 6 calendar days / 365 = 0.0164 years, or 4 / 252 = 0.01587 years).
for test_iv in [0.10, 0.11, 0.115, 0.12, 0.125, 0.13, 0.14]:
    for t_test, label in [(4/252.0, "4 trading days / 252"), (6/365.0, "6 cal days / 365"), (4/365.0, "4 cal days / 365")]:
        d1 = (np.log(spot_price / k) + (r + 0.5 * (test_iv**2)) * t_test) / (test_iv * np.sqrt(t_test))
        d2 = d1 - test_iv * np.sqrt(t_test)
        call = spot_price * norm.cdf(d1) - k * np.exp(-r * t_test) * norm.cdf(d2)
        print(f"IV={test_iv:.3f}, {label}: Call = {call:.2f}")
