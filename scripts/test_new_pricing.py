import numpy as np
from scipy.stats import norm

spot = 25800.0
k = 26100.0
r = 0.07
# Friday Nov 20 to Thursday Nov 26:
# Business days: Monday Nov 23, Tuesday Nov 24, Wednesday Nov 25, Thursday Nov 26 -> exactly 4 trading days
# In trading days / 252:
t_trading = 4 / 252.0

# If IV is around 10.5% (realistic for NIFTY 300-pt OTM CE):
for iv in [0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11]:
    d1 = (np.log(spot / k) + (r + 0.5 * (iv**2)) * t_trading) / (iv * np.sqrt(t_trading))
    d2 = d1 - iv * np.sqrt(t_trading)
    price = spot * norm.cdf(d1) - k * np.exp(-r * t_trading) * norm.cdf(d2)
    print(f"IV={iv*100:.1f}%, Price = {price:.2f}")

# And what if calendar days / 365 (e.g. 2-3 calendar days over weekend or 3 trading days)?
for dte in [2, 3, 4]:
    t = dte / 365.0
    for iv in [0.10, 0.11, 0.12]:
        d1 = (np.log(spot / k) + (r + 0.5 * (iv**2)) * t) / (iv * np.sqrt(t))
        d2 = d1 - iv * np.sqrt(t)
        price = spot * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
        print(f"DTE={dte} cal, IV={iv*100:.1f}%, Price = {price:.2f}")
