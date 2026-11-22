import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

spot_price = 25800.0
k = 26100.0
r = 0.07
target_price = 14.80

# Let's solve for implied vol or DTE that yields 14.80
for dte_days in [1, 2, 3, 4, 5]:
    t = dte_days / 365.0
    def bsm_call(iv):
        d1 = (np.log(spot_price / k) + (r + 0.5 * (iv**2)) * t) / (iv * np.sqrt(t))
        d2 = d1 - iv * np.sqrt(t)
        return spot_price * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2) - target_price
    try:
        sol_iv = brentq(bsm_call, 0.01, 1.0)
        print(f"DTE={dte_days} days (calendar / 365): IV needed for 14.80 = {sol_iv*100:.2f}%")
    except Exception as e:
        pass

for dte_days in [1, 2, 3, 4, 5]:
    t = dte_days / 252.0
    def bsm_call(iv):
        d1 = (np.log(spot_price / k) + (r + 0.5 * (iv**2)) * t) / (iv * np.sqrt(t))
        d2 = d1 - iv * np.sqrt(t)
        return spot_price * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2) - target_price
    try:
        sol_iv = brentq(bsm_call, 0.01, 1.0)
        print(f"DTE={dte_days} trading days (trading / 252): IV needed for 14.80 = {sol_iv*100:.2f}%")
    except Exception as e:
        pass
