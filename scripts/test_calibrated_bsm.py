import numpy as np
from scipy.stats import norm

spot_price = 25800.0
k = 26100.0
r = 0.07
base_iv = 0.115
dte_days = 4 # calendar days from Sunday to Thursday
t_years = dte_days / 365.0

moneyness = np.log(k / spot_price)
skew_term = -0.15 * moneyness
smile_term = 0.25 * (moneyness**2)
term_term = 0.005 * np.log(1.0 + t_years)
iv = max(0.06, base_iv + skew_term + smile_term + term_term)

d1 = (np.log(spot_price / k) + (r + 0.5 * (iv**2)) * t_years) / (iv * np.sqrt(t_years))
d2 = d1 - iv * np.sqrt(t_years)
call_theo = spot_price * norm.cdf(d1) - k * np.exp(-r * t_years) * norm.cdf(d2)

print(f"Calculated IV: {iv*100:.2f}%")
print(f"Call Theo Price: {call_theo:.2f}")
