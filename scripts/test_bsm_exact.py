import numpy as np
from scipy.stats import norm

spot = 25800.0
k = 26100.0
r = 0.07

# If today is Nov 22, expiry Nov 26:
# Remaining calendar time = 4.0 days
t_years = 4.0 / 365.0 # 0.0109589
# iv for 14.80:
# Let's test IV ~ 8.9%
iv = 0.0891
d1 = (np.log(spot / k) + (r + 0.5 * (iv**2)) * t_years) / (iv * np.sqrt(t_years))
d2 = d1 - iv * np.sqrt(t_years)
call = spot * norm.cdf(d1) - k * np.exp(-r * t_years) * norm.cdf(d2)
print("Call price:", round(call, 2))
