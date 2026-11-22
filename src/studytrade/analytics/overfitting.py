"""
StudyTrade — Overfitting Diagnostics and Deflated Sharpe Ratio
===============================================================

Statistical checks to guard against data-snooping and backtest overfitting:
  - Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
  - Out-of-sample performance degradation ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from studytrade.constants import TRADING_DAYS_PER_YEAR


def compute_deflated_sharpe_ratio(
    observed_sharpe: float,
    returns: pd.Series | np.ndarray,
    num_trials_tested: int = 20,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Calculate the Deflated Sharpe Ratio (DSR).

    Accounts for:
      1. Non-normal returns (negative skewness and high kurtosis typical of options).
      2. Multiple testing selection bias (searching across N parameter combinations).
      3. Short track record length.

    Formula (Bailey & Lopez de Prado, 2014):
      E[max(Sharpe)] ~= sqrt(2 * ln(N_trials)) + (1 - gamma) / sqrt(2 * ln(N_trials))
      DSR = Phi( (SR - E[max_SR]) * sqrt(n - 1) /
                 sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) )

    Returns
    -------
    float
        Probability (0.0 to 1.0) that the strategy's edge is real and not a fluke of data mining.
        DSR > 0.95 indicates a statistically significant edge after multiple testing.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    n = len(r)

    if n < 30 or num_trials_tested < 1:
        return 0.50

    # Compute higher moments
    skewness = float(skew(r))
    kurt = float(kurtosis(r, fisher=False))  # Pearson kurtosis (normal = 3)

    # Expected max Sharpe ratio under null hypothesis (Euler-Mascheroni constant gamma ~ 0.5772)
    em_gamma = 0.5772156649
    if num_trials_tested > 1:
        z = np.sqrt(2.0 * np.log(num_trials_tested))
        expected_max_sr = z + (1.0 - em_gamma) / z
    else:
        expected_max_sr = benchmark_sharpe

    # Annualize to daily units for variance calculation
    sr_daily = observed_sharpe / np.sqrt(TRADING_DAYS_PER_YEAR)
    expected_max_sr_daily = expected_max_sr / np.sqrt(TRADING_DAYS_PER_YEAR)

    var_sr = 1.0 - skewness * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily**2)
    if var_sr <= 0:
        return 0.50

    test_stat = (sr_daily - expected_max_sr_daily) * np.sqrt(n - 1) / np.sqrt(var_sr)
    dsr = float(norm.cdf(test_stat))

    return dsr
