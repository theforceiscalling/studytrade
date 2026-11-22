"""
StudyTrade — Volatility Estimators and GARCH(1,1) Forecasting
==============================================================

Advanced volatility analytics for systematic option buying:
  - Multi-estimator Realized Volatility (Close-to-Close, Parkinson, Garman-Klass, Rogers-Satchell).
  - GARCH(1,1) conditional volatility estimation and forward term forecasting.
  - Volatility Expansion indicators (Volatility Ratio, Squeeze Breakouts).

Mathematical Formulations:
--------------------------
1. Garman-Klass (1980) OHLC Volatility:
   Var_GK = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
   Efficiency ~8x higher than close-to-close.

2. Rogers-Satchell (1991) Volatility (handles non-zero drift):
   Var_RS = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)

3. GARCH(1,1) Model:
   sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
   Stationarity requires: alpha + beta < 1
   Long-run variance: V_L = omega / (1 - alpha - beta)
   k-step ahead forecast:
     E[sigma_{t+k}^2] = V_L + (alpha + beta)^k * (sigma_t^2 - V_L)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from studytrade.constants import TRADING_DAYS_PER_YEAR


def realized_vol_garman_klass(
    open_: pd.Series | np.ndarray,
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    window: int = 20,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Garman-Klass OHLC Realized Volatility estimator."""
    o = pd.Series(open_) if not isinstance(open_, pd.Series) else open_
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l_ = pd.Series(low) if not isinstance(low, pd.Series) else low
    c = pd.Series(close) if not isinstance(close, pd.Series) else close

    hl_term = 0.5 * (np.log(h / l_) ** 2)
    co_term = (2.0 * np.log(2.0) - 1.0) * (np.log(c / o) ** 2)
    var_gk = hl_term - co_term
    rolling_vol = np.sqrt(var_gk.rolling(window=window).mean() * annualization_factor)

    return rolling_vol


def realized_vol_rogers_satchell(
    open_: pd.Series | np.ndarray,
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    window: int = 20,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Rogers-Satchell Realized Volatility estimator (unbiased under drift)."""
    o = pd.Series(open_) if not isinstance(open_, pd.Series) else open_
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l_ = pd.Series(low) if not isinstance(low, pd.Series) else low
    c = pd.Series(close) if not isinstance(close, pd.Series) else close

    term1 = np.log(h / c) * np.log(h / o)
    term2 = np.log(l_ / c) * np.log(l_ / o)
    var_rs = term1 + term2
    rolling_vol = np.sqrt(
        np.maximum(0.0, var_rs.rolling(window=window).mean()) * annualization_factor
    )

    return rolling_vol


@dataclass(frozen=True)
class GARCHParams:
    """Parameters of fitted GARCH(1,1) process."""

    omega: float
    alpha: float
    beta: float
    long_run_vol: float
    persistence: float  # alpha + beta


class GARCH11:
    """
    GARCH(1,1) Volatility Model with Variance Targeting.
    """

    def __init__(self) -> None:
        self.params: GARCHParams | None = None
        self._last_variance: float = 0.0

    def fit(self, returns: pd.Series | np.ndarray) -> GARCHParams:
        """
        Fit GARCH(1,1) parameters via Maximum Likelihood Estimation with variance targeting.
        """
        r = np.asarray(returns, dtype=np.float64)
        r = r[~np.isnan(r)]
        n = len(r)
        if n < 30:
            raise ValueError("Insufficient observations for GARCH(1,1) fit (minimum 30)")

        # Demean returns
        eps = r - np.mean(r)
        sample_var = float(np.var(eps, ddof=1))

        # Variance targeting: omega = sample_var * (1 - alpha - beta)
        def neg_log_likelihood(params: np.ndarray) -> float:
            alpha, beta = params
            if alpha <= 0 or beta <= 0 or (alpha + beta) >= 0.999:
                return 1e10

            omega = sample_var * (1.0 - alpha - beta)
            var_t = sample_var
            nll = 0.0

            for t in range(n):
                var_t = omega + alpha * (eps[t] ** 2) + beta * var_t
                if var_t <= 1e-12:
                    return 1e10
                nll += 0.5 * (np.log(var_t) + (eps[t] ** 2) / var_t)

            return float(nll)

        # Optimize alpha and beta
        initial_guess = np.array([0.08, 0.88])
        bounds = [(0.01, 0.30), (0.60, 0.98)]
        res = minimize(neg_log_likelihood, initial_guess, bounds=bounds, method="L-BFGS-B")

        alpha_opt, beta_opt = res.x
        omega_opt = sample_var * (1.0 - alpha_opt - beta_opt)
        long_run_vol = np.sqrt(sample_var * TRADING_DAYS_PER_YEAR)
        persistence = alpha_opt + beta_opt

        self.params = GARCHParams(
            omega=float(omega_opt),
            alpha=float(alpha_opt),
            beta=float(beta_opt),
            long_run_vol=float(long_run_vol),
            persistence=float(persistence),
        )

        # Compute last conditional variance
        var_t = sample_var
        for t in range(n):
            var_t = omega_opt + alpha_opt * (eps[t] ** 2) + beta_opt * var_t
        self._last_variance = var_t

        return self.params

    def forecast(self, horizon: int = 5) -> np.ndarray:
        """
        Forecast conditional annualized volatility for k = 1..horizon days ahead.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before forecasting")

        long_run_var = self.params.omega / (1.0 - self.params.persistence)
        pers = self.params.persistence

        forecasted_vars = np.empty(horizon)
        var_t = self._last_variance

        for k in range(1, horizon + 1):
            f_var = long_run_var + (pers**k) * (var_t - long_run_var)
            forecasted_vars[k - 1] = np.sqrt(f_var * TRADING_DAYS_PER_YEAR)

        return forecasted_vars


def compute_volatility_expansion_ratio(
    atr_short: pd.Series | np.ndarray,
    atr_long: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Volatility Expansion Ratio:
      VR = ATR_short / ATR_long
    VR > 1.25 signals an active volatility expansion cycle where option buying pays off.
    VR < 0.80 signals volatility contraction / chop.
    """
    s_short = pd.Series(atr_short) if not isinstance(atr_short, pd.Series) else atr_short
    s_long = pd.Series(atr_long) if not isinstance(atr_long, pd.Series) else atr_long
    return s_short / np.maximum(s_long, 1e-12)
