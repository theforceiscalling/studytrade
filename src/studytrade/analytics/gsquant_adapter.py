"""
Optional gs-quant integration.

Only the offline ``gs_quant.timeseries`` analytics are used; they need no Goldman
Sachs Marquee credentials. Install with ``pip install -e ".[gsquant]"``.
"""

from __future__ import annotations

import pandas as pd

try:
    from gs_quant.timeseries import volatility as _gs_volatility

    GS_QUANT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _gs_volatility = None
    GS_QUANT_AVAILABLE = False


def realized_vol(prices: pd.Series, window: int | None = None) -> float | None:
    """
    Latest annualised realized volatility (decimal, e.g. 0.14) of a price series
    with a DatetimeIndex, computed by gs_quant. Returns None if gs-quant is not
    installed or there is too little data.
    """
    if not GS_QUANT_AVAILABLE or len(prices) < 3:
        return None
    series = _gs_volatility(prices, w=window or min(len(prices) - 1, 20))
    series = series.dropna()
    return float(series.iloc[-1]) / 100.0 if len(series) else None
