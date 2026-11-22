"""
StudyTrade — Feature Engineering Library
==========================================

Technical, volatility, options microstructure, and market regime features.
"""

from studytrade.features.gamma_exposure import (
    GEXProfile,
    compute_gex_profile,
    compute_option_gamma,
)
from studytrade.features.microstructure import (
    PCRResult,
    classify_oi_buildup,
    compute_max_pain,
    compute_pcr,
)
from studytrade.features.regime import MarketRegime, detect_regimes
from studytrade.features.technical import (
    compute_atr,
    compute_bollinger_bands,
    compute_donchian_breakout,
    compute_ema,
    compute_macd,
    compute_rsi,
)
from studytrade.features.volatility import (
    GARCH11,
    GARCHParams,
    compute_volatility_expansion_ratio,
    realized_vol_garman_klass,
    realized_vol_rogers_satchell,
)

__all__ = [
    "GARCH11",
    "GARCHParams",
    "GEXProfile",
    "MarketRegime",
    "PCRResult",
    "classify_oi_buildup",
    "compute_atr",
    "compute_bollinger_bands",
    "compute_donchian_breakout",
    "compute_ema",
    "compute_gex_profile",
    "compute_macd",
    "compute_max_pain",
    "compute_option_gamma",
    "compute_pcr",
    "compute_rsi",
    "compute_volatility_expansion_ratio",
    "detect_regimes",
    "realized_vol_garman_klass",
    "realized_vol_rogers_satchell",
]
