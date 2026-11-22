"""
StudyTrade — Strategy Signals
===============================
"""

from studytrade.strategy.signals.breakout import BreakoutStrategy
from studytrade.strategy.signals.debit_spread import DebitSpreadStrategy
from studytrade.strategy.signals.gamma_squeeze import GammaSqueezeStrategy
from studytrade.strategy.signals.momentum import MomentumStrategy
from studytrade.strategy.signals.vol_expansion import VolatilityExpansionStrategy

__all__ = [
    "BreakoutStrategy",
    "DebitSpreadStrategy",
    "GammaSqueezeStrategy",
    "MomentumStrategy",
    "VolatilityExpansionStrategy",
]
