"""
StudyTrade — Strategy Engine
==============================
"""

from studytrade.strategy.base import BaseStrategy, SignalDirection, TradeSignal
from studytrade.strategy.rules import ExitDecision, ExitReason, ExitRuleEvaluator
from studytrade.strategy.selector import (
    ContractSelector,
    DebitSpreadLegs,
    SelectedContract,
)
from studytrade.strategy.signals.breakout import BreakoutStrategy
from studytrade.strategy.signals.debit_spread import DebitSpreadStrategy
from studytrade.strategy.signals.momentum import MomentumStrategy
from studytrade.strategy.signals.vol_expansion import VolatilityExpansionStrategy

__all__ = [
    "BaseStrategy",
    "BreakoutStrategy",
    "ContractSelector",
    "DebitSpreadLegs",
    "DebitSpreadStrategy",
    "ExitDecision",
    "ExitReason",
    "ExitRuleEvaluator",
    "MomentumStrategy",
    "SelectedContract",
    "SignalDirection",
    "TradeSignal",
    "VolatilityExpansionStrategy",
]
