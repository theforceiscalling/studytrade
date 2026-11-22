"""
StudyTrade — Options Gamma Squeeze & Cascade Strategy
======================================================

Generates Call and Put buying signals (or defined-risk debit spreads)
by exploiting extreme dealer Gamma Exposure (GEX) positioning, Call/Put Wall
penetration, aggressive volume-to-OI surges, and near-expiry gamma acceleration.
"""

from __future__ import annotations

import pandas as pd

from studytrade.constants import DEFAULT_RISK_FREE_RATE, OptionType
from studytrade.features.gamma_exposure import compute_gex_profile
from studytrade.strategy.base import BaseStrategy, SignalDirection, TradeSignal


class GammaSqueezeStrategy(BaseStrategy):
    """
    Gamma Squeeze and Cascade Strategy.

    Detects imbalances in aggregate dealer gamma exposure where market makers
    are forced into pro-cyclical delta hedging:
    - Bullish Gamma Squeeze: Spot breaks through Call Wall + high call volume surge.
    - Bearish Gamma Cascade: Spot breaks below Put Wall + high put volume surge.
    """

    def __init__(
        self,
        squeeze_score_threshold: float = 65.0,
        target_delta: float = 0.40,
        min_dte: int = 0,
        max_dte: int = 4,
        use_spreads: bool = False,
        allow_bearish: bool = True,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ) -> None:
        """
        Initialize the Gamma Squeeze strategy.

        Parameters
        ----------
        squeeze_score_threshold : float, default 65.0
            Minimum composite vulnerability score (0 to 100) required to trigger a signal.
        target_delta : float, default 0.40
            Target delta for option strike selection (typically 0.35 - 0.45 for high gamma).
        min_dte : int, default 0
            Minimum days to expiration (0 allows 0DTE expiry trading).
        max_dte : int, default 4
            Maximum days to expiration (gamma is concentrated in short-dated contracts).
        use_spreads : bool, default False
            If True, generates defined-risk vertical spreads (BULL_CALL_SPREAD / BEAR_PUT_SPREAD).
            If False, generates naked single-leg options (BUY_CALL / BUY_PUT).
        allow_bearish : bool, default True
            If True, generates Put signals during bearish gamma cascade breakdowns.
        risk_free_rate : float, default 0.07
            Risk-free rate used for BSM gamma pricing.
        """
        super().__init__(name="Gamma_Squeeze")
        self.squeeze_score_threshold = squeeze_score_threshold
        self.target_delta = target_delta
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.use_spreads = use_spreads
        self.allow_bearish = allow_bearish
        self.risk_free_rate = risk_free_rate

    def generate_signals(
        self,
        spot_df: pd.DataFrame,
        chain_df: pd.DataFrame | None = None,
    ) -> list[TradeSignal]:
        """
        Analyze price action and option chain GEX profile to generate trade signals.

        Parameters
        ----------
        spot_df : pd.DataFrame
            Underlying OHLCV DataFrame up to decision point.
        chain_df : pd.DataFrame, optional
            Current option chain snapshot. If None or empty, returns empty list.

        Returns
        -------
        list of TradeSignal
        """
        if spot_df.empty or chain_df is None or chain_df.empty:
            return []

        cur_idx = len(spot_df) - 1
        cur_spot = float(spot_df["close"].iloc[cur_idx])
        cur_ts = spot_df["timestamp"].iloc[cur_idx]
        symbol = str(spot_df["symbol"].iloc[cur_idx]) if "symbol" in spot_df.columns else "NIFTY"

        # Filter option chain to relevant DTE window if expiry column is available
        working_chain = chain_df
        if "expiry" in chain_df.columns:
            exp_dates = pd.to_datetime(chain_df["expiry"]).dt.date
            cur_date = pd.to_datetime(cur_ts).date()
            dtes = (exp_dates - cur_date).apply(lambda d: max(0, d.days))

            # Filter chain to contracts within target DTE
            dte_mask = (dtes >= self.min_dte) & (dtes <= self.max_dte)
            if dte_mask.any():
                working_chain = chain_df[dte_mask]
            else:
                # Fallback to the nearest available expiry
                nearest_expiry = chain_df.loc[(dtes - self.min_dte).abs().idxmin(), "expiry"]
                working_chain = chain_df[chain_df["expiry"] == nearest_expiry]

        # Compute complete GEX Profile
        profile = compute_gex_profile(
            chain_df=working_chain,
            spot_price=cur_spot,
            risk_free_rate=self.risk_free_rate,
        )

        signals: list[TradeSignal] = []

        # 1. Bullish Gamma Squeeze Trigger
        if profile.call_squeeze_score >= self.squeeze_score_threshold:
            direction = (
                SignalDirection.BULL_CALL_SPREAD
                if self.use_spreads
                else SignalDirection.BUY_CALL
            )
            strength = min(1.0, profile.call_squeeze_score / 100.0)
            reason = (
                f"Bullish Gamma Squeeze: Score {profile.call_squeeze_score:.1f}/100, "
                f"Spot {cur_spot:.1f} vs Call Wall {profile.call_wall_strike:.1f}, "
                f"Flip {profile.gamma_flip_level:.1f}"
            )
            signals.append(
                TradeSignal(
                    timestamp=cur_ts,
                    underlying_symbol=symbol,
                    direction=direction,
                    strength=strength,
                    reason=reason,
                    target_delta=self.target_delta,
                    min_dte=self.min_dte,
                    max_dte=self.max_dte,
                )
            )

        # 2. Bearish Gamma Cascade Trigger
        if self.allow_bearish and profile.put_cascade_score >= self.squeeze_score_threshold:
            direction = (
                SignalDirection.BEAR_PUT_SPREAD
                if self.use_spreads
                else SignalDirection.BUY_PUT
            )
            strength = min(1.0, profile.put_cascade_score / 100.0)
            reason = (
                f"Bearish Gamma Cascade: Score {profile.put_cascade_score:.1f}/100, "
                f"Spot {cur_spot:.1f} vs Put Wall {profile.put_wall_strike:.1f}, "
                f"Flip {profile.gamma_flip_level:.1f}"
            )
            signals.append(
                TradeSignal(
                    timestamp=cur_ts,
                    underlying_symbol=symbol,
                    direction=direction,
                    strength=strength,
                    reason=reason,
                    target_delta=self.target_delta,
                    min_dte=self.min_dte,
                    max_dte=self.max_dte,
                )
            )

        return signals
