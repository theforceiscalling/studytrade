"""
StudyTrade — Strike and Expiry Selection Engine
=================================================

Selects option contracts matching strategy rules:
  - Expiry selection based on DTE bounds (avoids pin risk / extreme gamma).
  - Strike selection by target Delta (vectorized Greek inversion) or moneyness.
  - Liquidity filters (volume and Open Interest minimums).
  - Leg selector for defined-risk debit spreads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from studytrade.constants import TIMEZONE, OptionType
from studytrade.math.greeks import greeks


@dataclass(frozen=True)
class SelectedContract:
    """Option contract selected for trade entry."""

    symbol: str
    underlying_symbol: str
    expiry: datetime
    strike: float
    option_type: OptionType
    delta: float
    bid: float
    ask: float
    dte_days: int


@dataclass(frozen=True)
class DebitSpreadLegs:
    """Long and short legs for a defined-risk debit spread."""

    long_leg: SelectedContract
    short_leg: SelectedContract
    net_debit: float  # long_ask - short_bid


class ContractSelector:
    """
    Selects option strikes and expiries matching systematic criteria.
    """

    @staticmethod
    def select_expiry(
        chain_df: pd.DataFrame,
        current_time: datetime,
        min_dte: int = 5,
        max_dte: int = 35,
    ) -> datetime | None:
        """
        Select the most appropriate expiry date within [min_dte, max_dte].
        """
        if chain_df.empty:
            return None

        expiries = pd.to_datetime(chain_df["expiry"].unique())
        cur_ts = pd.to_datetime(current_time)
        cur_ts = cur_ts.tz_localize(TIMEZONE) if cur_ts.tz is None else cur_ts.tz_convert(TIMEZONE)

        valid_expiries = []
        for exp in expiries:
            exp_tz = exp.tz_localize(TIMEZONE) if exp.tz is None else exp.tz_convert(TIMEZONE)
            dte = (exp_tz.date() - cur_ts.date()).days
            if min_dte <= dte <= max_dte:
                valid_expiries.append((dte, exp_tz))

        if not valid_expiries:
            return None

        # Choose the earliest valid expiry (most liquid near-month / weekly)
        valid_expiries.sort(key=lambda x: x[0])
        return valid_expiries[0][1]

    @staticmethod
    def select_strike_by_delta(
        chain_df: pd.DataFrame,
        spot_price: float,
        expiry: datetime,
        option_type: OptionType,
        target_delta: float = 0.35,
        risk_free_rate: float = 0.07,
        min_volume: int = 50,
        min_oi: int = 500,
    ) -> SelectedContract | None:
        """
        Select the strike whose absolute Delta is closest to target_delta.
        """
        exp_ts = pd.to_datetime(expiry)
        exp_ts = exp_ts.tz_localize(TIMEZONE) if exp_ts.tz is None else exp_ts.tz_convert(TIMEZONE)

        mask = (
            (chain_df["expiry"] == exp_ts)
            & (chain_df["option_type"] == option_type.value)
            & (chain_df["volume"] >= min_volume)
            & (chain_df["open_interest"] >= min_oi)
        )
        candidates = chain_df[mask].copy()

        if candidates.empty:
            # Relax volume/OI constraint if needed
            mask_relaxed = (chain_df["expiry"] == exp_ts) & (
                chain_df["option_type"] == option_type.value
            )
            candidates = chain_df[mask_relaxed].copy()
            if candidates.empty:
                return None

        # Calculate analytical deltas
        cur_ts = candidates["timestamp"].iloc[0]
        dte_days = max(1, (exp_ts.date() - cur_ts.date()).days)
        t_years = max(1e-4, dte_days / 365.0)

        vols = candidates["implied_volatility"].fillna(0.16).values
        strikes = candidates["strike"].values

        g_res = greeks(
            option_type=option_type,
            s=spot_price,
            k=strikes,
            t=t_years,
            r=risk_free_rate,
            sigma=vols,
        )

        abs_deltas = np.abs(g_res.delta)
        delta_diff = np.abs(abs_deltas - abs(target_delta))
        best_idx = int(np.argmin(delta_diff))
        best_row = candidates.iloc[best_idx]

        return SelectedContract(
            symbol=str(best_row["symbol"]),
            underlying_symbol=str(best_row["underlying_symbol"]),
            expiry=exp_ts,
            strike=float(best_row["strike"]),
            option_type=option_type,
            delta=float(g_res.delta[best_idx]),
            bid=float(best_row["bid"]),
            ask=float(best_row["ask"]),
            dte_days=dte_days,
        )

    @classmethod
    def select_debit_spread(
        cls,
        chain_df: pd.DataFrame,
        spot_price: float,
        expiry: datetime,
        option_type: OptionType,
        long_delta: float = 0.50,  # ATM
        short_delta: float = 0.25,  # OTM
        risk_free_rate: float = 0.07,
    ) -> DebitSpreadLegs | None:
        """
        Select long and short legs for a defined-risk Bull Call or Bear Put spread.
        """
        long_leg = cls.select_strike_by_delta(
            chain_df,
            spot_price,
            expiry,
            option_type,
            target_delta=long_delta,
            risk_free_rate=risk_free_rate,
        )
        short_leg = cls.select_strike_by_delta(
            chain_df,
            spot_price,
            expiry,
            option_type,
            target_delta=short_delta,
            risk_free_rate=risk_free_rate,
        )

        if long_leg is None or short_leg is None:
            return None

        # For Call Spread: Long strike must be lower than Short strike
        # For Put Spread: Long strike must be higher than Short strike
        if option_type == OptionType.CALL and long_leg.strike >= short_leg.strike:
            return None
        if option_type == OptionType.PUT and long_leg.strike <= short_leg.strike:
            return None

        # Net debit: we buy long at Ask, sell short at Bid
        net_debit = long_leg.ask - short_leg.bid
        if net_debit <= 0:
            return None

        return DebitSpreadLegs(long_leg=long_leg, short_leg=short_leg, net_debit=net_debit)
