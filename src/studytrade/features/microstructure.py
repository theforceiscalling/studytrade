"""
StudyTrade — Options Microstructure Features (PCR, Max Pain, OI Dynamics)
===========================================================================

Computes market-wide and contract-level option chain microstructure metrics:
  - Put-Call Ratio (PCR) by Volume and Open Interest.
  - Max Pain (strike price causing maximum total option writer loss minimization).
  - Open Interest (OI) buildup classification (Long Buildup, Short Buildup, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PCRResult:
    """Put-Call Ratio summary."""

    pcr_oi: float
    pcr_volume: float
    total_call_oi: int
    total_put_oi: int
    total_call_vol: int
    total_put_vol: int


def compute_pcr(chain_df: pd.DataFrame) -> PCRResult:
    """
    Compute Put-Call Ratio (PCR) for an option chain snapshot.

    PCR_OI  = Total Put OI / Total Call OI
    PCR_Vol = Total Put Volume / Total Call Volume
    """
    if chain_df.empty:
        return PCRResult(np.nan, np.nan, 0, 0, 0, 0)

    calls = chain_df[chain_df["option_type"] == "CE"]
    puts = chain_df[chain_df["option_type"] == "PE"]

    call_oi = int(calls["open_interest"].sum())
    put_oi = int(puts["open_interest"].sum())
    call_vol = int(calls["volume"].sum())
    put_vol = int(puts["volume"].sum())

    pcr_oi = put_oi / max(1, call_oi)
    pcr_vol = put_vol / max(1, call_vol)

    return PCRResult(
        pcr_oi=float(pcr_oi),
        pcr_volume=float(pcr_vol),
        total_call_oi=call_oi,
        total_put_oi=put_oi,
        total_call_vol=call_vol,
        total_put_vol=put_vol,
    )


def compute_max_pain(chain_df: pd.DataFrame) -> float:
    """
    Calculate the Max Pain strike price.

    Max Pain Theory: At expiration, the underlying price tends to gravitate
    toward the strike where option writers (sellers) pay out the least total cash.

    Total Payout(K) = sum_i( Call_OI_i * max(0, K - Strike_i) )
                    + sum_j( Put_OI_j  * max(0, Strike_j - K) )
    Max Pain Strike = argmin_K Total Payout(K)
    """
    if chain_df.empty:
        return float("nan")

    strikes = np.sort(chain_df["strike"].unique())
    calls = chain_df[chain_df["option_type"] == "CE"][["strike", "open_interest"]]
    puts = chain_df[chain_df["option_type"] == "PE"][["strike", "open_interest"]]

    total_losses = np.zeros(len(strikes), dtype=np.float64)

    for idx, test_strike in enumerate(strikes):
        # Loss from Calls: intrinsic payout = max(0, test_strike - call_strike) * call_oi
        call_loss = np.sum(
            np.maximum(0.0, test_strike - calls["strike"].values) * calls["open_interest"].values
        )
        # Loss from Puts: intrinsic payout = max(0, put_strike - test_strike) * put_oi
        put_loss = np.sum(
            np.maximum(0.0, puts["strike"].values - test_strike) * puts["open_interest"].values
        )

        total_losses[idx] = call_loss + put_loss

    min_idx = int(np.argmin(total_losses))
    return float(strikes[min_idx])


def classify_oi_buildup(price_change: float, oi_change: float) -> str:
    """
    Classify contract / market Open Interest change dynamics.

    Returns one of:
      - 'LONG_BUILDUP': Price Up, OI Up (Aggressive buyers)
      - 'SHORT_BUILDUP': Price Down, OI Up (Aggressive sellers)
      - 'LONG_UNWINDING': Price Down, OI Down (Bulls giving up / stop losses)
      - 'SHORT_COVERING': Price Up, OI Down (Bears covering)
      - 'NEUTRAL': Negligible changes
    """
    eps = 1e-6
    if abs(price_change) < eps and abs(oi_change) < eps:
        return "NEUTRAL"

    if price_change > 0 and oi_change > 0:
        return "LONG_BUILDUP"
    elif price_change < 0 and oi_change > 0:
        return "SHORT_BUILDUP"
    elif price_change < 0 and oi_change < 0:
        return "LONG_UNWINDING"
    elif price_change > 0 and oi_change < 0:
        return "SHORT_COVERING"
    return "NEUTRAL"
