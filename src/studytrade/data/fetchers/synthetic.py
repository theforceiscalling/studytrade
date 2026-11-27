"""
StudyTrade — Synthetic Market Data Generator
==============================================

Generates realistic underlying spot OHLCV and option chain time-series
with volatility smile, term structure, and microstructure bid/ask spreads.

Mathematical Foundation:
------------------------
1. Spot Generation via Geometric Brownian Motion (GBM):
   S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z_t)
   where Z_t ~ N(0, 1), dt = 1 / 252 (daily).

2. High / Low Generation via Intraday Variance Simulation:
   High = max(Open, Close) + |Z_high| * S * sigma * sqrt(dt) * 0.5
   Low  = min(Open, Close) - |Z_low|  * S * sigma * sqrt(dt) * 0.5

3. Volatility Skew & Smile:
   IV(K, T) = sigma_base - 0.15 * ln(K/S) + 0.35 * (ln(K/S))^2 + 0.02 / sqrt(T)
   (Reflects standard equity index negative skew where OTM puts are expensive).

4. Option Pricing (Black-Scholes):
   d1 = (ln(S/K) + (r + 0.5 * sigma^2)*T) / (sigma * sqrt(T))
   d2 = d1 - sigma * sqrt(T)
   Call = S * N(d1) - K * exp(-r*T) * N(d2)
   Put  = K * exp(-r*T) * N(-d2) - S * N(-d1)

5. Realistic Bid-Ask Spread:
   Spread = max(tick_size, 0.015 * Price + 0.03 * |K - S| / S * Price)
   Bid = max(tick_size, round_to_tick(Price - Spread / 2))
   Ask = max(Bid + tick_size, round_to_tick(Price + Spread / 2))
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from studytrade.constants import (
    INSTRUMENTS,
    TIMEZONE,
    TRADING_DAYS_PER_YEAR,
    OptionType,
)
from studytrade.data.fetchers.base import BaseDataFetcher
from studytrade.data.schemas import (
    OPTION_COLUMNS,
    SPOT_COLUMNS,
    validate_options_dataframe,
    validate_spot_dataframe,
)


def _round_to_tick(val: np.ndarray | float, tick: float = 0.05) -> np.ndarray | float:
    """Round value(s) to nearest market tick."""
    return np.round(val / tick) * tick


class SyntheticDataGenerator(BaseDataFetcher):
    """
    Generates synthetic spot and option chains for testing and research
    without requiring paid real-time market data subscriptions.
    """

    def __init__(
        self,
        seed: int = 42,
        risk_free_rate: float = 0.07,  # RBI 91-day T-Bill ~6.5-7.0%
    ) -> None:
        self.seed = seed
        self.r = risk_free_rate
        self.rng = np.random.default_rng(seed)

    def generate_spot(
        self,
        symbol: str = "NIFTY",
        start_date: str | datetime = "2024-01-01",
        n_days: int = 252,
        spot_start: float = 21000.0,
        annual_vol: float = 0.16,
        annual_drift: float = 0.10,
        target_spot_end: float | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic daily OHLCV bars for an underlying index.
        """
        dt = 1.0 / TRADING_DAYS_PER_YEAR
        spec = INSTRUMENTS.get(symbol, INSTRUMENTS["NIFTY"])

        if spot_start == 21000.0:
            if symbol == "BANKNIFTY":
                spot_start = 48000.0
            elif symbol == "FINNIFTY":
                spot_start = 22000.0
            elif symbol == "CRUDEOILM":
                spot_start = 6200.0
                if annual_vol == 0.16:
                    annual_vol = 0.32
            elif symbol == "NATURALGAS":
                spot_start = 240.0
                if annual_vol == 0.16:
                    annual_vol = 0.50

        # Generate business day timestamps in IST
        date_range = pd.date_range(
            start=start_date,
            periods=n_days,
            freq="B",  # Business days
            tz=TIMEZONE,
        )

        if target_spot_end is not None and n_days > 1:
            # Calibrated Brownian bridge path to hit exact realistic current market level
            total_log_return = float(np.log(target_spot_end / spot_start))
            raw_z = self.rng.standard_normal(n_days)
            raw_z = raw_z - np.mean(raw_z)
            cum_diff = np.cumsum(raw_z * annual_vol * np.sqrt(dt))
            cum_diff_adj = cum_diff - np.linspace(0, cum_diff[-1], n_days)
            trend = np.linspace(0, total_log_return, n_days)
            close_prices = spot_start * np.exp(trend + cum_diff_adj)
            # Ensure final close matches target exactly
            close_prices[-1] = target_spot_end
        else:
            # Standard normal random shocks
            z = self.rng.standard_normal(n_days)
            drift_term = (annual_drift - 0.5 * (annual_vol**2)) * dt
            diffusion_term = annual_vol * np.sqrt(dt) * z
            log_returns = drift_term + diffusion_term
            close_prices = spot_start * np.exp(np.cumsum(log_returns))
        open_prices = np.empty(n_days)
        open_prices[0] = spot_start
        open_prices[1:] = close_prices[:-1] * (1.0 + self.rng.normal(0, 0.002, n_days - 1))

        # Intraday High and Low
        z_high = np.abs(self.rng.standard_normal(n_days))
        z_low = np.abs(self.rng.standard_normal(n_days))
        intraday_vol_scale = annual_vol * np.sqrt(dt) * 0.6

        high_prices = np.maximum(open_prices, close_prices) + (
            z_high * open_prices * intraday_vol_scale
        )
        low_prices = np.minimum(open_prices, close_prices) - (
            z_low * open_prices * intraday_vol_scale
        )

        # Low cannot be negative, round to tick
        high_prices = _round_to_tick(high_prices, spec.tick_size)
        low_prices = _round_to_tick(np.maximum(1.0, low_prices), spec.tick_size)
        open_prices = _round_to_tick(open_prices, spec.tick_size)
        close_prices = _round_to_tick(close_prices, spec.tick_size)

        # Daily volume: log-normal distribution with correlation to absolute return
        abs_ret = np.abs(np.diff(close_prices, prepend=spot_start) / close_prices)
        base_vol = self.rng.lognormal(mean=14.5, sigma=0.4, size=n_days)
        volumes = (base_vol * (1.0 + 5.0 * abs_ret)).astype(int)

        df = pd.DataFrame(
            {
                "timestamp": date_range,
                "symbol": symbol,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volumes,
            }
        )[SPOT_COLUMNS]

        clean_df, _ = validate_spot_dataframe(df)
        return clean_df

    def generate_option_chain_snapshot(
        self,
        timestamp: datetime,
        underlying_symbol: str,
        spot_price: float,
        expiries: list[datetime],
        base_iv: float = 0.16,
        num_strikes_each_side: int = 10,
    ) -> pd.DataFrame:
        """
        Generate option quotes across multiple expiries and strikes for a single timestamp.
        """
        spec = INSTRUMENTS.get(underlying_symbol, INSTRUMENTS["NIFTY"])
        interval = spec.strike_interval
        atm_strike = round(spot_price / interval) * interval

        # Generate strike ladder
        strikes = np.arange(
            atm_strike - (num_strikes_each_side * interval),
            atm_strike + ((num_strikes_each_side + 1) * interval),
            interval,
        )

        if base_iv == 0.16:
            if underlying_symbol == "NIFTY":
                base_iv = 0.115
            elif underlying_symbol == "BANKNIFTY":
                base_iv = 0.138
            elif underlying_symbol == "FINNIFTY":
                base_iv = 0.125
            elif underlying_symbol == "CRUDEOILM":
                base_iv = 0.50
            elif underlying_symbol == "NATURALGAS":
                base_iv = 0.55
            else:
                base_iv = 0.125

        rows: list[dict] = []

        for exp in expiries:
            # Time to expiry in calendar years (matching standard BSM convention for calendar dte)
            dte_days = max(0.5, (exp.date() - timestamp.date()).days)
            t_years = max(dte_days / 365.0, 1e-4)

            for k in strikes:
                moneyness = np.log(k / spot_price)

                # Calibrated Volatility Smile / Skew for Indian Index Options:
                # OTM puts (moneyness < 0) have higher IV;
                # OTM calls (moneyness > 0) have flatter/rising IV
                skew_term = -0.18 * moneyness
                smile_term = 0.28 * (moneyness**2)
                term_term = 0.005 * np.log(1.0 + t_years)
                noise = self.rng.normal(0, 0.0003)
                iv = max(0.06, base_iv + skew_term + smile_term + term_term + noise)

                # Black-Scholes analytical price
                d1 = (np.log(spot_price / k) + (self.r + 0.5 * (iv**2)) * t_years) / (
                    iv * np.sqrt(t_years)
                )
                d2 = d1 - iv * np.sqrt(t_years)

                discount = np.exp(-self.r * t_years)

                call_theoretical = spot_price * norm.cdf(d1) - k * discount * norm.cdf(d2)
                put_theoretical = k * discount * norm.cdf(-d2) - spot_price * norm.cdf(-d1)

                for opt_type, theo in [
                    (OptionType.CALL.value, call_theoretical),
                    (OptionType.PUT.value, put_theoretical),
                ]:
                    price = max(spec.tick_size, float(_round_to_tick(theo, spec.tick_size)))

                    # Bid-ask spread expands for OTM options, low prices, and near expiry
                    spread_pct = 0.015 + 0.04 * abs(moneyness)
                    spread = max(
                        spec.tick_size, float(_round_to_tick(price * spread_pct, spec.tick_size))
                    )

                    bid = max(
                        spec.tick_size, float(_round_to_tick(price - spread / 2.0, spec.tick_size))
                    )
                    ask = max(
                        bid + spec.tick_size,
                        float(_round_to_tick(price + spread / 2.0, spec.tick_size)),
                    )

                    # Microstructure: OI and Volume decay away from ATM
                    oi_decay = np.exp(-12.0 * (moneyness**2))
                    open_interest = int(self.rng.lognormal(9.0, 0.5) * oi_decay * spec.lot_size)
                    volume = int(open_interest * self.rng.uniform(0.1, 0.8))

                    # Option symbol (e.g. NIFTY24DEC21500CE)
                    exp_str = exp.strftime("%y%b").upper()
                    opt_sym = f"{underlying_symbol}{exp_str}{int(k)}{opt_type}"

                    rows.append(
                        {
                            "timestamp": timestamp,
                            "symbol": opt_sym,
                            "underlying_symbol": underlying_symbol,
                            "expiry": exp,
                            "strike": float(k),
                            "option_type": opt_type,
                            "open": price,
                            "high": price * 1.02,
                            "low": price * 0.98,
                            "close": price,
                            "bid": bid,
                            "ask": ask,
                            "volume": volume,
                            "open_interest": open_interest,
                            "implied_volatility": round(iv, 4),
                        }
                    )

        df = pd.DataFrame(rows)[OPTION_COLUMNS]
        clean_df, _ = validate_options_dataframe(df)
        return clean_df

    def get_spot(
        self,
        symbol: str,
        start_date: datetime | str,
        end_date: datetime | str,
    ) -> pd.DataFrame:
        """Fetch synthetic spot data between dates."""
        start_dt = (
            pd.to_datetime(start_date).tz_localize(TIMEZONE)
            if pd.to_datetime(start_date).tz is None
            else pd.to_datetime(start_date)
        )
        end_dt = (
            pd.to_datetime(end_date).tz_localize(TIMEZONE)
            if pd.to_datetime(end_date).tz is None
            else pd.to_datetime(end_date)
        )
        n_days = max(1, (end_dt.date() - start_dt.date()).days)

        df = self.generate_spot(symbol=symbol, start_date=start_date, n_days=n_days)
        return df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].reset_index(
            drop=True
        )

    def get_options(
        self,
        underlying_symbol: str,
        start_date: datetime | str,
        end_date: datetime | str,
        expiries: list[datetime | str] | None = None,
        strikes: list[float] | None = None,
        spot_df: pd.DataFrame | None = None,
        num_strikes_each_side: int = 15,
    ) -> pd.DataFrame:
        """Fetch synthetic option data across spot history."""
        if spot_df is None:
            spot_df = self.get_spot(underlying_symbol, start_date, end_date)
        all_chains: list[pd.DataFrame] = []

        for _, row in spot_df.iterrows():
            ts = row["timestamp"]
            if underlying_symbol in ("CRUDEOILM", "NATURALGAS"):
                # MCX Commodity monthly expiry cycle (~16th for Crude, ~24th for Natural Gas)
                target_day = 16 if underlying_symbol == "CRUDEOILM" else 24
                exp_year = ts.year
                exp_month = ts.month
                if ts.day >= target_day - 2:
                    exp_month += 1
                    if exp_month > 12:
                        exp_month = 1
                        exp_year += 1
                cur_exp = ts.replace(year=exp_year, month=exp_month, day=target_day, hour=0, minute=0, second=0, microsecond=0)
                next_month = exp_month + 1
                next_year = exp_year
                if next_month > 12:
                    next_month = 1
                    next_year += 1
                next_exp = ts.replace(year=next_year, month=next_month, day=target_day, hour=0, minute=0, second=0, microsecond=0)
                active_expiries = [cur_exp, next_exp]
            else:
                # NSE Equity Indices: next 2 Thursdays
                cur_thursday = ts + timedelta(days=(3 - ts.weekday()) % 7)
                if cur_thursday <= ts:
                    cur_thursday += timedelta(days=7)
                next_thursday = cur_thursday + timedelta(days=7)
                active_expiries = [cur_thursday, next_thursday]

            if expiries is not None:
                parsed_exp = [pd.to_datetime(e).tz_convert(TIMEZONE) for e in expiries]
                active_expiries = [e for e in active_expiries if e in parsed_exp]

            if active_expiries:
                chain = self.generate_option_chain_snapshot(
                    timestamp=ts,
                    underlying_symbol=underlying_symbol,
                    spot_price=row["close"],
                    expiries=active_expiries,
                    num_strikes_each_side=num_strikes_each_side,
                )
                if strikes is not None:
                    chain = chain[chain["strike"].isin(strikes)]
                all_chains.append(chain)

        if not all_chains:
            return pd.DataFrame(columns=OPTION_COLUMNS)

        full_df = pd.concat(all_chains, ignore_index=True)
        clean_df, _ = validate_options_dataframe(full_df)
        return clean_df
