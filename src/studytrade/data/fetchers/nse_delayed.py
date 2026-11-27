"""
StudyTrade — Official NSE India Website Option Chain Delayed Feed
==================================================================

Fetches official delayed option chain snapshots directly from nseindia.com (via Option Chain v3).
Extracts top-of-book bid, ask, last traded price (LTP), implied volatility (IV),
and open interest (OI) for NIFTY and BANKNIFTY contracts.

Streams live/delayed quotes directly into:
1. `LiveTickCache` for sub-second in-memory matching.
2. `paper_open_positions` in `paper_journal.db` for mark-to-market PnL and auto-square-off evaluation.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from curl_cffi import requests

from studytrade.live.session_engine import get_live_tick_cache
from studytrade.paper.journal import PaperJournal

logger = logging.getLogger("studytrade.nse_delayed")


class NSEOptionChainFetcher:
    """
    Scrapes official NSE India Option Chain (Option Chain v3) with browser impersonation
    and populates LiveTickCache & PaperJournal.
    """

    BASE_URL = "https://www.nseindia.com"
    CHAIN_PAGE = "https://www.nseindia.com/option-chain"
    CONTRACT_INFO_API = "https://www.nseindia.com/api/option-chain-contract-info?symbol={symbol}"
    CHAIN_V3_API = "https://www.nseindia.com/api/option-chain-v3?type={chain_type}&symbol={symbol}&expiry={expiry}"
    INDEX_SYMBOLS = frozenset({"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"})

    @classmethod
    def chain_type(cls, symbol: str) -> str:
        """NSE distinguishes index ('Indices') and stock ('Equity') option chains."""
        return "Indices" if symbol.upper() in cls.INDEX_SYMBOLS else "Equity"

    def __init__(self, cache_ttl_seconds: int = 45) -> None:
        self.cache_ttl_seconds = cache_ttl_seconds
        self._session: requests.Session | None = None
        self._last_fetch_time: dict[str, float] = {}
        self._cached_results: dict[str, dict[str, Any]] = {}
        self._last_sync_stats: dict[str, Any] = {
            "last_synced": None,
            "ticks_updated": 0,
            "status": "IDLE",
        }

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session(impersonate="chrome120")
            self._session.headers.update({
                "Referer": self.CHAIN_PAGE,
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "application/json, text/plain, */*",
            })
            # Warm up session with option chain page to establish Akamai cookies
            try:
                self._session.get(self.CHAIN_PAGE, timeout=10)
            except Exception as e:
                logger.warning(f"Failed to initialize NSE session cookies: {e}")
        return self._session

    @staticmethod
    def format_contract_symbol(underlying: str, expiry_str: str, strike: float, opt_type: str) -> str:
        """
        Formats contract symbol matching StudyTrade convention:
        e.g. 'NIFTY26NOV25950CE'
        """
        dt = None
        for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(expiry_str, fmt)
                break
            except ValueError:
                pass
        if dt is None:
            dt = datetime.now()

        strike_int = int(strike) if strike == int(strike) else strike
        mon = dt.strftime("%b").upper()
        day = dt.strftime("%d")
        return f"{underlying.upper()}{day}{mon}{strike_int}{opt_type.upper()}"

    def fetch_index_chain(self, symbol: str = "NIFTY", expiry: str | None = None) -> dict[str, Any] | None:
        """
        Fetch option chain snapshot for an index from NSE website.
        Respects `cache_ttl_seconds` to prevent excessive requests to NSE.
        """
        now = time.time()
        cache_key = f"{symbol}|{expiry or ''}"
        # Return cached result if within TTL
        if cache_key in self._last_fetch_time and (now - self._last_fetch_time[cache_key]) < self.cache_ttl_seconds:
            return self._cached_results.get(cache_key)

        sess = self._get_session()
        try:
            target_expiry = expiry
            if not target_expiry:
                info_url = self.CONTRACT_INFO_API.format(symbol=symbol)
                r_info = sess.get(info_url, timeout=10)
                if r_info.status_code == 200:
                    exp_dates = r_info.json().get("expiryDates", [])
                    if exp_dates:
                        target_expiry = exp_dates[0]
                elif r_info.status_code in (401, 403, 404):
                    # Session expired or blocked, re-initialize
                    self._session = None
                    sess = self._get_session()
                    r_info = sess.get(info_url, timeout=10)
                    if r_info.status_code == 200:
                        exp_dates = r_info.json().get("expiryDates", [])
                        if exp_dates:
                            target_expiry = exp_dates[0]

            if not target_expiry:
                logger.warning(f"No expiry date resolved for {symbol}")
                return None

            # Fetch Option Chain v3
            chain_url = self.CHAIN_V3_API.format(
                chain_type=self.chain_type(symbol), symbol=symbol, expiry=target_expiry
            )
            r_chain = sess.get(chain_url, timeout=12)
            if r_chain.status_code != 200:
                logger.warning(f"NSE chain API returned status {r_chain.status_code} for {symbol}")
                return None

            raw_data = r_chain.json()
            records = raw_data.get("records", {})
            spot_price = float(records.get("underlyingValue") or 0.0)

            result = {
                "symbol": symbol,
                "spot_price": spot_price,
                "expiry": target_expiry,
                "timestamp": records.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "strikes": records.get("data", []),
            }
            self._last_fetch_time[cache_key] = now
            self._cached_results[cache_key] = result
            return result
        except Exception as exc:
            logger.error(f"Error fetching NSE option chain for {symbol}: {exc}")
            return None

    def sync_to_tick_cache(
        self,
        symbols: list[str] = ["NIFTY", "BANKNIFTY"],
        journal: PaperJournal | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Pull latest delayed quotes from NSE and update LiveTickCache & PaperJournal.
        Matches both exact contract symbols and open position attributes (underlying, strike, option_type).
        """
        if force:
            self._last_fetch_time.clear()

        tick_cache = get_live_tick_cache()
        updated_count = 0
        jnl = journal or PaperJournal()

        open_positions = jnl.get_open_positions()

        # Build attribute lookup for open positions: (underlying, strike, opt_type) -> list of pos dicts
        open_pos_attr_map: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
        for p in open_positions:
            und = str(p.get("underlying_symbol", "")).upper()
            stk = float(p.get("strike", 0.0))
            ot = str(p.get("option_type", "")).upper()
            key = (und, stk, ot)
            open_pos_attr_map.setdefault(key, []).append(p)

        spots_synced: dict[str, float] = {}

        for sym in symbols:
            chain_data = self.fetch_index_chain(sym)
            if not chain_data:
                continue

            spot = chain_data["spot_price"]
            exp = chain_data["expiry"]
            if spot > 0:
                spots_synced[sym] = spot
                tick_cache.update_tick(
                    sym,
                    {
                        "symbol": sym,
                        "last_price": spot,
                        "is_fresh": True,
                        "timestamp": chain_data["timestamp"],
                    },
                )

            for row in chain_data.get("strikes", []):
                strike = float(row.get("strikePrice", 0.0))
                for opt_type in ("CE", "PE"):
                    sub_row = row.get(opt_type)
                    if not sub_row:
                        continue

                    ltp = float(sub_row.get("lastPrice", 0.0))
                    bid = float(sub_row.get("buyPrice1") or sub_row.get("lastPrice") or 0.0)
                    ask = float(sub_row.get("sellPrice1") or sub_row.get("lastPrice") or 0.0)
                    iv = float(sub_row.get("impliedVolatility", 0.0))
                    oi = int(sub_row.get("openInterest", 0))

                    # 1. Standard format (e.g. NIFTY23NOV25950CE)
                    std_sym = self.format_contract_symbol(sym, exp, strike, opt_type)

                    tick_data = {
                        "symbol": std_sym,
                        "underlying": sym,
                        "strike": strike,
                        "option_type": opt_type,
                        "last_price": ltp,
                        "bid": bid,
                        "ask": ask,
                        "close": ltp,
                        "implied_volatility": iv / 100.0 if iv > 1.0 else iv,
                        "open_interest": oi,
                        "is_fresh": True,
                        "timestamp": chain_data["timestamp"],
                    }
                    tick_cache.update_tick(std_sym, tick_data)
                    updated_count += 1

                    # 2. Match any open positions with same (underlying, strike, option_type)
                    match_key = (sym.upper(), strike, opt_type.upper())
                    if match_key in open_pos_attr_map and ltp > 0:
                        for matched_pos in open_pos_attr_map[match_key]:
                            pos_sym = matched_pos["symbol"]
                            # Also update tick cache under the exact position symbol
                            custom_tick = tick_data.copy()
                            custom_tick["symbol"] = pos_sym
                            tick_cache.update_tick(pos_sym, custom_tick)

                            # Update MTM current price in database ledger
                            jnl.update_current_price(pos_sym, ltp)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._last_sync_stats = {
            "last_synced": now_str,
            "ticks_updated": updated_count,
            "spots": spots_synced,
            "status": "ACTIVE" if updated_count > 0 else "NO_DATA",
        }
        return self._last_sync_stats

    @property
    def sync_stats(self) -> dict[str, Any]:
        return self._last_sync_stats


_fetcher_instance: NSEOptionChainFetcher | None = None


def get_nse_delayed_fetcher() -> NSEOptionChainFetcher:
    """Singleton getter for NSEOptionChainFetcher."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = NSEOptionChainFetcher()
    return _fetcher_instance
