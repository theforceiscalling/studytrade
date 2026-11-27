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

from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import time
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any

from curl_cffi import requests

from studytrade.live.session_engine import get_live_tick_cache
from studytrade.paper.journal import PaperJournal

logger = logging.getLogger("studytrade.nse_delayed")
IST = timezone(timedelta(hours=5, minutes=30))


def _is_nse_market_open_ist() -> bool:
    """Return True if NSE F&O market is currently in live session (Mon-Fri 09:15-15:35 IST)."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(9, 10) <= t <= dtime(15, 35)


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

    def __init__(self, cache_ttl_seconds: int = 10) -> None:
        self.cache_ttl_seconds = cache_ttl_seconds
        self._off_hours_ttl_seconds = 900
        self._lock = threading.Lock()
        self._session: requests.Session | None = None
        self._last_fetch_time: dict[str, float] = {}
        self._cached_results: dict[str, dict[str, Any]] = {}
        self._last_sync_stats: dict[str, Any] = {
            "last_synced": None,
            "ticks_updated": 0,
            "status": "IDLE",
        }

    def _get_session(self) -> requests.Session:
        with self._lock:
            if self._session is None:
                self._session = requests.Session(impersonate="chrome120")
                self._session.headers.update({
                    "Referer": self.CHAIN_PAGE,
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "application/json, text/plain, */*",
                })
                # Warm up session with option chain page to establish Akamai cookies
                try:
                    self._session.get(self.CHAIN_PAGE, timeout=5)
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
        Uses tight TTL during live NSE hours and longer TTL after hours so
        MCX/portal syncs remain sub-second.
        """
        now = time.time()
        cache_key = f"{symbol}|{expiry or ''}"
        eff_ttl = self.cache_ttl_seconds if _is_nse_market_open_ist() else self._off_hours_ttl_seconds
        if cache_key in self._last_fetch_time and (now - self._last_fetch_time[cache_key]) < eff_ttl:
            return self._cached_results.get(cache_key)

        sess = self._get_session()
        try:
            target_expiry = expiry
            if not target_expiry:
                info_url = self.CONTRACT_INFO_API.format(symbol=symbol)
                r_info = sess.get(info_url, timeout=5)
                if r_info.status_code == 200:
                    exp_dates = r_info.json().get("expiryDates", [])
                    if exp_dates:
                        target_expiry = exp_dates[0]
                elif r_info.status_code in (401, 403, 404):
                    # Session expired or blocked, re-initialize
                    with self._lock:
                        self._session = None
                    sess = self._get_session()
                    r_info = sess.get(info_url, timeout=5)
                    if r_info.status_code == 200:
                        exp_dates = r_info.json().get("expiryDates", [])
                        if exp_dates:
                            target_expiry = exp_dates[0]

            if not target_expiry:
                logger.warning(f"No expiry date resolved for {symbol}")
                self._last_fetch_time[cache_key] = now
                return self._cached_results.get(cache_key)

            # Fetch Option Chain v3
            chain_url = self.CHAIN_V3_API.format(
                chain_type=self.chain_type(symbol), symbol=symbol, expiry=target_expiry
            )
            r_chain = sess.get(chain_url, timeout=6)
            if r_chain.status_code != 200:
                logger.warning(f"NSE chain API returned status {r_chain.status_code} for {symbol}")
                self._last_fetch_time[cache_key] = now
                return self._cached_results.get(cache_key)

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

    def _sync_chain_to_cache(
        self,
        sym: str,
        chain_data: dict[str, Any],
        symbol_formatter: Any,
        tick_cache: Any,
        jnl: PaperJournal,
        open_pos_attr_map: dict[tuple[str, float, str], list[dict[str, Any]]],
        spots_synced: dict[str, float],
        updated_pos_symbols: set[str],
    ) -> int:
        """Process a normalized option chain snapshot into LiveTickCache and PaperJournal."""
        spot = float(chain_data.get("spot_price") or 0.0)
        exp = str(chain_data.get("expiry") or "")
        exp_iso = str(chain_data.get("expiry_iso") or "")
        ts = chain_data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_count = 0

        if spot > 0:
            spots_synced[sym] = spot
            tick_cache.update_tick(
                sym,
                {
                    "symbol": sym,
                    "last_price": spot,
                    "is_fresh": True,
                    "timestamp": ts,
                },
            )

        for row in chain_data.get("strikes", []):
            strike = float(row.get("strikePrice", 0.0))
            for opt_type in ("CE", "PE"):
                sub_row = row.get(opt_type)
                if not sub_row:
                    continue

                ltp = float(sub_row.get("lastPrice", 0.0))
                bid = float(sub_row.get("buyPrice1") or ltp or 0.0)
                ask = float(sub_row.get("sellPrice1") or ltp or 0.0)
                iv = float(sub_row.get("impliedVolatility", 0.0))
                oi = int(sub_row.get("openInterest", 0))
                if ltp <= 0 and bid <= 0 and ask <= 0:
                    continue

                std_sym = symbol_formatter(sym, exp, strike, opt_type)
                attr_key = f"{sym.upper()}|{float(strike)}|{opt_type.upper()}"
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
                    "timestamp": ts,
                }
                tick_cache.update_tick(std_sym, tick_data)
                tick_cache.update_tick(attr_key, tick_data)
                updated_count += 1

                # Match open positions by (underlying, strike, option_type) and expiry if specified
                match_key = (sym.upper(), strike, opt_type.upper())
                if match_key in open_pos_attr_map and ltp > 0:
                    for matched_pos in open_pos_attr_map[match_key]:
                        pos_exp = str(matched_pos.get("expiry", ""))[:10]
                        if exp_iso and pos_exp and pos_exp != exp_iso and pos_exp[:7] != exp_iso[:7]:
                            continue
                        pos_sym = matched_pos["symbol"]
                        custom_tick = tick_data.copy()
                        custom_tick["symbol"] = pos_sym
                        tick_cache.update_tick(pos_sym, custom_tick)
                        jnl.update_current_price(pos_sym, ltp)
                        updated_pos_symbols.add(pos_sym)

        return updated_count

    def sync_to_tick_cache(
        self,
        symbols: list[str] = ["NIFTY", "BANKNIFTY", "FINNIFTY"],
        journal: PaperJournal | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Pull latest delayed quotes from NSE & MCX and update LiveTickCache & PaperJournal.
        Matches both exact contract symbols and open position attributes (underlying, strike, option_type).
        """
        from studytrade.data.fetchers.mcx_delayed import MCX_SYMBOLS, get_mcx_delayed_fetcher

        mcx_fetcher = get_mcx_delayed_fetcher()
        if force:
            self._last_fetch_time.clear()
            mcx_fetcher.clear_cache()

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
        updated_pos_symbols: set[str] = set()

        # 1. Fetch NSE Equity Indices and MCX Commodities in parallel
        nse_syms = [s for s in symbols if s.upper() not in MCX_SYMBOLS]
        mcx_targets: set[tuple[str, str | None]] = {("CRUDEOILM", None), ("NATURALGAS", None)}
        for p in open_positions:
            und = str(p.get("underlying_symbol", "")).upper()
            if und in MCX_SYMBOLS:
                exp_str = str(p.get("expiry", ""))[:10] or None
                mcx_targets.add((und, exp_str))
        mcx_list = sorted(mcx_targets, key=lambda x: (x[0], x[1] or ""))

        nse_results: list[tuple[str, dict[str, Any] | None]] = []
        mcx_results: list[tuple[str, dict[str, Any] | None]] = []

        with ThreadPoolExecutor(max_workers=5) as pool:
            nse_futs = {pool.submit(self.fetch_index_chain, sym): sym for sym in nse_syms}
            mcx_futs = {pool.submit(mcx_fetcher.fetch_option_chain, sym, exp, force): sym for sym, exp in mcx_list}
            for fut, sym in nse_futs.items():
                try:
                    nse_results.append((sym, fut.result()))
                except Exception as exc:
                    logger.warning("NSE parallel fetch error for %s: %s", sym, exc)
            for fut, sym in mcx_futs.items():
                try:
                    mcx_results.append((sym, fut.result()))
                except Exception as exc:
                    logger.warning("MCX parallel fetch error for %s: %s", sym, exc)

        for sym, chain_data in nse_results:
            if chain_data:
                updated_count += self._sync_chain_to_cache(
                    sym, chain_data, self.format_contract_symbol, tick_cache, jnl,
                    open_pos_attr_map, spots_synced, updated_pos_symbols,
                )

        fetched_mcx_keys: set[tuple[str, str]] = set()
        for sym, chain_data in mcx_results:
            if chain_data:
                resolved_exp = str(chain_data.get("expiry", ""))
                if (sym, resolved_exp) in fetched_mcx_keys:
                    continue
                fetched_mcx_keys.add((sym, resolved_exp))
                updated_count += self._sync_chain_to_cache(
                    sym, chain_data, mcx_fetcher.format_contract_symbol, tick_cache, jnl,
                    open_pos_attr_map, spots_synced, updated_pos_symbols,
                )

        # 3. Fallback BSM MTM only for open MCX positions that were not returned by live MCX API
        missing_mcx = [
            p for p in open_positions
            if str(p.get("underlying_symbol", "")).upper() in MCX_SYMBOLS
            and p.get("symbol") not in updated_pos_symbols
        ]
        if missing_mcx:
            try:
                import pandas as pd
                from studytrade.constants import OptionType
                from studytrade.data.store import DataStore
                from studytrade.math.bsm import bsm_price

                store = DataStore()
                for p in missing_mcx:
                    und = str(p.get("underlying_symbol", "")).upper()
                    pos_sym = str(p.get("symbol", ""))
                    stk = float(p.get("strike", 0.0))
                    ot = str(p.get("option_type", "CE")).upper()
                    exp_str = str(p.get("expiry", ""))

                    spot_df = store.load_spot(und)
                    cur_spot = float(spot_df["close"].iloc[-1]) if not spot_df.empty else (7735.0 if und == "CRUDEOILM" else 334.5)
                    spots_synced.setdefault(und, cur_spot)
                    iv_val = 0.50 if und == "CRUDEOILM" else 0.55

                    try:
                        dtes = max(1.0, float((pd.to_datetime(exp_str).date() - datetime.now().date()).days)) if exp_str else 19.0
                    except Exception:
                        dtes = 19.0
                    theo = bsm_price(OptionType.CALL if ot == "CE" else OptionType.PUT, cur_spot, stk, dtes / 365.0, 0.07, iv_val)
                    fallback_ltp = round(float(theo), 2)

                    tick_cache.update_tick(
                        pos_sym,
                        {
                            "symbol": pos_sym,
                            "underlying": und,
                            "strike": stk,
                            "option_type": ot,
                            "last_price": fallback_ltp,
                            "bid": round(fallback_ltp * 0.99, 2),
                            "ask": round(fallback_ltp * 1.01, 2),
                            "close": fallback_ltp,
                            "implied_volatility": iv_val,
                            "is_fresh": False,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    jnl.update_current_price(pos_sym, fallback_ltp)
                    updated_count += 1
            except Exception as e:
                logger.warning("Error syncing fallback commodity open positions: %s", e)

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
