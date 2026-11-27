"""
StudyTrade — Official MCX India Option Chain Delayed Feed
==========================================================

Fetches official delayed option chain snapshots directly from mcxindia.com
(`/GetOptionChain`) with Chrome TLS impersonation for MCX commodities
(`CRUDEOILM`, `NATURALGAS`, etc.).

Normalizes quotes (underlying spot, LTP, top-of-book bid/ask, OI, volume) into the
same schema used by `NSEOptionChainFetcher` so `LiveTickCache`, `PaperJournal`,
live pricing, and the Gamma Squeeze Cockpit treat NSE and MCX uniformly.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from curl_cffi import requests

logger = logging.getLogger("studytrade.mcx_delayed")

IST = timezone(timedelta(hours=5, minutes=30))
MCX_SYMBOLS = frozenset({"CRUDEOILM", "NATURALGAS", "CRUDEOIL", "GOLD", "GOLDM", "SILVER", "SILVERM", "COPPER", "ZINC"})


class MCXOptionChainFetcher:
    """Fetches live/delayed MCX commodity option chains from mcxindia.com."""

    BASE_URL = "https://www.mcxindia.com"
    OPTION_CHAIN_PAGE = "https://www.mcxindia.com/market-data/option-chain"
    GET_CHAIN_API = "https://www.mcxindia.com/GetOptionChain"

    def __init__(self, cache_ttl_seconds: int = 14) -> None:
        self.cache_ttl_seconds = cache_ttl_seconds
        self._session: requests.Session | None = None
        self._symbol_expiries: dict[str, list[str]] = {}
        self._metadata_fetched_at: float = 0.0
        self._last_fetch_time: dict[str, float] = {}
        self._cached_results: dict[str, dict[str, Any]] = {}

    def _get_session(self, force_refresh: bool = False) -> requests.Session:
        if self._session is None or force_refresh:
            self._session = requests.Session(impersonate="chrome120")
            self._session.headers.update({
                "Referer": self.OPTION_CHAIN_PAGE,
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
            })
            self._refresh_symbol_metadata()
        elif (time.time() - self._metadata_fetched_at) > 3600 or not self._symbol_expiries:
            self._refresh_symbol_metadata()
        return self._session

    def _refresh_symbol_metadata(self) -> None:
        """Load the option chain page to establish session cookies and parse available expiries."""
        if self._session is None:
            return
        try:
            r = self._session.get(self.OPTION_CHAIN_PAGE, timeout=12)
            if r.status_code != 200:
                logger.warning("MCX option-chain page returned HTTP %s", r.status_code)
                return
            match = re.search(r'id=["\']symbol-data["\'][^>]*value=(["\'])(\[.*?\])\1', r.text, re.DOTALL)
            if not match:
                return
            raw_json = html.unescape(match.group(2))
            entries = json.loads(raw_json)
            exp_map: dict[str, list[str]] = {}
            for item in entries:
                sym = str(item.get("Symbol", "")).upper().strip()
                exp = str(item.get("ExpiryDate", "")).strip()
                if sym and exp:
                    lst = exp_map.setdefault(sym, [])
                    if exp not in lst:
                        lst.append(exp)
            if exp_map:
                self._symbol_expiries = exp_map
                self._metadata_fetched_at = time.time()
        except Exception as exc:
            logger.warning("Failed to refresh MCX session metadata: %s", exc)

    @staticmethod
    def _parse_mcx_date_ms(val: Any) -> datetime | None:
        if not val or not isinstance(val, str):
            return None
        m = re.search(r"/Date\((\d+)\)/", val)
        if not m:
            return None
        try:
            ts_sec = int(m.group(1)) / 1000.0
            return datetime.fromtimestamp(ts_sec, tz=IST)
        except Exception:
            return None

    @staticmethod
    def _normalize_expiry_to_mcx(expiry: str | None, available: list[str]) -> str | None:
        """Convert ISO/dashed expiry (e.g. '2026-12-16') to MCX format (e.g. '16DEC2026')."""
        if not available:
            return expiry.upper().replace("-", "") if expiry else None
        if not expiry:
            return available[0]

        cleaned = expiry.strip().upper()
        for candidate in available:
            if candidate.upper() == cleaned:
                return candidate

        # Try matching parsed date against available MCX expiry strings
        target_dt = None
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d%b%Y", "%d-%m-%Y"):
            try:
                target_dt = datetime.strptime(cleaned[:10] if "-" in cleaned and len(cleaned) >= 10 else cleaned, fmt).date()
                break
            except ValueError:
                continue

        if target_dt:
            for candidate in available:
                try:
                    cand_dt = datetime.strptime(candidate.upper(), "%d%b%Y").date()
                    if cand_dt == target_dt:
                        return candidate
                    # Also match same year + month if exact day differs slightly
                    if cand_dt.year == target_dt.year and cand_dt.month == target_dt.month:
                        return candidate
                except ValueError:
                    continue

        return available[0]

    @staticmethod
    def _expiry_to_iso(expiry_str: str) -> str:
        for fmt in ("%d%b%Y", "%d-%b-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(expiry_str.strip().upper(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return expiry_str

    @staticmethod
    def format_contract_symbol(underlying: str, expiry_str: str, strike: float, opt_type: str) -> str:
        """
        Format contract symbol matching StudyTrade's MCX monthly convention:
        e.g. 'CRUDEOILM26DEC7600PE'
        """
        dt = None
        for fmt in ("%d%b%Y", "%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                dt = datetime.strptime(expiry_str.strip().upper(), fmt)
                break
            except ValueError:
                continue
        if dt is None:
            dt = datetime.now(IST)

        strike_fmt = int(strike) if strike == int(strike) else strike
        yy = dt.strftime("%y")
        mon = dt.strftime("%b").upper()
        return f"{underlying.upper()}{yy}{mon}{strike_fmt}{opt_type.upper()}"

    def clear_cache(self) -> None:
        self._last_fetch_time.clear()

    def fetch_option_chain(
        self,
        symbol: str = "CRUDEOILM",
        expiry: str | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """
        Fetch live/delayed MCX option chain for a commodity symbol.
        Returns normalized dictionary compatible with `NSEOptionChainFetcher`.
        """
        sym = symbol.upper().strip()
        now = time.time()
        cache_key = f"{sym}|{expiry or ''}"
        if not force and cache_key in self._last_fetch_time and (now - self._last_fetch_time[cache_key]) < self.cache_ttl_seconds:
            return self._cached_results.get(cache_key)

        sess = self._get_session()
        avail_expiries = self._symbol_expiries.get(sym, [])
        target_expiry = self._normalize_expiry_to_mcx(expiry, avail_expiries)
        if not target_expiry:
            logger.warning("No MCX expiry resolved for %s", sym)
            return None

        params = {
            "InstrumentType": "OPTFUT",
            "Symbol": sym,
            "Expiry": target_expiry,
        }

        try:
            r = sess.get(self.GET_CHAIN_API, params=params, timeout=12)
            if r.status_code in (401, 403, 404):
                sess = self._get_session(force_refresh=True)
                r = sess.get(self.GET_CHAIN_API, params=params, timeout=12)

            if r.status_code != 200:
                logger.warning("MCX GetOptionChain returned HTTP %s for %s", r.status_code, sym)
                return self._cached_results.get(cache_key)

            payload = r.json()
            raw_rows = payload.get("Data") or []
            if not raw_rows:
                logger.warning("MCX GetOptionChain returned empty rows for %s (%s)", sym, target_expiry)
                return self._cached_results.get(cache_key)

            summary = payload.get("Summary") or {}
            as_on_dt = self._parse_mcx_date_ms(summary.get("AsOn")) or datetime.now(IST)
            ts_str = as_on_dt.strftime("%Y-%m-%d %H:%M:%S")

            spot_price = 0.0
            normalized_strikes: list[dict[str, Any]] = []

            for row in raw_rows:
                uval = float(row.get("UnderlyingValue") or 0.0)
                if uval > 0 and spot_price <= 0:
                    spot_price = uval

                strike = float(row.get("CE_StrikePrice") or row.get("PE_StrikePrice") or 0.0)
                if strike <= 0:
                    continue

                def _build_leg(prefix: str) -> dict[str, Any]:
                    raw_ltp = float(row.get(f"{prefix}_LTP") or 0.0)
                    bid = float(row.get(f"{prefix}_BidPrice") or 0.0)
                    ask = float(row.get(f"{prefix}_AskPrice") or 0.0)
                    # Deep ITM/OTM MCX contracts may have 0 LTP if untraded today; fall back to mid/bid
                    eff_ltp = raw_ltp
                    if eff_ltp <= 0:
                        if bid > 0 and ask > 0:
                            eff_ltp = round((bid + ask) / 2.0, 2)
                        elif bid > 0:
                            eff_ltp = bid
                        elif ask > 0:
                            eff_ltp = ask

                    return {
                        "lastPrice": eff_ltp,
                        "rawLastPrice": raw_ltp,
                        "buyPrice1": bid if bid > 0 else eff_ltp,
                        "sellPrice1": ask if ask > 0 else eff_ltp,
                        "impliedVolatility": 0.0,
                        "openInterest": int(row.get(f"{prefix}_OpenInterest") or 0),
                        "changeinOpenInterest": int(row.get(f"{prefix}_ChangeInOI") or 0),
                        "totalTradedVolume": int(row.get(f"{prefix}_Volume") or 0),
                    }

                normalized_strikes.append({
                    "strikePrice": strike,
                    "CE": _build_leg("CE"),
                    "PE": _build_leg("PE"),
                })

            result = {
                "symbol": sym,
                "spot_price": spot_price,
                "expiry": target_expiry,
                "expiry_iso": self._expiry_to_iso(target_expiry),
                "available_expiries": avail_expiries,
                "timestamp": ts_str,
                "strikes": normalized_strikes,
            }
            self._last_fetch_time[cache_key] = now
            self._cached_results[cache_key] = result
            return result
        except Exception as exc:
            logger.error("Error fetching MCX option chain for %s: %s", sym, exc)
            return self._cached_results.get(cache_key)


_mcx_fetcher_instance: MCXOptionChainFetcher | None = None


def get_mcx_delayed_fetcher() -> MCXOptionChainFetcher:
    """Singleton getter for MCXOptionChainFetcher."""
    global _mcx_fetcher_instance
    if _mcx_fetcher_instance is None:
        _mcx_fetcher_instance = MCXOptionChainFetcher()
    return _mcx_fetcher_instance


def overlay_live_mcx_on_chain(
    symbol: str,
    spot_close: float,
    opt_slice: Any,
) -> tuple[float, Any]:
    """
    Overlay live/delayed MCX option chain quotes onto a DataStore option slice
    for today's active expiry and warm LiveTickCache so signal recommendations
    reflect real live MCX market prices instead of static parquet closes.
    """
    sym = symbol.upper().strip()
    if sym not in MCX_SYMBOLS or opt_slice is None or opt_slice.empty:
        return spot_close, opt_slice

    fetcher = get_mcx_delayed_fetcher()
    chain = fetcher.fetch_option_chain(sym)
    if not chain or float(chain.get("spot_price", 0.0)) <= 0:
        return spot_close, opt_slice

    live_spot = float(chain["spot_price"])
    exp_str = chain.get("expiry", "")
    ts_str = chain.get("timestamp") or datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    try:
        from studytrade.live.session_engine import get_live_tick_cache
        tick_cache = get_live_tick_cache()
    except Exception:
        tick_cache = None

    quote_lookup: dict[tuple[float, str], dict[str, Any]] = {}
    for row in chain.get("strikes", []):
        strike = round(float(row.get("strikePrice", 0.0)), 2)
        for ot in ("CE", "PE"):
            leg = row.get(ot)
            if not leg:
                continue
            ltp = float(leg.get("lastPrice", 0.0))
            if ltp <= 0:
                continue
            quote_lookup[(strike, ot)] = leg
            if tick_cache is not None:
                std_sym = fetcher.format_contract_symbol(sym, exp_str, strike, ot)
                tick_cache.update_tick(
                    std_sym,
                    {
                        "symbol": std_sym,
                        "underlying": sym,
                        "strike": strike,
                        "option_type": ot,
                        "last_price": ltp,
                        "bid": float(leg.get("buyPrice1") or ltp),
                        "ask": float(leg.get("sellPrice1") or ltp),
                        "close": ltp,
                        "open_interest": int(leg.get("openInterest") or 0),
                        "volume": int(leg.get("totalTradedVolume") or 0),
                        "is_fresh": True,
                        "timestamp": ts_str,
                    },
                )

    if tick_cache is not None:
        tick_cache.update_tick(
            sym,
            {
                "symbol": sym,
                "last_price": live_spot,
                "is_fresh": True,
                "timestamp": ts_str,
            },
        )

    if not quote_lookup:
        return live_spot, opt_slice

    df = opt_slice.copy()
    for idx, r in df.iterrows():
        key = (round(float(r.get("strike", 0.0)), 2), str(r.get("option_type", "")).upper())
        leg = quote_lookup.get(key)
        if not leg:
            continue
        ltp = float(leg["lastPrice"])
        df.at[idx, "close"] = ltp
        if "bid" in df.columns:
            df.at[idx, "bid"] = float(leg.get("buyPrice1") or ltp)
        if "ask" in df.columns:
            df.at[idx, "ask"] = float(leg.get("sellPrice1") or ltp)
        if "open_interest" in df.columns and int(leg.get("openInterest") or 0) > 0:
            df.at[idx, "open_interest"] = int(leg["openInterest"])
        if "volume" in df.columns and int(leg.get("totalTradedVolume") or 0) > 0:
            df.at[idx, "volume"] = int(leg["totalTradedVolume"])

    return live_spot, df
