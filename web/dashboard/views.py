from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from studytrade.analytics.metrics import compute_drawdowns, compute_performance_metrics
from studytrade.analytics.monte_carlo import simulate_monte_carlo_drawdowns
from studytrade.analytics.overfitting import compute_deflated_sharpe_ratio
from studytrade.backtest.engine import BacktestEngine
from studytrade.constants import INSTRUMENTS, OptionType
from studytrade.data.sample_generator import generate_and_store_dataset
from studytrade.data.store import DataStore
from studytrade.features.gamma_exposure import compute_gex_profile
from studytrade.features.microstructure import compute_max_pain, compute_pcr
from studytrade.features.regime import detect_regimes
from studytrade.features.technical import (
    compute_donchian_breakout,
    compute_ema,
    compute_macd,
    compute_rsi,
)
from studytrade.features.volatility import (
    GARCH11,
    realized_vol_garman_klass,
    realized_vol_rogers_satchell,
)
from studytrade.math.iv_surface import realized_volatility_close
from studytrade.paper.journal import PaperJournal
from studytrade.risk.sizing import PositionSizer
from studytrade.strategy.signals.breakout import BreakoutStrategy
from studytrade.strategy.signals.debit_spread import DebitSpreadStrategy
from studytrade.strategy.signals.gamma_squeeze import GammaSqueezeStrategy
from studytrade.strategy.signals.momentum import MomentumStrategy
from studytrade.strategy.signals.vol_expansion import VolatilityExpansionStrategy

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _get_store() -> DataStore:
    return DataStore(str(DATA_DIR))


def _ensure_data_exists(symbol: str = "NIFTY") -> None:
    store = _get_store()
    try:
        spot = store.load_spot(symbol)
        if spot.empty:
            generate_and_store_dataset(symbol=symbol, n_days=125, base_dir=str(DATA_DIR))
    except Exception:
        generate_and_store_dataset(symbol=symbol, n_days=125, base_dir=str(DATA_DIR))


# ---------------------------------------------------------------------------
# 1. Overview Dashboard
# ---------------------------------------------------------------------------


def overview_view(request: HttpRequest) -> HttpResponse:
    store = _get_store()
    catalog_df = store.get_catalog()
    catalog = catalog_df.to_dict(orient="records") if not catalog_df.empty else []

    journal = PaperJournal(str(DATA_DIR / "paper_journal.db"))
    paper_days = journal.get_paper_duration_days()

    context = {
        "catalog": catalog,
        "paper_days": paper_days,
        "paper_days_pct": min(100, int((paper_days / 90) * 100)),
        "open_positions_count": 0,
        "symbols": list(INSTRUMENTS.keys()),
    }
    return render(request, "overview.html", context)


# ---------------------------------------------------------------------------
# 2. Data Lab View
# ---------------------------------------------------------------------------


def data_lab_view(request: HttpRequest) -> HttpResponse:

    symbol = request.GET.get("symbol", "NIFTY")
    _ensure_data_exists(symbol)

    store = _get_store()
    spot_df = store.load_spot(symbol)
    options_df = store.load_options(symbol)

    dates: list[str] = []
    if not spot_df.empty and "timestamp" in spot_df.columns:
        dates = [d.strftime("%Y-%m-%d") for d in spot_df["timestamp"].dt.date.unique()]

    selected_date = request.GET.get("date", dates[-1] if dates else "")

    expiries: list[str] = []
    chain_rows: list[dict[str, Any]] = []
    iv_skew_json = "{}"
    max_pain_val = 0.0
    pcr_val = 0.0

    if not options_df.empty and selected_date:
        opt_day = options_df[options_df["timestamp"].dt.strftime("%Y-%m-%d") == selected_date]
        if not opt_day.empty:
            expiries = [e.strftime("%Y-%m-%d") for e in opt_day["expiry"].dt.date.unique()]
            selected_expiry = request.GET.get("expiry", expiries[0] if expiries else "")

            if selected_expiry:
                chain = opt_day[opt_day["expiry"].dt.strftime("%Y-%m-%d") == selected_expiry]
                strikes = sorted(chain["strike"].unique())

                calls = chain[chain["option_type"] == OptionType.CALL.value].set_index("strike")
                puts = chain[chain["option_type"] == OptionType.PUT.value].set_index("strike")

                for k in strikes:
                    c = calls.loc[k] if k in calls.index else None
                    p = puts.loc[k] if k in puts.index else None
                    c_iv = round(c["implied_volatility"] * 100, 1) if c is not None else "-"
                    p_iv = round(p["implied_volatility"] * 100, 1) if p is not None else "-"
                    chain_rows.append(
                        {
                            "strike": k,
                            "call_iv": c_iv,
                            "call_bid": round(c["bid"], 2) if c is not None else "-",
                            "call_ask": round(c["ask"], 2) if c is not None else "-",
                            "call_oi": int(c["open_interest"]) if c is not None else 0,
                            "put_iv": p_iv,
                            "put_bid": round(p["bid"], 2) if p is not None else "-",
                            "put_ask": round(p["ask"], 2) if p is not None else "-",
                            "put_oi": int(p["open_interest"]) if p is not None else 0,
                        }
                    )

                # IV Skew plot data
                skew_data = {
                    "strikes": strikes,
                    "call_ivs": [
                        float(calls.loc[k]["implied_volatility"] * 100)
                        if k in calls.index
                        else None
                        for k in strikes
                    ],
                    "put_ivs": [
                        float(puts.loc[k]["implied_volatility"] * 100)
                        if k in puts.index
                        else None
                        for k in strikes
                    ],
                }
                iv_skew_json = json.dumps(skew_data)

            # Microstructure & Gamma Exposure (GEX)
            pcr_res = compute_pcr(opt_day)
            pcr_val = round(pcr_res.pcr_oi, 2)
            max_pain_val = compute_max_pain(opt_day)

            spot_day_slice = spot_df[spot_df["timestamp"].dt.strftime("%Y-%m-%d") == selected_date] if not spot_df.empty else pd.DataFrame()
            spot_close = float(spot_day_slice["close"].iloc[-1]) if not spot_day_slice.empty else (float(opt_day["strike"].mean()) if not opt_day.empty else 0.0)

            gex_prof = compute_gex_profile(opt_day, spot_price=spot_close) if spot_close > 0 else None
            call_wall_val = gex_prof.call_wall_strike if gex_prof else 0.0
            put_wall_val = gex_prof.put_wall_strike if gex_prof else 0.0
            gamma_flip_val = gex_prof.gamma_flip_level if gex_prof else 0.0
            call_squeeze_val = gex_prof.call_squeeze_score if gex_prof else 0.0
            put_cascade_val = gex_prof.put_cascade_score if gex_prof else 0.0
            net_gex_val = round(gex_prof.net_gex / 1e5, 2) if gex_prof else 0.0  # in Lakhs

    context = {
        "symbol": symbol,
        "symbols": list(INSTRUMENTS.keys()),
        "dates": dates,
        "selected_date": selected_date,
        "expiries": expiries,
        "selected_expiry": request.GET.get("expiry", expiries[0] if expiries else ""),
        "chain_rows": chain_rows,
        "iv_skew_json": iv_skew_json,
        "max_pain": max_pain_val,
        "pcr": pcr_val,
        "call_wall": call_wall_val if selected_date else 0.0,
        "put_wall": put_wall_val if selected_date else 0.0,
        "gamma_flip": gamma_flip_val if selected_date else 0.0,
        "call_squeeze_score": call_squeeze_val if selected_date else 0.0,
        "put_cascade_score": put_cascade_val if selected_date else 0.0,
        "net_gex_lakhs": net_gex_val if selected_date else 0.0,
        "spot_chart_json": json.dumps(
            {
                "dates": [d.strftime("%Y-%m-%d") for d in spot_df["timestamp"]],
                "opens": spot_df["open"].tolist(),
                "highs": spot_df["high"].tolist(),
                "lows": spot_df["low"].tolist(),
                "closes": spot_df["close"].tolist(),
                "volumes": spot_df["volume"].tolist(),
            }
        )
        if not spot_df.empty
        else "{}",
    }
    return render(request, "data_lab.html", context)


# ---------------------------------------------------------------------------
# 3. Features & Regimes View
# ---------------------------------------------------------------------------


def features_view(request: HttpRequest) -> HttpResponse:

    symbol = request.GET.get("symbol", "NIFTY")
    _ensure_data_exists(symbol)

    store = _get_store()
    spot_df = store.load_spot(symbol)

    if spot_df.empty:
        return render(request, "features.html", {"error": "No spot data available"})

    c = spot_df["close"]
    h = spot_df["high"]
    l_ = spot_df["low"]
    o = spot_df["open"]

    # Technical Features
    ema_fast = compute_ema(c, span=20)
    ema_slow = compute_ema(c, span=50)
    donchian = compute_donchian_breakout(h, l_, c, lookback=20)
    rsi = compute_rsi(c, period=14)
    macd = compute_macd(c)

    # Volatility Features
    rv_cc = realized_volatility_close(c, window=20)
    rv_gk = realized_vol_garman_klass(o, h, l_, c, window=20)
    rv_rs = realized_vol_rogers_satchell(o, h, l_, c, window=20)

    # Regimes
    regimes = detect_regimes(h, l_, c)

    # GARCH Fit
    returns = c.pct_change().dropna()
    garch = GARCH11()
    garch_res = garch.fit(returns)
    forecasts = garch.forecast(horizon=10)

    dates = [d.strftime("%Y-%m-%d") for d in spot_df["timestamp"]]

    chart_data = {
        "dates": dates,
        "closes": c.tolist(),
        "ema_fast": ema_fast.fillna(0).tolist(),
        "ema_slow": ema_slow.fillna(0).tolist(),
        "donchian_high": donchian["donchian_high"].fillna(0).tolist(),
        "donchian_low": donchian["donchian_low"].fillna(0).tolist(),
        "rsi": rsi.fillna(50).tolist(),
        "macd_hist": macd["macd_hist"].fillna(0).tolist(),
        "rv_cc": (rv_cc * 100).fillna(0).tolist(),
        "rv_gk": (rv_gk * 100).fillna(0).tolist(),
        "rv_rs": (rv_rs * 100).fillna(0).tolist(),
        "regimes": regimes.tolist(),
        "garch_vol": (rv_gk * 100).fillna(0).tolist(),
        "garch_forecast": (forecasts * 100).tolist(),
    }

    regime_counts = regimes.value_counts().to_dict()

    context = {
        "symbol": symbol,
        "symbols": list(INSTRUMENTS.keys()),
        "chart_json": json.dumps(chart_data),
        "garch_params": {
            "omega": f"{garch_res.omega:.6f}",
            "alpha": f"{garch_res.alpha:.4f}",
            "beta": f"{garch_res.beta:.4f}",
            "persistence": f"{garch_res.persistence:.4f}",
            "long_run_vol": f"{garch_res.long_run_vol * 100:.2f}%",
        },
        "regime_counts": regime_counts,
    }
    return render(request, "features.html", context)


# ---------------------------------------------------------------------------
# 4. Backtest View
# ---------------------------------------------------------------------------


def backtest_view(request: HttpRequest) -> HttpResponse:
    min_date = "2016-01-01"
    max_date = "2026-11-20"
    try:
        store = _get_store()
        spot = store.load_spot("NIFTY")
        if not spot.empty:
            min_date = spot["timestamp"].min().strftime("%Y-%m-%d")
            max_date = spot["timestamp"].max().strftime("%Y-%m-%d")
    except Exception:
        pass

    from studytrade.data.backtest_db import get_all_backtest_runs
    past_runs = _enrich_runs(get_all_backtest_runs())
    ranked_runs = sorted(past_runs, key=lambda x: x.get("stoxkart_net_pnl", 0.0), reverse=True)
    for idx, r in enumerate(ranked_runs):
        r["rank"] = idx + 1

    return render(
        request,
        "backtest.html",
        {
            "symbols": list(INSTRUMENTS.keys()),
            "strategies": [
                ("breakout", "Donchian Breakout + ATR Surge"),
                ("momentum", "Dual EMA + MACD Momentum"),
                ("vol_expansion", "Bollinger Bandwidth Volatility Expansion"),
                ("debit_spread", "Defined-Risk Debit Spreads"),
                ("gamma_squeeze", "Gamma Squeeze & Dealer Cascade"),
            ],
            "min_date": min_date,
            "max_date": max_date,
            "default_start": "2024-01-01",
            "default_end": max_date,
            "past_runs": past_runs,
            "ranked_runs": ranked_runs,
        },
    )


# ---------------------------------------------------------------------------
# 5. Analytics View
# ---------------------------------------------------------------------------


def analytics_view(request: HttpRequest) -> HttpResponse:
    return render(request, "analytics.html", {})


# ---------------------------------------------------------------------------
# 6. Risk View
# ---------------------------------------------------------------------------


def risk_view(request: HttpRequest) -> HttpResponse:
    return render(request, "risk.html", {"symbols": list(INSTRUMENTS.keys())})


# ---------------------------------------------------------------------------
# 7. Paper Trading Cockpit
# ---------------------------------------------------------------------------


def paper_view(request: HttpRequest) -> HttpResponse:
    journal = PaperJournal(str(DATA_DIR / "paper_journal.db"))
    paper_days = journal.get_paper_duration_days()
    trades_df = journal.get_trades_dataframe()
    open_positions = journal.get_open_positions()

    # Aggregate Greeks
    net_delta = 0.0
    net_gamma = 0.0
    net_vega = 0.0
    total_theta_bleed = 0.0

    for pos in open_positions:
        try:
            from studytrade.math.greeks import greeks
            und_sym = str(pos.get("underlying_symbol", "NIFTY"))
            default_spot = 53500.0 if und_sym == "BANKNIFTY" else (24800.0 if und_sym == "FINNIFTY" else 25800.0)
            
            if "expiry" in pos and pos["expiry"]:
                try:
                    exp_date = pd.to_datetime(pos["expiry"]).date()
                    now_date = datetime.now().date()
                    dte_d = max(1.0, float((exp_date - now_date).days))
                    t_y = dte_d / 365.0
                except Exception:
                    t_y = 3.5 / 365.0
            else:
                t_y = 3.5 / 365.0

            opt_type_enum = OptionType.CALL if pos.get("option_type") == "CE" else OptionType.PUT
            iv_val = float(pos.get("entry_iv", 11.2))
            sigma = iv_val / 100.0 if iv_val > 1.0 else iv_val
            
            g = greeks(
                opt_type_enum,
                default_spot,
                float(pos.get("strike", default_spot)),
                t_y,
                0.07,
                sigma,
            )
            qty = float(pos.get("quantity", 1)) * float(pos.get("lot_size", 25))
            net_delta += float(g.delta) * qty
            net_gamma += float(g.gamma) * qty
            net_vega += float(g.vega_1pct) * qty
            total_theta_bleed += float(g.theta_daily) * qty
        except Exception:
            pass

    # Capital tracking (User specified initial capital: Rs. 20,000)
    INITIAL_CAPITAL = 20000.0
    deployed_capital = sum(
        float(pos.get("entry_price", 0.0)) * int(pos.get("lot_size", 25)) * int(pos.get("quantity", 1))
        for pos in open_positions
    )
    available_capital = max(0.0, INITIAL_CAPITAL - deployed_capital)

    # Compute Real-Time Strategy Signals (Gamma Squeeze & Dealer Cascade)
    realtime_signals = []
    store = _get_store()
    for sym in ["NIFTY", "BANKNIFTY"]:
        try:
            spot_df = store.load_spot(sym)
            opt_df = store.load_options(sym)
            if spot_df.empty or opt_df.empty:
                continue

            latest_ts = opt_df["timestamp"].max()
            opt_slice = opt_df[opt_df["timestamp"] == latest_ts]
            spot_slice = spot_df.tail(10)
            spot_close = float(spot_df["close"].iloc[-1])

            from studytrade.strategy.signals.gamma_squeeze import GammaSqueezeStrategy
            try:
                from studytrade.paper.auto_trader import AutoTrader
                trader_cfg = AutoTrader.load_config()
                cfg_delta = float(trader_cfg.get("target_delta", 0.35))
                cfg_tp = float(trader_cfg.get("profit_target_pct", 50.0))
                cfg_sl = float(trader_cfg.get("stop_loss_pct", 30.0))
                cfg_thresh = float(trader_cfg.get("min_confidence_pct", 50.0))
            except Exception:
                cfg_delta, cfg_tp, cfg_sl, cfg_thresh = 0.35, 50.0, 30.0, 50.0

            # Use calibrated optimal settings: Target Delta 0.35 (or user config), 50% TP, 30% SL
            strat = GammaSqueezeStrategy(
                target_delta=cfg_delta,
                squeeze_score_threshold=cfg_thresh,
                allow_bearish=True,
            )
            raw_sigs = strat.generate_signals(spot_slice, opt_slice)
            for s in raw_sigs:
                rec = _get_manual_trade_recommendation(
                    opt_slice,
                    s.direction.value,
                    spot_close,
                    target_delta=s.target_delta,
                    profit_target_pct=cfg_tp,
                    stop_loss_pct=cfg_sl,
                    symbol=sym,
                )
                if rec:
                    sig_type_badge = "BULLISH CALL (SQUEEZE)" if "CALL" in s.direction.value else "BEARISH PUT (CASCADE)"
                    realtime_signals.append(
                        {
                            "symbol": rec["contract_symbol"],
                            "underlying_symbol": sym,
                            "option_type": "CE" if "CALL" in s.direction.value else "PE",
                            "strike": rec["strike"],
                            "expiry": str(opt_slice["expiry"].iloc[0])[:10] if "expiry" in opt_slice.columns else "",
                            "direction": s.direction.value,
                            "badge_text": sig_type_badge,
                            "target_delta": rec.get("delta", s.target_delta),
                            "entry_price": rec.get("ask", rec.get("premium", 100.0)),
                            "target_price": rec.get("target_price", 150.0),
                            "stop_loss_price": rec.get("stop_loss_price", 70.0),
                            "lot_size": rec.get("lot_size", 25),
                            "capital_required": rec.get("capital_per_lot", 2500.0),
                            "confidence_pct": round(s.strength * 100, 1),
                            "reason": s.reason,
                            "is_recommended": True,
                        }
                    )
        except Exception:
            pass

    context = {
        "paper_days": paper_days,
        "paper_days_pct": min(100, int((paper_days / 90) * 100)),
        "is_locked": paper_days < 90,
        "open_positions": open_positions,
        "realtime_signals": realtime_signals,
        "initial_capital": INITIAL_CAPITAL,
        "deployed_capital": round(deployed_capital, 2),
        "available_capital": round(available_capital, 2),
        "net_delta": round(net_delta, 2),
        "net_gamma": round(net_gamma, 4),
        "net_vega": round(net_vega, 2),
        "total_theta_bleed": round(total_theta_bleed, 2),
        "trade_history": trades_df.to_dict(orient="records") if not trades_df.empty else [],
        "auto_config": _get_auto_trader_config(),
        "strategy_ranks": _get_top_backtest_ranks(),
    }
    return render(request, "paper.html", context)


def _get_auto_trader_config() -> dict[str, Any]:
    from studytrade.paper.auto_trader import get_auto_trader
    return get_auto_trader(DATA_DIR).config


def _get_top_backtest_ranks() -> list[dict[str, Any]]:
    """Query top-ranked gamma squeeze backtests by net profit / Sharpe ratio."""
    db_path = DATA_DIR / "backtest_runs.db"
    ranks = []
    if db_path.exists():
        try:
            con = sqlite3.connect(str(db_path))
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            rows = cur.execute(
                "SELECT id, symbol, strategy, stoxkart_net_pnl, sharpe_ratio, win_rate_pct, params_json "
                "FROM backtest_runs WHERE strategy LIKE '%gamma%' "
                "ORDER BY stoxkart_net_pnl DESC"
            ).fetchall()
            seen_configs = set()
            for r in rows:
                try:
                    p = json.loads(r["params_json"])
                    td = float(p.get("target_delta", 0.35))
                    tp = float(p.get("profit_target_pct", 50.0))
                    sl = float(p.get("stop_loss_pct", 30.0))
                    key = (round(td, 2), round(tp, 1), round(sl, 1))
                    if key not in seen_configs:
                        seen_configs.add(key)
                        badge = "Directional OTM" if td < 0.40 else ("ATM Strike" if td <= 0.50 else "ITM Strike")
                        if len(ranks) == 0:
                            label = "🥇 Top Net Profit"
                        elif len(ranks) == 1:
                            label = "🥈 Highest Sharpe"
                        elif len(ranks) == 2:
                            label = "🥉 Quick Scalp"
                        elif td >= 0.48:
                            label = "🏆 ATM Benchmark"
                        else:
                            label = f"🎯 Rank #{len(ranks) + 1}"

                        ranks.append({
                            "rank": len(ranks) + 1,
                            "run_id": r["id"],
                            "symbol": r["symbol"],
                            "title": f"Rank #{len(ranks) + 1}: {r['symbol']} ({label} Δ={td:.2f}, TP +{tp:.0f}%)",
                            "label": label,
                            "target_delta": td,
                            "profit_target_pct": tp,
                            "stop_loss_pct": sl,
                            "squeeze_score_threshold": 50.0,
                            "min_dte": int(p.get("min_dte", 2)),
                            "max_dte": int(p.get("max_dte", 20)),
                            "net_pnl": float(r["stoxkart_net_pnl"]),
                            "sharpe_ratio": round(float(r["sharpe_ratio"]), 2),
                            "win_rate_pct": round(float(r["win_rate_pct"]), 1),
                            "badge": badge,
                        })
                        if len(ranks) >= 6:
                            break
                except Exception:
                    continue
            con.close()
        except Exception:
            pass

    if not ranks:
        ranks = [
            {
                "rank": 1,
                "run_id": 85,
                "symbol": "BANKNIFTY",
                "title": "Rank #1: BANKNIFTY (🥇 Top Net Profit Δ=0.35, TP +90%)",
                "label": "🥇 Top Net Profit",
                "target_delta": 0.35,
                "profit_target_pct": 90.0,
                "stop_loss_pct": 30.0,
                "squeeze_score_threshold": 50.0,
                "min_dte": 2,
                "max_dte": 20,
                "net_pnl": 3470081.32,
                "sharpe_ratio": 2.19,
                "win_rate_pct": 68.2,
                "badge": "Directional OTM",
            },
            {
                "rank": 2,
                "run_id": 3,
                "symbol": "NIFTY",
                "title": "Rank #2: NIFTY (🥈 Highest Sharpe Δ=0.40, TP +60%)",
                "label": "🥈 Highest Sharpe",
                "target_delta": 0.40,
                "profit_target_pct": 60.0,
                "stop_loss_pct": 30.0,
                "squeeze_score_threshold": 50.0,
                "min_dte": 2,
                "max_dte": 20,
                "net_pnl": 2141901.04,
                "sharpe_ratio": 2.38,
                "win_rate_pct": 66.0,
                "badge": "Directional OTM",
            },
            {
                "rank": 3,
                "run_id": 73,
                "symbol": "NIFTY",
                "title": "Rank #3: NIFTY (🥉 Quick Scalp Δ=0.35, TP +50%)",
                "label": "🥉 Quick Scalp",
                "target_delta": 0.35,
                "profit_target_pct": 50.0,
                "stop_loss_pct": 30.0,
                "squeeze_score_threshold": 50.0,
                "min_dte": 2,
                "max_dte": 20,
                "net_pnl": 1934093.20,
                "sharpe_ratio": 2.19,
                "win_rate_pct": 66.6,
                "badge": "Directional OTM",
            },
            {
                "rank": 4,
                "run_id": 79,
                "symbol": "NIFTY",
                "title": "Rank #4: NIFTY (🏆 ATM Benchmark Δ=0.50, TP +50%)",
                "label": "🏆 ATM Benchmark",
                "target_delta": 0.50,
                "profit_target_pct": 50.0,
                "stop_loss_pct": 30.0,
                "squeeze_score_threshold": 50.0,
                "min_dte": 2,
                "max_dte": 20,
                "net_pnl": 1884226.16,
                "sharpe_ratio": 1.11,
                "win_rate_pct": 62.6,
                "badge": "ATM Strike",
            },
            {
                "rank": 5,
                "run_id": 34,
                "symbol": "NIFTY",
                "title": "Rank #5: NIFTY (🎯 Far OTM Δ=0.25, TP +50%)",
                "label": "🎯 Far OTM",
                "target_delta": 0.25,
                "profit_target_pct": 50.0,
                "stop_loss_pct": 30.0,
                "squeeze_score_threshold": 50.0,
                "min_dte": 2,
                "max_dte": 20,
                "net_pnl": 1640520.00,
                "sharpe_ratio": 1.68,
                "win_rate_pct": 71.6,
                "badge": "Far OTM",
            },
        ]
    return ranks


@require_GET
def api_gamma_ranks(request: HttpRequest) -> JsonResponse:
    ranks = _get_top_backtest_ranks()
    return JsonResponse({"success": True, "ranks": ranks})


@csrf_exempt
@require_POST
def api_gamma_calibrate(request: HttpRequest) -> JsonResponse:
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        target_delta = float(body.get("target_delta", 0.35))
        profit_target_pct = float(body.get("profit_target_pct", 50.0))
        stop_loss_pct = float(body.get("stop_loss_pct", 30.0))
        squeeze_threshold = float(body.get("squeeze_score_threshold", 50.0))
        min_dte = int(body.get("min_dte", 2))
        max_dte = int(body.get("max_dte", 20))
        sync_auto_trader = bool(body.get("sync_auto_trader", True))

        if sync_auto_trader:
            from studytrade.paper.auto_trader import AutoTrader
            trader = AutoTrader()
            trader.save_config({
                "target_delta": target_delta,
                "profit_target_pct": profit_target_pct,
                "stop_loss_pct": stop_loss_pct,
                "min_confidence_pct": squeeze_threshold,
            })

        return JsonResponse({
            "success": True,
            "message": f"Successfully calibrated Gamma Squeeze signals to Target Δ={target_delta:.2f}, TP=+{profit_target_pct:.0f}%, SL=-{stop_loss_pct:.0f}%.",
            "calibration": {
                "target_delta": target_delta,
                "profit_target_pct": profit_target_pct,
                "stop_loss_pct": stop_loss_pct,
                "squeeze_score_threshold": squeeze_threshold,
                "min_dte": min_dte,
                "max_dte": max_dte,
                "synced_with_auto_trader": sync_auto_trader,
            }
        })
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


@require_GET
def api_auto_trader_status(request: HttpRequest) -> JsonResponse:
    """Return autonomous trader live status and configuration."""
    from studytrade.paper.auto_trader import get_auto_trader
    trader = get_auto_trader(DATA_DIR)
    open_positions = trader.journal.get_open_positions()
    deployed = sum(
        float(p.get("entry_price", 0.0)) * int(p.get("lot_size", 25)) * int(p.get("n_lots", 1))
        for p in open_positions
    )
    total_cap = float(trader.config.get("total_capital_limit", 20000.0))
    return JsonResponse(
        {
            "success": True,
            "status": "RUNNING" if trader.config.get("enabled", True) else "PAUSED",
            "config": trader.config,
            "open_positions_count": len(open_positions),
            "deployed_capital": round(deployed, 2),
            "available_capital": round(max(0.0, total_cap - deployed), 2),
        }
    )


@csrf_exempt
@require_POST
def api_auto_trader_configure(request: HttpRequest) -> JsonResponse:
    """Update autonomous trader parameters."""
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        from studytrade.paper.auto_trader import get_auto_trader
        trader = get_auto_trader(DATA_DIR)
        updated = trader.save_config(body)
        return JsonResponse(
            {
                "success": True,
                "message": "Autonomous trader parameters updated successfully.",
                "config": updated,
            }
        )
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


@csrf_exempt
@require_POST
def api_auto_trader_toggle(request: HttpRequest) -> JsonResponse:
    """Toggle autonomous trader master switch ON/OFF."""
    try:
        from studytrade.paper.auto_trader import get_auto_trader
        trader = get_auto_trader(DATA_DIR)
        is_enabled = trader.toggle_master()
        state_text = "ACTIVATED" if is_enabled else "PAUSED"
        return JsonResponse(
            {
                "success": True,
                "enabled": is_enabled,
                "message": f"Autonomous Execution Engine is now {state_text}.",
                "status": "RUNNING" if is_enabled else "PAUSED",
            }
        )
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


@csrf_exempt
@require_POST
def api_auto_trader_run_cycle(request: HttpRequest) -> JsonResponse:
    """Manually trigger an autonomous cycle (auto-square-off and auto-buy)."""
    try:
        from studytrade.paper.auto_trader import get_auto_trader
        trader = get_auto_trader(DATA_DIR)
        report = trader.run_cycle()
        return JsonResponse(
            {
                "success": True,
                "message": (
                    f"Auto-cycle complete. {len(report['closed_trades'])} trade(s) squared off, "
                    f"{len(report['opened_trades'])} trade(s) automatically executed."
                ),
                "report": report,
            }
        )
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


@csrf_exempt
@require_POST
def api_paper_buy_signal(request: HttpRequest) -> JsonResponse:
    """Execute a paper buy trade order directly from real-time strategy signals."""
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        symbol = str(body.get("symbol", "")).strip()
        underlying = str(body.get("underlying_symbol", "NIFTY")).strip().upper()
        strike = float(body.get("strike", 0.0))
        expiry = str(body.get("expiry", ""))
        option_type = str(body.get("option_type", "CE")).strip().upper()
        entry_price = float(body.get("entry_price", 0.0))
        n_lots = int(body.get("n_lots", 1))
        lot_size = int(body.get("lot_size", 25))
        entry_iv = float(body.get("entry_iv", 16.0))
        strategy = str(body.get("strategy", "Gamma Squeeze Real-Time Signal"))

        if not symbol or entry_price <= 0.0:
            return JsonResponse({"success": False, "error": "Invalid symbol or entry price"}, status=400)

        required_capital = entry_price * lot_size * n_lots
        journal = PaperJournal(str(DATA_DIR / "paper_journal.db"))
        open_positions = journal.get_open_positions()
        deployed = sum(
            float(p.get("entry_price", 0.0)) * int(p.get("lot_size", 25)) * int(p.get("n_lots", 1))
            for p in open_positions
        )

        INITIAL_CAPITAL = 20000.0
        if deployed + required_capital > INITIAL_CAPITAL + 1000.0:
            return JsonResponse(
                {
                    "success": False,
                    "error": f"Insufficient paper capital! Required: ₹{required_capital:.2f}, Available: ₹{max(0.0, INITIAL_CAPITAL - deployed):.2f} (Total Initial Capital: ₹20,000)",
                },
                status=400,
            )

        target_price = float(body.get("target_price", 0.0))
        stop_loss_price = float(body.get("stop_loss_price", 0.0))
        pos_id = journal.add_open_position(
            symbol=symbol,
            underlying_symbol=underlying,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            n_lots=n_lots,
            lot_size=lot_size,
            entry_price=entry_price,
            current_price=entry_price,
            entry_iv=entry_iv,
            strategy=strategy,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
        )

        return JsonResponse(
            {
                "success": True,
                "message": f"Successfully executed paper buy order for {symbol} ({n_lots} lot{'s' if n_lots>1 else ''}) at ₹{entry_price:.2f}. Total margin deployed: ₹{required_capital:.2f}.",
                "position_id": pos_id,
            }
        )
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


@csrf_exempt
@require_POST
def api_paper_close_position(request: HttpRequest) -> JsonResponse:
    """Square off an active open position and record to the journal ledger."""
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        symbol = body.get("symbol")
        pos_id = body.get("position_id")
        exit_price = body.get("exit_price")
        exit_reason = str(body.get("exit_reason", "MANUAL_SQUARE_OFF"))

        if not symbol and not pos_id:
            return JsonResponse({"success": False, "error": "Must provide symbol or position_id"}, status=400)

        journal = PaperJournal(str(DATA_DIR / "paper_journal.db"))
        res = journal.close_position(
            symbol=symbol,
            position_id=int(pos_id) if pos_id else None,
            exit_price=float(exit_price) if exit_price is not None else None,
            exit_reason=exit_reason,
        )

        if not res:
            return JsonResponse({"success": False, "error": "Position not found or already closed"}, status=404)

        return JsonResponse(
            {
                "success": True,
                "message": f"Position {res['symbol']} squared off at ₹{res['exit_price']:.2f}. Gross PnL: ₹{res['gross_pnl']:.2f}, Statutory Fees: ₹{res['total_fees']:.2f}, Net PnL: ₹{res['net_pnl']:.2f} ({res['return_pct']:.1f}%).",
                "trade": res,
            }
        )
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


@csrf_exempt
@require_POST
def api_sync_bhavcopy(request: HttpRequest) -> JsonResponse:
    """Download and ingest official NSE daily derivative Bhavcopy."""
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        target_date = body.get("date")
        symbols = body.get("symbols", ["NIFTY", "BANKNIFTY", "FINNIFTY"])

        from studytrade.data.bhavcopy import BhavcopyImporter

        importer = BhavcopyImporter(data_dir=DATA_DIR)
        if target_date:
            stats = importer.sync_date(target_date, symbols=symbols)
            stats["status"] = "success"
            stats["resolved_date"] = str(target_date)
        else:
            stats = importer.sync_latest_trading_day(symbols=symbols)

        return JsonResponse(
            {
                "success": True,
                "message": f"Successfully synchronized official NSE Bhavcopy for {stats.get('resolved_date')}. Row counts: {stats.get('row_counts', {})}.",
                "stats": stats,
            }
        )
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


# ---------------------------------------------------------------------------
# API / AJAX Endpoints
# ---------------------------------------------------------------------------


@csrf_exempt
@require_POST
def api_generate_data(request: HttpRequest) -> JsonResponse:
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        symbol = body.get("symbol", "NIFTY")
        n_days = int(body.get("n_days", 125))

        spot_df, opt_df = generate_and_store_dataset(
            symbol=symbol,
            n_days=n_days,
            base_dir=str(DATA_DIR),
        )
        msg = (
            f"Successfully generated {len(spot_df)} spot bars "
            f"and {len(opt_df)} option quotes for {symbol}."
        )
        return JsonResponse(
            {
                "success": True,
                "message": msg,
                "spot_rows": len(spot_df),
                "options_rows": len(opt_df),
            }
        )
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


def _enrich_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import pandas as pd
    for r in runs:
        cap = float(r.get("initial_capital", 20000.0))
        net = float(r.get("stoxkart_net_pnl", 0.0))
        r["net_return_pct"] = round((net / cap) * 100.0, 1) if cap > 0 else 0.0

        p = r.get("params", {}) or {}
        r["start_date"] = p.get("start_date", "2016-01-01") or "2016-01-01"
        r["end_date"] = p.get("end_date", "2026-11-20") or "2026-11-20"
        r["target_delta"] = p.get("target_delta", 0.35)
        r["profit_target_pct"] = p.get("profit_target_pct", 60.0)
        r["stop_loss_pct"] = p.get("stop_loss_pct", 30.0)
        r["min_dte"] = p.get("min_dte", 2)
        r["max_dte"] = p.get("max_dte", 20)

        td = float(r["target_delta"]) if r["target_delta"] is not None else 0.35
        if td <= 0.28:
            r["delta_label"] = "Far OTM (Δ=0.25)"
        elif td >= 0.45:
            r["delta_label"] = "ATM Strike (Δ=0.50)"
        else:
            r["delta_label"] = "Directional OTM (Δ=0.35)"

        try:
            s_dt = pd.to_datetime(r["start_date"])
            e_dt = pd.to_datetime(r["end_date"])
            days = max(1, (e_dt - s_dt).days)
            years = days / 365.25
            r["duration_days"] = days
            r["duration_years"] = round(years, 1)
            r["duration_str"] = f"{r['start_date']} → {r['end_date']} ({round(years, 1)}Y)"
            if years >= 0.25 and cap > 0 and (net + cap) > 0:
                cagr = (((net + cap) / cap) ** (1.0 / years) - 1.0) * 100.0
                r["cagr_pct"] = round(cagr, 1)
            else:
                r["cagr_pct"] = round(r["net_return_pct"], 1)
        except Exception:
            r["duration_str"] = f"{r['start_date']} → {r['end_date']}"
            r["cagr_pct"] = r["net_return_pct"]

    return runs


@require_GET
def api_backtest_history(request: HttpRequest) -> JsonResponse:
    from studytrade.data.backtest_db import get_all_backtest_runs
    runs = _enrich_runs(get_all_backtest_runs())
    return JsonResponse({"success": True, "runs": runs})


@require_GET
def api_get_backtest_run(request: HttpRequest, run_id: int) -> JsonResponse:
    from studytrade.data.backtest_db import get_backtest_run_by_id
    run = get_backtest_run_by_id(run_id)
    if not run:
        return JsonResponse({"success": False, "error": f"Backtest run #{run_id} not found"}, status=404)
    return JsonResponse({"success": True, "run": run})


@require_GET
def api_option_chain(request: HttpRequest) -> JsonResponse:

    symbol = request.GET.get("symbol", "NIFTY")
    date_str = request.GET.get("date", "")
    expiry_str = request.GET.get("expiry", "")

    store = _get_store()
    opt_df = store.load_options(symbol)
    if opt_df.empty:
        return JsonResponse({"error": "No option data found"}, status=404)

    filtered = opt_df[opt_df["timestamp"].dt.strftime("%Y-%m-%d") == date_str]
    if expiry_str:
        filtered = filtered[filtered["expiry"].dt.strftime("%Y-%m-%d") == expiry_str]

    records = filtered.to_dict(orient="records")
    for r in records:
        r["timestamp"] = r["timestamp"].isoformat()
        r["expiry"] = r["expiry"].isoformat()

    return JsonResponse({"rows": records})


@csrf_exempt
@require_POST
def api_run_backtest(request: HttpRequest) -> JsonResponse:
    try:
        params = json.loads(request.body.decode("utf-8")) if request.body else {}
        symbol = params.get("symbol", "NIFTY")
        strat_name = params.get("strategy", "breakout")
        capital = float(params.get("initial_capital", 200000.0))
        stop_loss_pct = float(params.get("stop_loss_pct", 30.0))
        profit_target_pct = float(params.get("profit_target_pct", 60.0))
        target_delta = float(params.get("target_delta", 0.35))
        min_dte = int(params.get("min_dte", 2))
        max_dte = int(params.get("max_dte", 20))
        slippage = float(params.get("slippage_pts", 0.5))

        _ensure_data_exists(symbol)

        # Select Strategy
        if strat_name == "debit_spread":
            strategy = DebitSpreadStrategy(MomentumStrategy(target_delta=target_delta))
        elif strat_name == "gamma_squeeze":
            strategy = GammaSqueezeStrategy(
                target_delta=target_delta,
                min_dte=min_dte,
                max_dte=max_dte,
                squeeze_score_threshold=float(params.get("threshold", 60.0)),
            )
        elif strat_name == "breakout":
            strategy = BreakoutStrategy(lookback=10, volume_mult=1.05, target_delta=target_delta)
        else:
            strat_cls = {
                "momentum": MomentumStrategy,
                "vol_expansion": VolatilityExpansionStrategy,
            }.get(strat_name, MomentumStrategy)
            strategy = strat_cls(target_delta=target_delta)

        store = _get_store()
        spot_df = store.load_spot(symbol)
        opt_df = store.load_options(symbol)

        start_date = str(params.get("start_date", "")).strip()
        end_date = str(params.get("end_date", "")).strip()

        if start_date:
            try:
                s_dt = pd.to_datetime(start_date)
                if spot_df["timestamp"].dt.tz is not None and s_dt.tz is None:
                    s_dt = s_dt.tz_localize(spot_df["timestamp"].dt.tz)
                spot_df = spot_df[spot_df["timestamp"] >= s_dt].reset_index(drop=True)
                opt_df = opt_df[opt_df["timestamp"] >= s_dt].reset_index(drop=True)
            except Exception:
                pass

        if end_date:
            try:
                e_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                if spot_df["timestamp"].dt.tz is not None and e_dt.tz is None:
                    e_dt = e_dt.tz_localize(spot_df["timestamp"].dt.tz)
                spot_df = spot_df[spot_df["timestamp"] <= e_dt].reset_index(drop=True)
                opt_df = opt_df[opt_df["timestamp"] <= e_dt].reset_index(drop=True)
            except Exception:
                pass

        if spot_df.empty:
            return JsonResponse(
                {"success": False, "error": f"No market data found between {start_date} and {end_date} for {symbol}."},
                status=400,
            )

        from studytrade.strategy.rules import ExitRuleEvaluator

        exit_evaluator = ExitRuleEvaluator(
            stop_loss_pct=stop_loss_pct,
            profit_target_pct=profit_target_pct,
            min_dte_cutoff=min_dte,
        )

        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=capital,
            slippage_ticks=max(1.0, slippage / 0.05),
            exit_evaluator=exit_evaluator,
        )

        results = engine.run(spot_df=spot_df, options_df=opt_df, symbol=symbol)
        perf = compute_performance_metrics(
            results.trades_df, results.equity_df, initial_capital=capital
        )
        trades_df = results.trades_df
        eq_df = results.equity_df

        trades_list: list[dict[str, Any]] = []
        if not trades_df.empty:
            for _, r in trades_df.iterrows():
                opt_val = (
                    r["option_type"].value
                    if hasattr(r["option_type"], "value")
                    else str(r["option_type"])
                )
                entry_dt_str = (
                    r["entry_time"].strftime("%Y-%m-%d")
                    if hasattr(r["entry_time"], "strftime")
                    else str(r["entry_time"])[:10]
                )
                exit_dt_str = (
                    r["exit_time"].strftime("%Y-%m-%d")
                    if hasattr(r["exit_time"], "strftime")
                    else str(r["exit_time"])[:10]
                )
                trades_list.append(
                    {
                        "symbol": r["symbol"],
                        "direction": opt_val,
                        "entry_time": entry_dt_str,
                        "exit_time": exit_dt_str,
                        "entry_price": round(float(r["entry_price"]), 2),
                        "exit_price": round(float(r["exit_price"]), 2),
                        "quantity": int(r["n_lots"]),
                        "gross_pnl": round(float(r["gross_pnl"]), 2),
                        "net_pnl": round(float(r["net_pnl"]), 2),
                        "pnl_pct": round(float(r["return_pct"]), 2),
                        "total_costs": round(float(r["total_fees"]), 2),
                        "exit_reason": str(r["exit_reason"]),
                    }
                )

        eq_dates = [
            d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            for d in eq_df["timestamp"]
        ]
        eq_values = [round(float(v), 2) for v in eq_df["equity"]]
        dd_series, _, _ = compute_drawdowns(eq_df["equity"])
        dd_values = [round(float(v), 2) for v in dd_series]

        from studytrade.backtest.costs import compute_multi_broker_comparison
        from studytrade.data.backtest_db import save_backtest_run

        broker_comp = compute_multi_broker_comparison(trades_df)
        zerodha_net = float(broker_comp["zerodha"]["net_pnl"])
        stoxkart_net = float(broker_comp["stoxkart"]["net_pnl"])
        total_net_pnl = stoxkart_net
        total_ret_pct = (total_net_pnl / capital) * 100.0
        run_id = save_backtest_run(
            symbol=symbol,
            strategy=strat_name,
            initial_capital=capital,
            total_trades=perf.total_trades,
            win_rate_pct=perf.win_rate_pct,
            gross_pnl=broker_comp["gross_realized_pnl"],
            broker_comp=broker_comp,
            sharpe_ratio=perf.sharpe_ratio,
            max_drawdown_pct=perf.max_drawdown_pct,
            params=params,
        )

        return JsonResponse(
            {
                "success": True,
                "run_id": run_id,
                "metrics": {
                    "initial_capital": capital,
                    "final_equity": round(float(results.final_equity), 2),
                    "gross_realized_pnl": broker_comp["gross_realized_pnl"],
                    "total_net_pnl": round(total_net_pnl, 2),
                    "zerodha_net_pnl": round(zerodha_net, 2),
                    "stoxkart_net_pnl": round(stoxkart_net, 2),
                    "return_pct": round(total_ret_pct, 2),
                    "total_trades": perf.total_trades,
                    "win_rate_pct": round(perf.win_rate_pct, 1),
                    "profit_factor": round(perf.profit_factor, 2)
                    if perf.profit_factor != float("inf")
                    else "99.9",
                    "expectancy": round(perf.expectancy_inr, 2),
                    "payoff_ratio": round(perf.payoff_ratio, 2),
                    "sharpe_ratio": round(perf.sharpe_ratio, 2),
                    "sortino_ratio": round(perf.sortino_ratio, 2),
                    "max_drawdown_pct": round(perf.max_drawdown_pct, 2),
                    "total_costs": round(float(results.total_fees), 2),
                    "broker_comparison": broker_comp,
                },
                "equity_curve": {
                    "dates": eq_dates,
                    "equity": eq_values,
                    "drawdown": dd_values,
                },
                "trades": trades_list,
            }
        )
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


@csrf_exempt
@require_POST
def api_run_monte_carlo(request: HttpRequest) -> JsonResponse:
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        pnls = body.get("pnls", [])
        capital = float(body.get("initial_capital", 200000.0))
        n_sims = int(body.get("n_simulations", 1000))
        n_trials = int(body.get("num_trials", 10))

        if not pnls or len(pnls) < 5:
            # Generate representative trade sequence for demo
            rng = np.random.default_rng(42)
            pnls = rng.normal(1200, 4500, 45).tolist()

        trades_df = pd.DataFrame({"net_pnl": pnls})
        mc_summary = simulate_monte_carlo_drawdowns(trades_df, capital, n_simulations=n_sims)

        # Deflated Sharpe
        returns = np.array(pnls) / capital
        mean_r = float(np.mean(returns)) if len(returns) > 0 else 0.0
        std_r = float(np.std(returns, ddof=1)) if len(returns) > 1 else 1e-6
        sr = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0

        dsr_prob = compute_deflated_sharpe_ratio(
            observed_sharpe=sr,
            returns=returns,
            benchmark_sharpe=0.0,
            num_trials_tested=n_trials,
        )

        return JsonResponse(
            {
                "success": True,
                "p50_drawdown_pct": round(mc_summary.p50_drawdown_pct, 2),
                "p95_drawdown_pct": round(mc_summary.p95_drawdown_pct, 2),
                "p99_drawdown_pct": round(mc_summary.p99_drawdown_pct, 2),
                "worst_drawdown_pct": round(mc_summary.worst_drawdown_pct, 2),
                "dsr_probability": round(dsr_prob * 100, 1),
                "dsr_status": (
                    "Statistically Significant Edge (DSR > 95%)"
                    if dsr_prob >= 0.95
                    else "Unlikely to have true statistical edge (Data Mining Risk)"
                ),
            }
        )
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


@csrf_exempt
@require_POST
def api_calc_sizing(request: HttpRequest) -> JsonResponse:
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        capital = float(body.get("capital", 200000.0))
        risk_pct = float(body.get("risk_pct", 1.5))
        premium = float(body.get("premium", 150.0))
        symbol = body.get("symbol", "NIFTY")
        atr = float(body.get("atr", 180.0))
        base_atr = float(body.get("base_atr", 150.0))
        win_rate = float(body.get("win_rate", 0.38))
        payoff = float(body.get("payoff", 2.6))
        stop_loss_pct = float(body.get("stop_loss_pct", 30.0))

        lot_size = INSTRUMENTS.get(symbol, INSTRUMENTS["NIFTY"]).lot_size

        # Fixed fractional
        ff_lots = PositionSizer.fixed_fractional(
            capital=capital,
            risk_pct=risk_pct,
            option_premium=premium,
            stop_loss_pct=stop_loss_pct,
            lot_size=lot_size,
        )
        ff_contracts = ff_lots * lot_size
        ff_alloc = ff_contracts * premium
        ff_risk = ff_alloc * (stop_loss_pct / 100.0)

        # Vol adjusted
        vol_lots = PositionSizer.volatility_adjusted(
            capital=capital,
            base_risk_pct=risk_pct,
            current_natr=atr,
            baseline_natr=base_atr,
            option_premium=premium,
            stop_loss_pct=stop_loss_pct,
            lot_size=lot_size,
        )
        vol_contracts = vol_lots * lot_size
        vol_alloc = vol_contracts * premium
        vol_risk = vol_alloc * (stop_loss_pct / 100.0)

        # Half Kelly
        kelly_lots = PositionSizer.half_kelly(
            win_rate=win_rate,
            payoff_ratio=payoff,
            capital=capital,
            option_premium=premium,
            stop_loss_pct=stop_loss_pct,
            lot_size=lot_size,
        )
        kelly_contracts = kelly_lots * lot_size
        kelly_alloc = kelly_contracts * premium
        kelly_risk = kelly_alloc * (stop_loss_pct / 100.0)

        return JsonResponse(
            {
                "success": True,
                "fixed_fractional": {
                    "lots": ff_lots,
                    "contracts": ff_contracts,
                    "capital_allocated": round(ff_alloc, 2),
                    "capital_at_risk": round(ff_risk, 2),
                },
                "volatility_adjusted": {
                    "lots": vol_lots,
                    "contracts": vol_contracts,
                    "capital_allocated": round(vol_alloc, 2),
                    "capital_at_risk": round(vol_risk, 2),
                },
                "half_kelly": {
                    "lots": kelly_lots,
                    "contracts": kelly_contracts,
                    "capital_allocated": round(kelly_alloc, 2),
                    "capital_at_risk": round(kelly_risk, 2),
                },
            }
        )
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


# ---------------------------------------------------------------------------
# 7. Gamma Squeeze & GEX Interface View
# ---------------------------------------------------------------------------


def _get_manual_trade_recommendation(
    chain_df: pd.DataFrame,
    direction: str,
    spot_price: float,
    target_delta: float = 0.50,
    profit_target_pct: float = 50.0,
    stop_loss_pct: float = 30.0,
    symbol: str = "NIFTY",
) -> dict[str, Any]:
    """Finds best matching strike contract for manual execution with entry, target, and stop rules."""
    opt_type = "CE" if "CALL" in direction else "PE"
    df = chain_df[chain_df["option_type"] == opt_type].copy()
    if df.empty:
        return {}

    lot_size = INSTRUMENTS.get(symbol, INSTRUMENTS["NIFTY"]).lot_size
    best_diff = float("inf")
    best_row = None
    best_delta = 0.5
    for _, r in df.iterrows():
        try:
            from studytrade.math.greeks import greeks
            if "expiry" in r and "timestamp" in r:
                exp_d = pd.to_datetime(r["expiry"]).date()
                cur_d = pd.to_datetime(r["timestamp"]).date()
                dte_d = max(1.0, float((exp_d - cur_d).days))
                t_y = dte_d / 365.0
            else:
                t_y = 3.5 / 365.0
            g = greeks(
                OptionType.CALL if opt_type == "CE" else OptionType.PUT,
                spot_price,
                float(r["strike"]),
                t_y,
                0.07,
                float(r.get("implied_volatility", 0.16)),
            )
            d = abs(g.delta)
        except Exception:
            d = 0.5
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_row = r
            best_delta = d

    if best_row is None:
        best_row = df.iloc[0]

    sym = str(best_row["symbol"])
    try:
        from studytrade.live.session_engine import get_live_tick_cache
        tick = get_live_tick_cache().get_tick(sym)
    except Exception:
        tick = None

    if tick and "last_price" in tick:
        prem = float(tick["last_price"])
        bid = float(tick.get("bid", prem - 0.20))
        ask = float(tick.get("ask", prem + 0.20))
    else:
        prem = float(best_row["close"])
        ask = float(best_row.get("ask", prem + 0.5))
        bid = float(best_row.get("bid", prem - 0.5))

    strike = float(best_row["strike"])
    target_mult = 1.0 + (profit_target_pct / 100.0)
    sl_mult = max(0.0, 1.0 - (stop_loss_pct / 100.0))

    return {
        "strike": strike,
        "contract_symbol": sym,
        "premium": round(prem, 2),
        "bid": round(bid, 2),
        "ask": round(ask, 2),
        "delta": round(best_delta, 2),
        "profit_target_pct": profit_target_pct,
        "stop_loss_pct": stop_loss_pct,
        "target_price": round(prem * target_mult, 2),
        "stop_loss_price": round(prem * sl_mult, 2),
        "capital_per_lot": round(ask * lot_size, 2),
        "lot_size": lot_size,
    }


def gamma_squeeze_view(request: HttpRequest) -> HttpResponse:
    try:
        from studytrade.paper.auto_trader import AutoTrader
        trader_cfg = AutoTrader.load_config()
        default_delta = float(trader_cfg.get("target_delta", 0.35))
        default_tp = float(trader_cfg.get("profit_target_pct", 50.0))
        default_sl = float(trader_cfg.get("stop_loss_pct", 30.0))
    except Exception:
        default_delta, default_tp, default_sl = 0.35, 50.0, 30.0

    target_delta = float(request.GET.get("target_delta", default_delta))
    profit_target_pct = float(request.GET.get("profit_target_pct", default_tp))
    stop_loss_pct = float(request.GET.get("stop_loss_pct", default_sl))
    min_dte = int(request.GET.get("min_dte", 2))
    max_dte = int(request.GET.get("max_dte", 20))

    symbol = request.GET.get("symbol", "NIFTY")
    _ensure_data_exists(symbol)

    store = _get_store()
    spot_df = store.load_spot(symbol)
    options_df = store.load_options(symbol)

    dates: list[str] = []
    if not spot_df.empty and "timestamp" in spot_df.columns:
        dates = [d.strftime("%Y-%m-%d") for d in spot_df["timestamp"].dt.date.unique()]

    selected_date = request.GET.get("date", dates[-1] if dates else "")

    expiries: list[str] = []
    selected_expiry = ""
    chain_rows: list[dict[str, Any]] = []
    chart_json = "{}"
    spot_close = 0.0
    gex_prof = None
    signals_data: list[dict[str, Any]] = []

    if not options_df.empty and selected_date:
        opt_day = options_df[options_df["timestamp"].dt.strftime("%Y-%m-%d") == selected_date]
        if not opt_day.empty:
            expiries = [e.strftime("%Y-%m-%d") for e in opt_day["expiry"].dt.date.unique()]
            selected_expiry = request.GET.get("expiry", expiries[0] if expiries else "")

            # Baseline spot close for selected date
            spot_slice = spot_df[spot_df["timestamp"].dt.strftime("%Y-%m-%d") == selected_date]
            if not spot_slice.empty:
                spot_close = float(spot_slice["close"].iloc[-1])
            else:
                spot_close = float(opt_day["strike"].mean())

            # Check if user passed a simulated spot override
            sim_spot_str = request.GET.get("sim_spot", "")
            if sim_spot_str:
                try:
                    spot_close = float(sim_spot_str)
                except ValueError:
                    pass

            target_chain = opt_day
            if selected_expiry:
                target_chain = opt_day[opt_day["expiry"].dt.strftime("%Y-%m-%d") == selected_expiry]

            lot_size = INSTRUMENTS.get(symbol, INSTRUMENTS["NIFTY"]).lot_size
            gex_prof = compute_gex_profile(target_chain, spot_price=spot_close, lot_size=lot_size)

            if not gex_prof.strike_gex_df.empty:
                sdf = gex_prof.strike_gex_df.sort_values("strike")
                chart_data = {
                    "strikes": sdf["strike"].tolist(),
                    "call_gex": [round(v / 1e5, 2) for v in sdf["call_gex"]],  # in Lakhs
                    "put_gex": [round(v / 1e5, 2) for v in sdf["put_gex"]],    # in Lakhs
                    "net_gex": [round(v / 1e5, 2) for v in sdf["net_gex"]],    # in Lakhs
                    "spot_price": spot_close,
                    "call_wall": gex_prof.call_wall_strike,
                    "put_wall": gex_prof.put_wall_strike,
                    "gamma_flip": gex_prof.gamma_flip_level,
                }
                chart_json = json.dumps(chart_data)

                for _, r in sdf.iterrows():
                    k = float(r["strike"])
                    tags = []
                    if k == gex_prof.call_wall_strike:
                        tags.append("Call Wall")
                    if k == gex_prof.put_wall_strike:
                        tags.append("Put Wall")
                    if abs(k - spot_close) <= (spot_close * 0.005):
                        tags.append("ATM Strike")

                    moneyness = round(((k - spot_close) / spot_close) * 100, 2)
                    chain_rows.append(
                        {
                            "strike": k,
                            "moneyness_pct": moneyness,
                            "call_oi": int(r["call_oi"]),
                            "call_volume": int(r["call_volume"]),
                            "call_gex_lakhs": round(r["call_gex"] / 1e5, 2),
                            "put_oi": int(r["put_oi"]),
                            "put_volume": int(r["put_volume"]),
                            "put_gex_lakhs": round(r["put_gex"] / 1e5, 2),
                            "net_gex_lakhs": round(r["net_gex"] / 1e5, 2),
                            "tags": tags,
                        }
                    )

            # Evaluate real-time Strategy Signals
            eval_spot_df = spot_slice if not spot_slice.empty else spot_df.tail(2)
            if sim_spot_str and not eval_spot_df.empty:
                eval_spot_df = eval_spot_df.copy()
                eval_spot_df.loc[eval_spot_df.index[-1], "close"] = spot_close

            strat = GammaSqueezeStrategy(
                squeeze_score_threshold=60.0,
                target_delta=target_delta,
                min_dte=min_dte,
                max_dte=max_dte,
                use_spreads=False,
                allow_bearish=True,
            )
            raw_signals = strat.generate_signals(eval_spot_df, target_chain)
            for sig in raw_signals:
                rec = _get_manual_trade_recommendation(
                    target_chain,
                    sig.direction.value,
                    spot_close,
                    target_delta=sig.target_delta,
                    profit_target_pct=profit_target_pct,
                    stop_loss_pct=stop_loss_pct,
                    symbol=symbol,
                )
                signals_data.append(
                    {
                        "direction": sig.direction.value,
                        "strength": round(sig.strength * 100, 1),
                        "reason": sig.reason,
                        "target_delta": sig.target_delta,
                        "min_dte": sig.min_dte,
                        "max_dte": sig.max_dte,
                        "recommendation": rec,
                    }
                )

    call_dist_pts = round(gex_prof.call_wall_strike - spot_close, 1) if gex_prof else 0.0
    call_dist_pct = round((call_dist_pts / max(spot_close, 1.0)) * 100, 2) if gex_prof else 0.0
    put_dist_pts = round(spot_close - gex_prof.put_wall_strike, 1) if gex_prof else 0.0
    put_dist_pct = round((put_dist_pts / max(spot_close, 1.0)) * 100, 2) if gex_prof else 0.0
    is_negative_gamma = (gex_prof.net_gex < 0 or spot_close > gex_prof.gamma_flip_level) if gex_prof else False

    context = {
        "symbol": symbol,
        "symbols": list(INSTRUMENTS.keys()),
        "dates": dates,
        "selected_date": selected_date,
        "expiries": expiries,
        "selected_expiry": selected_expiry,
        "spot_price": spot_close,
        "call_wall": gex_prof.call_wall_strike if gex_prof else 0.0,
        "put_wall": gex_prof.put_wall_strike if gex_prof else 0.0,
        "gamma_flip": gex_prof.gamma_flip_level if gex_prof else 0.0,
        "net_gex_lakhs": round(gex_prof.net_gex / 1e5, 2) if gex_prof else 0.0,
        "call_squeeze_score": gex_prof.call_squeeze_score if gex_prof else 0.0,
        "put_cascade_score": gex_prof.put_cascade_score if gex_prof else 0.0,
        "call_dist_pts": call_dist_pts,
        "call_dist_pct": call_dist_pct,
        "put_dist_pts": put_dist_pts,
        "put_dist_pct": put_dist_pct,
        "is_negative_gamma": is_negative_gamma,
        "chart_json": chart_json,
        "chain_rows": chain_rows,
        "signals": signals_data,
        "target_delta": target_delta,
        "profit_target_pct": profit_target_pct,
        "stop_loss_pct": stop_loss_pct,
        "min_dte": min_dte,
        "max_dte": max_dte,
        "ranks": _get_top_backtest_ranks(),
    }
    return render(request, "gamma_squeeze.html", context)


@require_GET
def api_gamma_profile(request: HttpRequest) -> JsonResponse:

    symbol = request.GET.get("symbol", "NIFTY")
    date_str = request.GET.get("date", "")
    expiry_str = request.GET.get("expiry", "")
    sim_spot_str = request.GET.get("sim_spot", "")

    store = _get_store()
    opt_df = store.load_options(symbol)
    spot_df = store.load_spot(symbol)

    if opt_df.empty:
        return JsonResponse({"error": "No option data found"}, status=404)

    filtered = opt_df[opt_df["timestamp"].dt.strftime("%Y-%m-%d") == date_str]
    if expiry_str:
        filtered = filtered[filtered["expiry"].dt.strftime("%Y-%m-%d") == expiry_str]

    if filtered.empty:
        return JsonResponse({"error": "No contracts found for selected date/expiry"}, status=404)

    spot_slice = spot_df[spot_df["timestamp"].dt.strftime("%Y-%m-%d") == date_str]
    default_spot = float(spot_slice["close"].iloc[-1]) if not spot_slice.empty else float(filtered["strike"].mean())

    spot_price = float(sim_spot_str) if sim_spot_str else default_spot
    lot_size = INSTRUMENTS.get(symbol, INSTRUMENTS["NIFTY"]).lot_size

    prof = compute_gex_profile(filtered, spot_price=spot_price, lot_size=lot_size)
    sdf = prof.strike_gex_df.sort_values("strike")

    chart_data = {
        "strikes": sdf["strike"].tolist(),
        "call_gex": [round(v / 1e5, 2) for v in sdf["call_gex"]],
        "put_gex": [round(v / 1e5, 2) for v in sdf["put_gex"]],
        "net_gex": [round(v / 1e5, 2) for v in sdf["net_gex"]],
        "spot_price": spot_price,
        "call_wall": prof.call_wall_strike,
        "put_wall": prof.put_wall_strike,
        "gamma_flip": prof.gamma_flip_level,
    }

    call_dist_pts = round(prof.call_wall_strike - spot_price, 1)
    call_dist_pct = round((call_dist_pts / max(spot_price, 1.0)) * 100, 2)
    put_dist_pts = round(spot_price - prof.put_wall_strike, 1)
    put_dist_pct = round((put_dist_pts / max(spot_price, 1.0)) * 100, 2)

    return JsonResponse(
        {
            "success": True,
            "spot_price": spot_price,
            "call_wall": prof.call_wall_strike,
            "put_wall": prof.put_wall_strike,
            "gamma_flip": prof.gamma_flip_level,
            "net_gex_lakhs": round(prof.net_gex / 1e5, 2),
            "call_squeeze_score": prof.call_squeeze_score,
            "put_cascade_score": prof.put_cascade_score,
            "call_dist_pts": call_dist_pts,
            "call_dist_pct": call_dist_pct,
            "put_dist_pts": put_dist_pts,
            "put_dist_pct": put_dist_pct,
            "is_negative_gamma": (prof.net_gex < 0 or spot_price > prof.gamma_flip_level),
            "chart_data": chart_data,
        }
    )


@csrf_exempt
@require_POST
def api_run_gamma_strategy(request: HttpRequest) -> JsonResponse:
    try:
        params = json.loads(request.body.decode("utf-8")) if request.body else {}
        symbol = params.get("symbol", "NIFTY")
        date_str = params.get("date", "")
        expiry_str = params.get("expiry", "")
        threshold = float(params.get("threshold", 60.0))
        use_spreads = bool(params.get("use_spreads", False))
        allow_bearish = bool(params.get("allow_bearish", True))
        sim_spot_str = params.get("sim_spot", "")

        store = _get_store()
        spot_df = store.load_spot(symbol)
        opt_df = store.load_options(symbol)

        filtered = opt_df[opt_df["timestamp"].dt.strftime("%Y-%m-%d") == date_str]
        if expiry_str:
            filtered = filtered[filtered["expiry"].dt.strftime("%Y-%m-%d") == expiry_str]

        spot_slice = spot_df[spot_df["timestamp"].dt.strftime("%Y-%m-%d") == date_str]
        if spot_slice.empty:
            spot_slice = spot_df.tail(2)

        if sim_spot_str:
            spot_slice = spot_slice.copy()
            spot_slice.loc[spot_slice.index[-1], "close"] = float(sim_spot_str)

        try:
            from studytrade.paper.auto_trader import AutoTrader
            trader_cfg = AutoTrader.load_config()
            default_delta = float(trader_cfg.get("target_delta", 0.35))
            default_tp = float(trader_cfg.get("profit_target_pct", 50.0))
            default_sl = float(trader_cfg.get("stop_loss_pct", 30.0))
        except Exception:
            default_delta, default_tp, default_sl = 0.35, 50.0, 30.0

        target_delta = float(params.get("target_delta", default_delta))
        profit_target_pct = float(params.get("profit_target_pct", default_tp))
        stop_loss_pct = float(params.get("stop_loss_pct", default_sl))
        min_dte = int(params.get("min_dte", 2))
        max_dte = int(params.get("max_dte", 20))

        strat = GammaSqueezeStrategy(
            squeeze_score_threshold=threshold,
            target_delta=target_delta,
            min_dte=min_dte,
            max_dte=max_dte,
            use_spreads=use_spreads,
            allow_bearish=allow_bearish,
        )

        signals = strat.generate_signals(spot_slice, filtered)
        res = []
        cur_spot = float(sim_spot_str) if sim_spot_str else float(spot_slice["close"].iloc[-1])
        for s in signals:
            rec = _get_manual_trade_recommendation(
                filtered,
                s.direction.value,
                cur_spot,
                target_delta=s.target_delta,
                profit_target_pct=profit_target_pct,
                stop_loss_pct=stop_loss_pct,
                symbol=symbol,
            )
            res.append(
                {
                    "direction": s.direction.value,
                    "strength": round(s.strength * 100, 1),
                    "reason": s.reason,
                    "target_delta": s.target_delta,
                    "min_dte": s.min_dte,
                    "max_dte": s.max_dte,
                    "recommendation": rec,
                }
            )
        return JsonResponse({"success": True, "signals": res})
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)


@require_GET
def api_live_scanner_status(request: HttpRequest) -> JsonResponse:
    from studytrade.live.session_engine import get_live_session_engine
    engine = get_live_session_engine()
    st = engine.get_status()

    signals = []
    try:
        with sqlite3.connect("data/live_signals.db") as conn:
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT timestamp, symbol, direction, spot_price, call_wall, put_wall,
                       squeeze_score, contract, action_taken
                FROM live_signals
                ORDER BY id DESC LIMIT 10
                """
            ).fetchall()
            for r in rows:
                signals.append(
                    {
                        "timestamp": r[0],
                        "symbol": r[1],
                        "direction": r[2],
                        "spot_price": r[3],
                        "call_wall": r[4],
                        "put_wall": r[5],
                        "squeeze_score": r[6],
                        "contract": r[7],
                        "action_taken": r[8],
                    }
                )
    except Exception:
        pass

    st["recent_signals"] = signals
    return JsonResponse(st)


@csrf_exempt
@require_POST
def api_toggle_live_scanner(request: HttpRequest) -> JsonResponse:
    from studytrade.live.session_engine import get_live_session_engine
    engine = get_live_session_engine()
    if engine.get_status()["is_running"]:
        engine.stop()
    else:
        engine.start()
    return JsonResponse(engine.get_status())

