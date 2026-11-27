"""
Data-driven recalibration of the Gamma Squeeze option-buying parameters.

Runs a small parameter grid through the backtest engine on the most recent
``lookback_days`` of stored data. Candidates are fitted on the older 70% and ranked
by their result on the newest 30% (out-of-sample), to limit curve-fitting.
"""

from __future__ import annotations

import itertools
import json
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from studytrade.analytics.metrics import compute_performance_metrics
from studytrade.backtest.engine import BacktestEngine
from studytrade.strategy.rules import ExitRuleEvaluator
from studytrade.strategy.signals.gamma_squeeze import GammaSqueezeStrategy

DEFAULT_GRID: dict[str, list[float]] = {
    "target_delta": [0.30, 0.40, 0.50],
    "profit_target_pct": [50.0, 70.0, 100.0],
    "stop_loss_pct": [30.0],
    "squeeze_score_threshold": [50.0, 65.0],
}
MIN_TRADES_PER_WINDOW = 5


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    mask = df["timestamp"] >= start
    if end is not None:
        mask &= df["timestamp"] < end
    return df[mask].reset_index(drop=True)


def _run(spot: pd.DataFrame, opts: pd.DataFrame, symbol: str, p: dict[str, float],
         min_dte: int, max_dte: int, capital: float) -> dict[str, float]:
    strategy = GammaSqueezeStrategy(
        squeeze_score_threshold=p["squeeze_score_threshold"],
        target_delta=p["target_delta"],
        min_dte=min_dte,
        max_dte=max_dte,
    )
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=capital,
        exit_evaluator=ExitRuleEvaluator(
            stop_loss_pct=p["stop_loss_pct"],
            profit_target_pct=p["profit_target_pct"],
            min_dte_cutoff=min_dte,
        ),
    )
    res = engine.run(spot, opts, symbol=symbol)
    perf = compute_performance_metrics(res.trades_df, res.equity_df, initial_capital=capital)
    return {
        "trades": perf.total_trades,
        "net_pnl": round(res.net_profit, 2),
        "win_rate_pct": round(perf.win_rate_pct, 1),
        "profit_factor": round(perf.profit_factor, 2),
        "expectancy_inr": round(perf.expectancy_inr, 2),
        "sharpe": round(perf.sharpe_ratio, 2),
        "max_dd_pct": round(perf.max_drawdown_pct, 2),
    }


def recalibrate_gamma_squeeze(
    spot_df: pd.DataFrame,
    options_df: pd.DataFrame,
    symbol: str = "NIFTY",
    lookback_days: int = 180,
    min_dte: int = 2,
    max_dte: int = 20,
    capital: float = 500000.0,
    grid: dict[str, list[float]] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """
    Returns ``{"best": {...params, in_sample, out_of_sample}, "candidates": [...], "window": {...}}``
    or ``best=None`` when no candidate trades enough in both windows.
    """
    grid = grid or DEFAULT_GRID
    spot_df = spot_df.sort_values("timestamp")
    end_ts = spot_df["timestamp"].max()
    start_ts = end_ts - pd.Timedelta(days=lookback_days)
    split_ts = start_ts + (end_ts - start_ts) * 0.7

    spot_in, spot_out = _slice(spot_df, start_ts, split_ts), _slice(spot_df, split_ts, None)
    opts_in, opts_out = _slice(options_df, start_ts, split_ts), _slice(options_df, split_ts, None)
    if spot_in.empty or spot_out.empty or opts_in.empty or opts_out.empty:
        raise ValueError("Not enough recent data to recalibrate")

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    candidates: list[dict[str, Any]] = []
    for i, combo in enumerate(combos):
        if on_progress:
            on_progress(i, len(combos))
        p = dict(zip(keys, combo, strict=True))
        try:
            ins = _run(spot_in, opts_in, symbol, p, min_dte, max_dte, capital)
            oos = _run(spot_out, opts_out, symbol, p, min_dte, max_dte, capital)
        except Exception:
            continue
        eligible = (
            ins["trades"] >= MIN_TRADES_PER_WINDOW
            and oos["trades"] >= MIN_TRADES_PER_WINDOW
            and ins["expectancy_inr"] > 0
        )
        candidates.append({**p, "in_sample": ins, "out_of_sample": oos, "eligible": eligible})

    ranked = sorted(
        candidates,
        key=lambda c: (c["eligible"], c["out_of_sample"]["expectancy_inr"], c["out_of_sample"]["sharpe"]),
        reverse=True,
    )
    best = ranked[0] if ranked and ranked[0]["eligible"] and ranked[0]["out_of_sample"]["expectancy_inr"] > 0 else None
    return {
        "best": best,
        "candidates": ranked[:10],
        "window": {
            "start": start_ts.date().isoformat(),
            "split": split_ts.date().isoformat(),
            "end": end_ts.date().isoformat(),
            "min_dte": min_dte,
            "max_dte": max_dte,
        },
    }


_job_lock = threading.Lock()
_job: dict[str, Any] = {"state": "idle"}


def get_calibration_status() -> dict[str, Any]:
    with _job_lock:
        return dict(_job)


def start_calibration_job(
    data_dir: str | Path,
    symbol: str = "NIFTY",
    lookback_days: int = 180,
    apply: bool = True,
) -> bool:
    """Run recalibration in a background thread. Returns False if one is already running."""
    with _job_lock:
        if _job.get("state") == "running":
            return False
        _job.clear()
        _job.update(state="running", symbol=symbol, done=0, total=0, started=datetime.now().isoformat(timespec="seconds"))

    def _progress(done: int, total: int) -> None:
        with _job_lock:
            _job.update(done=done, total=total)

    def _work() -> None:
        try:
            from studytrade.data.store import DataStore

            store = DataStore(data_dir)
            result = recalibrate_gamma_squeeze(
                store.load_spot(symbol),
                store.load_options(symbol),
                symbol=symbol,
                lookback_days=lookback_days,
                on_progress=_progress,
            )
            applied = False
            best = result["best"]
            if best and apply:
                from studytrade.paper.auto_trader import get_auto_trader

                get_auto_trader(data_dir).save_config({
                    "target_delta": best["target_delta"],
                    "profit_target_pct": best["profit_target_pct"],
                    "stop_loss_pct": best["stop_loss_pct"],
                    "min_confidence_pct": best["squeeze_score_threshold"],
                })
                applied = True
            result.update(symbol=symbol, applied=applied, calibrated_at=datetime.now().isoformat(timespec="seconds"))
            Path(data_dir, "gamma_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            with _job_lock:
                _job.update(state="done", result=result)
        except Exception as exc:
            with _job_lock:
                _job.update(state="error", error=str(exc))

    threading.Thread(target=_work, name="studytrade-calibration", daemon=True).start()
    return True
