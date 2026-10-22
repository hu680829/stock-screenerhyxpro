"""Fetch a point-in-time portfolio snapshot and write a fail-closed JSON report.

This is monitoring only: it does not place orders. Returns are calculated over
the latest two dates shared by all holdings and USD/CNH to avoid mixing
different market sessions. The configured weights are inherited from the
existing project configuration and should be reviewed before use.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Mapping

import pandas as pd

from portfolio_risk import portfolio_snapshot

LOGGER = logging.getLogger("portfolio_monitor")
DEFAULT_WEIGHTS = {
    "AAPL": 0.15,
    "NVDA": 0.20,
    "BTC-USD": 0.15,
    "000300.SS": 0.20,
    "SGOV": 0.30,
}
USD_TICKERS = {"AAPL", "NVDA", "BTC-USD", "SGOV"}


def calculate_snapshot(close_prices: pd.DataFrame, target_weights: Mapping[str, float] = DEFAULT_WEIGHTS) -> dict:
    """Calculate a snapshot from close prices indexed by date, requiring two shared sessions."""
    if close_prices is None or close_prices.empty:
        raise ValueError("No price data supplied")
    required = set(target_weights) | {"USDCNH=X"}
    missing_columns = sorted(required - set(close_prices.columns))
    if missing_columns:
        raise ValueError(f"Missing price columns: {', '.join(missing_columns)}")

    data = close_prices[list(required)].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.where(data > 0)
    data = data.sort_index()
    # Require all holdings and FX on the same two observations; do not silently
    # substitute another date or calculate a partial portfolio.
    shared = data.dropna(subset=list(required))
    if len(shared) < 2:
        raise ValueError("Fewer than two shared valid sessions for all holdings and USD/CNH")

    previous = shared.iloc[-2]
    latest = shared.iloc[-1]
    local_returns = {
        ticker: float(latest[ticker] / previous[ticker] - 1.0)
        for ticker in target_weights
    }
    fx_return = float(latest["USDCNH=X"] / previous["USDCNH=X"] - 1.0)
    result = portfolio_snapshot(
        local_returns,
        target_weights,
        usd_tickers=USD_TICKERS & set(target_weights),
        usd_cnh_return=fx_return,
    )
    result["previous_session"] = str(shared.index[-2].date() if hasattr(shared.index[-2], "date") else shared.index[-2])
    result["latest_session"] = str(shared.index[-1].date() if hasattr(shared.index[-1], "date") else shared.index[-1])
    result["usd_cnh_return"] = fx_return
    result["status"] = "ok"
    return result


def fetch_close_prices(period: str = "15d") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
    tickers = list(DEFAULT_WEIGHTS) + ["USDCNH=X"]
    raw = yf.download(
        tickers, period=period, interval="1d", auto_adjust=True,
        progress=False, threads=False, group_by="column",
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no portfolio price data")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("Yahoo Finance response has no Close field")
        close = raw["Close"].copy()
    else:
        raise RuntimeError("Yahoo Finance response did not contain multi-ticker close columns")
    close = close.loc[:, ~close.columns.duplicated(keep="last")]
    return close


def write_report(output: str | Path = "reports/portfolio_snapshot.json") -> dict:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = calculate_snapshot(fetch_close_prices())
    except Exception as exc:
        LOGGER.exception("Portfolio snapshot unavailable")
        snapshot = {
            "status": "data_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "rebalance_alerts": {},
        }
    destination.write_text(
        json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Portfolio snapshot status: {snapshot['status']} (saved to {destination})")
    return snapshot


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    write_report(os.getenv("PORTFOLIO_REPORT_PATH", "reports/portfolio_snapshot.json"))
    # Data failure is explicitly represented in the report; don't prevent the
    # separate stock ranking report from being delivered.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
