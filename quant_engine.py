"""Transparent, long-only multi-factor stock screener.

This module ranks historical price series; it is a research aid, not investment advice.
Signals are calculated from completed daily bars. No orders are placed.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("quant_engine")

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "AMD", "TSM", "SPY"]


def _close_series(history: pd.DataFrame) -> pd.Series:
    if "Close" not in history.columns:
        raise ValueError("history must contain a 'Close' column")
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if len(close) < 30:
        raise ValueError("at least 30 valid daily closes are required")
    if (close <= 0).any():
        raise ValueError("closing prices must be positive")
    return close.astype(float)


def calculate_factors(history: pd.DataFrame) -> dict[str, float | bool]:
    """Calculate interpretable factors from one ticker's daily OHLCV history."""
    close = _close_series(history)
    returns = close.pct_change().dropna()
    last = float(close.iloc[-1])
    ret20 = last / float(close.iloc[-21]) - 1 if len(close) >= 21 else np.nan
    ret60 = last / float(close.iloc[-61]) - 1 if len(close) >= 61 else np.nan
    ret120 = last / float(close.iloc[-121]) - 1 if len(close) >= 121 else np.nan
    sma20 = float(close.tail(20).mean())
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else float(close.mean())
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else float(close.mean())
    vol20 = float(returns.tail(20).std(ddof=1) * np.sqrt(252)) if len(returns) >= 20 else np.nan
    peak120 = float(close.tail(min(120, len(close))).max())
    drawdown120 = last / peak120 - 1
    volume_ratio = np.nan
    if "Volume" in history.columns:
        volume = pd.to_numeric(history["Volume"], errors="coerce").dropna()
        if len(volume) >= 60:
            mean20 = float(volume.tail(20).mean())
            mean60 = float(volume.tail(60).mean())
            if mean60 > 0:
                volume_ratio = mean20 / mean60
    return {
        "price": last,
        "return_20d": ret20,
        "return_60d": ret60,
        "return_120d": ret120,
        "volatility_20d_ann": vol20,
        "drawdown_120d": drawdown120,
        "volume_ratio_20d_60d": volume_ratio,
        "above_sma20": last > sma20,
        "above_sma50": last > sma50,
        "above_sma200": last > sma200,
    }


def _percentile_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(50.0, index=values.index)
    ranks = values.rank(pct=True, method="average")
    if not higher_is_better:
        ranks = 1 - ranks
    return ranks * 100.0


def rank_universe(factors: pd.DataFrame) -> pd.DataFrame:
    """Score cross-sectional factors and apply explicit risk gates.

    Weights: 20d momentum 20%, 60d momentum 25%, 120d momentum 25%,
    trend confirmation 20%, and lower realized volatility 10%.
    Missing medium/long lookbacks receive neutral scores and should be reviewed.
    """
    if factors.empty:
        return factors.copy()
    required = {"ticker", "return_20d", "return_60d", "return_120d",
                "volatility_20d_ann", "drawdown_120d", "above_sma50"}
    missing = required - set(factors.columns)
    if missing:
        raise ValueError(f"missing factor columns: {sorted(missing)}")

    out = factors.copy()
    out["score_mom20"] = _percentile_score(out["return_20d"])
    out["score_mom60"] = _percentile_score(out["return_60d"])
    out["score_mom120"] = _percentile_score(out["return_120d"])
    out["score_low_vol"] = _percentile_score(out["volatility_20d_ann"], higher_is_better=False)
    trend = out["above_sma50"].fillna(False).astype(bool).astype(float) * 100.0
    if "above_sma200" in out:
        trend = (trend + out["above_sma200"].fillna(False).astype(bool).astype(float) * 100.0) / 2
    out["score_trend"] = trend
    out["composite_score"] = (
        out["score_mom20"] * 0.20
        + out["score_mom60"] * 0.25
        + out["score_mom120"] * 0.25
        + out["score_trend"] * 0.20
        + out["score_low_vol"] * 0.10
    ).round(2)

    out["risk_pass"] = (
        pd.to_numeric(out["drawdown_120d"], errors="coerce").ge(-0.35)
        & pd.to_numeric(out["volatility_20d_ann"], errors="coerce").le(1.00)
        & out["above_sma50"].fillna(False).astype(bool)
        & pd.to_numeric(out["return_20d"], errors="coerce").gt(-0.15)
    )
    out["recommendation"] = np.where(out["risk_pass"], "REVIEW", "RISK_FILTERED")
    out = out.sort_values(["risk_pass", "composite_score"], ascending=[False, False])
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def scan_tickers(tickers: Iterable[str], period: str = "1y") -> pd.DataFrame:
    """Download daily data with yfinance and rank available tickers."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    rows: list[dict] = []
    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()
        if not ticker:
            continue
        try:
            history = yf.download(
                ticker, period=period, interval="1d", auto_adjust=True,
                progress=False, threads=False
            )
            if history is None or history.empty:
                LOGGER.warning("%s: no data returned; skipped", ticker)
                continue
            # Recent yfinance versions may return a MultiIndex even for one ticker.
            if isinstance(history.columns, pd.MultiIndex):
                history.columns = history.columns.get_level_values(0)
            factors = calculate_factors(history)
            rows.append({"ticker": ticker, **factors})
        except Exception as exc:  # isolate individual data failures
            LOGGER.warning("%s: skipped (%s)", ticker, exc)
    if not rows:
        raise RuntimeError("No ticker produced usable data; check network and symbols")
    return rank_universe(pd.DataFrame(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily multi-factor stock screener (no trading)")
    parser.add_argument("--tickers", default=os.getenv("TICKERS", ",".join(DEFAULT_TICKERS)),
                        help="comma-separated ticker symbols")
    parser.add_argument("--period", default=os.getenv("DATA_PERIOD", "1y"),
                        help="yfinance period, e.g. 6mo, 1y, 2y")
    parser.add_argument("--output", default="reports/quant_rankings.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = scan_tickers(args.tickers.split(","), period=args.period)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result[["rank", "ticker", "composite_score", "risk_pass", "recommendation"]].to_string(index=False))
    print(f"Saved {len(result)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
