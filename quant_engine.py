"""Transparent multi-factor stock screener with strict lookback handling."""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("quant_engine")
MIN_FACTOR_COVERAGE = 0.70
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

def calculate_factors(history: pd.DataFrame) -> dict:
    close = _close_series(history)
    returns = close.pct_change().dropna()
    last = float(close.iloc[-1])
    ret20 = last / float(close.iloc[-21]) - 1 if len(close) >= 21 else np.nan
    ret60 = last / float(close.iloc[-61]) - 1 if len(close) >= 61 else np.nan
    ret120 = last / float(close.iloc[-121]) - 1 if len(close) >= 121 else np.nan
    sma20 = float(close.tail(20).mean()) if len(close) >= 20 else np.nan
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else np.nan
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else np.nan
    vol20 = float(returns.tail(20).std(ddof=1) * np.sqrt(252)) if len(returns) >= 20 else np.nan
    peak120 = float(close.tail(min(120, len(close))).max())
    drawdown120 = last / peak120 - 1
    volume_ratio = np.nan
    if "Volume" in history.columns:
        volume = pd.to_numeric(history["Volume"], errors="coerce").dropna()
        if len(volume) >= 60:
            mean20, mean60 = float(volume.tail(20).mean()), float(volume.tail(60).mean())
            if mean60 > 0:
                volume_ratio = mean20 / mean60
    return {
        "price": last, "history_bars": int(len(close)),
        "return_20d": ret20, "return_60d": ret60, "return_120d": ret120,
        "volatility_20d_ann": vol20, "drawdown_120d": drawdown120,
        "volume_ratio_20d_60d": volume_ratio,
        "sma50_available": len(close) >= 50, "sma200_available": len(close) >= 200,
        "above_sma20": bool(last > sma20) if pd.notna(sma20) else np.nan,
        "above_sma50": bool(last > sma50) if pd.notna(sma50) else np.nan,
        "above_sma200": bool(last > sma200) if pd.notna(sma200) else np.nan,
    }

def _percentile_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    valid = values.notna()
    if valid.sum() <= 1:
        return pd.Series(np.nan, index=values.index, dtype=float)
    ranks = values.rank(pct=True, method="average")
    if not higher_is_better:
        ranks = 1 - ranks
    return (ranks * 100.0).where(valid, np.nan)

def rank_universe(factors: pd.DataFrame) -> pd.DataFrame:
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
    trend_cols = []
    for col in ("above_sma50", "above_sma200"):
        if col in out:
            vals = out[col].astype("boolean")
            trend_cols.append(vals.astype(float) * 100.0)
    trend = pd.concat(trend_cols, axis=1)
    out["score_trend"] = trend.mean(axis=1, skipna=True).where(trend.notna().any(axis=1))
    weights = {"score_mom20": .20, "score_mom60": .25, "score_mom120": .25,
               "score_trend": .20, "score_low_vol": .10}
    numerator = sum(out[k].fillna(0.0) * w for k, w in weights.items())
    available = sum(out[k].notna().astype(float) * w for k, w in weights.items())
    out["factor_coverage"] = available / sum(weights.values())
    out["composite_score"] = (numerator / available.replace(0, np.nan)).round(2)
    dd = pd.to_numeric(out["drawdown_120d"], errors="coerce")
    vol = pd.to_numeric(out["volatility_20d_ann"], errors="coerce")
    ret20 = pd.to_numeric(out["return_20d"], errors="coerce")
    ret120 = pd.to_numeric(out["return_120d"], errors="coerce")
    above50 = out["above_sma50"].astype("boolean").fillna(False).astype(bool)
    # Fail closed when the ranking is based on too few available factors.
    # This avoids short-history symbols ranking highly from a partial score.
    coverage = pd.to_numeric(out["factor_coverage"], errors="coerce")
    out["risk_pass"] = (coverage.ge(MIN_FACTOR_COVERAGE) & coverage.notna()
                        & dd.notna() & dd.ge(-.35) & vol.notna() & vol.le(1.0)
                        & above50 & ret20.notna() & ret20.gt(-.15)
                        & ret120.notna() & out["composite_score"].notna())
    out["recommendation"] = np.where(out["risk_pass"], "REVIEW", "RISK_FILTERED")
    out = out.sort_values(["risk_pass", "composite_score"], ascending=[False, False], na_position="last")
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)

def scan_tickers(tickers: Iterable[str], period: str = "1y") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
    rows = []
    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()
        if not ticker:
            continue
        try:
            history = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                                  progress=False, threads=False)
            if history is None or history.empty:
                LOGGER.warning("%s: no data returned; skipped", ticker)
                continue
            if isinstance(history.columns, pd.MultiIndex):
                history.columns = history.columns.get_level_values(0)
            rows.append({"ticker": ticker, **calculate_factors(history)})
        except Exception as exc:
            LOGGER.warning("%s: skipped (%s)", ticker, exc)
    if not rows:
        raise RuntimeError("No ticker produced usable data; check network and symbols")
    return rank_universe(pd.DataFrame(rows))

def main() -> int:
    parser = argparse.ArgumentParser(description="Daily multi-factor stock screener (no trading)")
    parser.add_argument("--tickers", default=os.getenv("TICKERS", ",".join(DEFAULT_TICKERS)))
    parser.add_argument("--period", default=os.getenv("DATA_PERIOD", "1y"))
    parser.add_argument("--output", default="reports/quant_rankings.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = scan_tickers(args.tickers.split(","), period=args.period)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result[["rank", "ticker", "composite_score", "factor_coverage", "risk_pass", "recommendation"]].to_string(index=False))
    print(f"Saved {len(result)} rows to {output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
