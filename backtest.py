"""Walk-forward backtest using the screener's multi-factor signal definitions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from quant_engine import MIN_FACTOR_COVERAGE

WEIGHTS = {"mom20": .20, "mom60": .25, "mom120": .25, "trend": .20, "low_vol": .10}

def _pct(row: pd.Series, higher: bool = True) -> pd.Series:
    row = pd.to_numeric(row, errors="coerce")
    valid = row.notna()
    if valid.sum() <= 1:
        return pd.Series(np.nan, index=row.index, dtype=float)
    ranks = row.rank(pct=True, method="average")
    if not higher:
        ranks = 1 - ranks
    return ranks.where(valid, np.nan) * 100.0

def calculate_signal_frame(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate close-t scores and risk gate from trailing data only."""
    px = prices.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).copy()
    px = px.loc[~px.index.duplicated(keep="last")].sort_index().dropna(axis=1, how="all")
    if px.empty:
        raise ValueError("prices must contain at least one usable ticker")
    if (px.dropna(how="all") <= 0).any().any():
        raise ValueError("prices must contain positive numeric prices")
    ret = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    mom20, mom60, mom120 = (px.div(px.shift(n)).sub(1).replace([np.inf, -np.inf], np.nan) for n in (20, 60, 120))
    vol20 = ret.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    sma50 = px.rolling(50, min_periods=50).mean()
    sma200 = px.rolling(200, min_periods=200).mean()
    peak120 = px.rolling(120, min_periods=1).max()
    dd120 = px.div(peak120).sub(1)
    scores = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    risk = pd.DataFrame(False, index=px.index, columns=px.columns)
    for i in range(len(px)):
        parts = {
            "mom20": _pct(mom20.iloc[i]),
            "mom60": _pct(mom60.iloc[i]),
            "mom120": _pct(mom120.iloc[i]),
            "low_vol": _pct(vol20.iloc[i], higher=False),
        }
        above50 = px.iloc[i].gt(sma50.iloc[i]).where(sma50.iloc[i].notna())
        above200 = px.iloc[i].gt(sma200.iloc[i]).where(sma200.iloc[i].notna())
        trend_raw = pd.concat([above50.astype(float) * 100, above200.astype(float) * 100], axis=1)
        parts["trend"] = trend_raw.mean(axis=1, skipna=True).where(trend_raw.notna().any(axis=1))
        numerator = sum(parts[k].fillna(0) * w for k, w in WEIGHTS.items())
        available = sum(parts[k].notna().astype(float) * w for k, w in WEIGHTS.items())
        coverage = available / sum(WEIGHTS.values())
        scores.iloc[i] = (numerator / available.replace(0, np.nan)).round(2)
        risk.iloc[i] = (coverage.ge(MIN_FACTOR_COVERAGE) & coverage.notna()
                        & dd120.iloc[i].ge(-.35) & dd120.iloc[i].notna()
                        & vol20.iloc[i].le(1.0) & vol20.iloc[i].notna()
                        & px.iloc[i].gt(sma50.iloc[i]) & sma50.iloc[i].notna()
                        & mom20.iloc[i].gt(-.15) & mom20.iloc[i].notna()
                        & mom120.iloc[i].notna() & scores.iloc[i].notna()).fillna(False)
    return scores, risk

def run_backtest(prices: pd.DataFrame, lookback: int | None = None, top_n: int = 3,
                 rebalance_every: int = 21, cost_bps: float = 10.0,
                 benchmark: str = "SPY") -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest same factor weights/risk gate as screener; lookback kept for API compatibility."""
    if top_n < 1 or rebalance_every < 1:
        raise ValueError("top_n >= 1 and rebalance_every >= 1 required")
    if lookback is not None and lookback < 2:
        raise ValueError("lookback, when supplied, must be >= 2")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and non-negative")
    if prices is None or prices.empty:
        raise ValueError("prices must contain at least one ticker column")
    px = prices.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).copy()
    px = px.loc[~px.index.duplicated(keep="last")].sort_index().dropna(axis=1, how="all")
    if px.empty:
        raise ValueError("prices must contain at least one usable ticker")
    if (px.dropna(how="all") <= 0).any().any():
        raise ValueError("prices must contain positive numeric prices")
    scores, risk = calculate_signal_frame(px)
    assets = [c for c in px.columns if c != benchmark]
    targets = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    scheduled = pd.Series(False, index=px.index)
    for i in range(200, len(px), rebalance_every):
        scheduled.iloc[i] = True
        eligible = scores.iloc[i].reindex(assets)
        eligible = eligible[risk.iloc[i].reindex(assets).fillna(False)].dropna()
        selected = eligible.sort_values(ascending=False, kind="mergesort").head(top_n)
        if len(selected):
            targets.loc[px.index[i], selected.index] = 1.0 / len(selected)
    holdings = targets.copy()
    last = pd.Series(0.0, index=px.columns)
    for i in range(len(holdings)):
        if scheduled.iloc[i]:
            last = targets.iloc[i].copy()  # zero eligible candidates means cash
        holdings.iloc[i] = last
    held = holdings.shift(1).fillna(0.0)
    daily_returns = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    # Fail closed instead of treating a missing held-asset return as a zero return.
    missing_held_returns = held.gt(0) & daily_returns.isna()
    if missing_held_returns.any().any():
        row_pos, col_pos = np.argwhere(missing_held_returns.to_numpy())[0]
        missing_date = missing_held_returns.index[row_pos]
        missing_ticker = missing_held_returns.columns[col_pos]
        raise ValueError(
            f"missing daily return for held asset {missing_ticker!r} on {missing_date}; "
            "repair/align the price data before backtesting"
        )
    gross = (held * daily_returns.fillna(0.0)).sum(axis=1)
    turnover = holdings.diff().abs().sum(axis=1).fillna(0.0)
    if len(turnover):
        turnover.iloc[0] = float(holdings.iloc[0].abs().sum())
    costs = turnover.shift(1).fillna(0.0) * cost_bps / 10000.0
    net = gross - costs
    output = pd.DataFrame({"strategy_return": net, "gross_return": gross,
                           "turnover": turnover, "transaction_cost": costs}, index=px.index)
    if benchmark in daily_returns:
        output["benchmark_return"] = daily_returns[benchmark].fillna(0.0)
    valid = output["strategy_return"].iloc[201:]
    if valid.empty:
        raise ValueError("not enough history to evaluate the 120-day multifactor signal and 200-day trend")
    equity = (1 + valid).cumprod()
    total = float(equity.iloc[-1] - 1)
    years = max(len(valid) / 252, 1 / 252)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0
    dd = equity / equity.cummax() - 1
    std = valid.std(ddof=1)
    vol = float(std * np.sqrt(252)) if len(valid) > 1 else 0.0
    sharpe = float(valid.mean() / std * np.sqrt(252)) if len(valid) > 1 and pd.notna(std) and std > 0 else 0.0
    metrics = {"total_return": total, "cagr": cagr, "max_drawdown": float(dd.min()),
               "annualized_volatility": vol, "sharpe_ratio_rf0": sharpe,
               "average_daily_turnover": float(output["turnover"].iloc[201:].mean()),
               "total_transaction_cost": float(output["transaction_cost"].iloc[201:].sum()),
               "cost_bps_one_way": float(cost_bps), "observations": float(len(valid))}
    if "benchmark_return" in output:
        b = output["benchmark_return"].iloc[201:]
        metrics["benchmark_total_return"] = float((1 + b).prod() - 1)
    return output, metrics
