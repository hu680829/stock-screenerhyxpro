"""Simple long-only walk-forward backtest for research, not a trading recommendation.

Input is a wide DataFrame of adjusted close prices (date index, ticker columns).
Signals use information available at the close and positions earn the *next* bar's
return. Rebalancing turnover incurs a configurable one-way cost in basis points.
This deliberately omits delisted securities and point-in-time fundamentals, so
results can still suffer survivorship and data-source bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    prices: pd.DataFrame,
    lookback: int = 120,
    top_n: int = 3,
    rebalance_every: int = 21,
    cost_bps: float = 10.0,
    benchmark: str = "SPY",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return daily strategy/benchmark returns and summary metrics.

    Momentum ranks are calculated from trailing lookback returns. A signal formed
    at close t is shifted one row before being applied to returns t->t+1.
    """
    if lookback < 2 or top_n < 1 or rebalance_every < 1:
        raise ValueError("lookback >= 2, top_n >= 1 and rebalance_every >= 1 required")
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if prices is None or prices.empty or prices.shape[1] < 1:
        raise ValueError("prices must contain at least one ticker column")

    px = prices.copy()
    px = px.apply(pd.to_numeric, errors="coerce")
    px = px.loc[~px.index.duplicated(keep="last")].sort_index()
    px = px.dropna(axis=1, how="all")
    if px.empty or (px.dropna(how="all") <= 0).any().any():
        raise ValueError("prices must contain positive numeric prices")
    daily_returns = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    momentum = px.div(px.shift(lookback)).sub(1).replace([np.inf, -np.inf], np.nan)

    # Weight target holdings at each rebalance close; no same-bar return capture.
    targets = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for i in range(lookback, len(px), rebalance_every):
        scores = momentum.iloc[i].dropna()
        scores = scores[scores.index != benchmark].sort_values(ascending=False)
        selected = scores.head(top_n)
        if len(selected):
            targets.iloc[i, targets.columns.get_indexer(selected.index)] = 1.0 / len(selected)

    # Carry the last rebalance weights forward, then shift to avoid look-ahead.
    holdings = targets.replace(0.0, np.nan).ffill().fillna(0.0)
    # A zero-weight asset at a rebalance must be explicitly removed; forward-fill
    # targets row-wise using a separate rebalance marker.
    rebalance_rows = targets.sum(axis=1) > 0
    holdings = targets.copy()
    last = pd.Series(0.0, index=px.columns)
    for i in range(len(holdings)):
        if rebalance_rows.iloc[i]:
            last = targets.iloc[i].copy()
        holdings.iloc[i] = last
    held = holdings.shift(1).fillna(0.0)

    gross = (held * daily_returns.fillna(0.0)).sum(axis=1)
    turnover = holdings.diff().abs().sum(axis=1).fillna(holdings.iloc[0].abs().sum())
    # A rebalance signal is formed at close t, so its transaction cost belongs
    # to the next return interval, alongside the shifted holdings.
    costs = turnover.shift(1).fillna(0.0) * (cost_bps / 10000.0)
    net = gross - costs
    output = pd.DataFrame({"strategy_return": net, "gross_return": gross, "turnover": turnover}, index=px.index)
    if benchmark in daily_returns.columns:
        output["benchmark_return"] = daily_returns[benchmark].fillna(0.0)

    valid = output["strategy_return"].iloc[lookback + 1:]
    if valid.empty:
        raise ValueError("not enough price history for the requested lookback")
    equity = (1.0 + valid).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = max(len(valid) / 252.0, 1 / 252.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0
    drawdown = equity / equity.cummax() - 1.0
    volatility = float(valid.std(ddof=1) * np.sqrt(252)) if len(valid) > 1 else 0.0
    sharpe = float(valid.mean() / valid.std(ddof=1) * np.sqrt(252)) if len(valid) > 1 and valid.std(ddof=1) > 0 else 0.0
    metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": float(drawdown.min()),
        "annualized_volatility": volatility,
        "sharpe_ratio_rf0": sharpe,
        "average_daily_turnover": float(output["turnover"].iloc[lookback + 1:].mean()),
        "cost_bps_one_way": float(cost_bps),
        "observations": float(len(valid)),
    }
    if "benchmark_return" in output:
        b = output["benchmark_return"].iloc[lookback + 1:]
        be = (1.0 + b).cumprod()
        metrics["benchmark_total_return"] = float(be.iloc[-1] - 1.0)
    return output, metrics
