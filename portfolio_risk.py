"""Pure portfolio-risk helpers shared by screening and reporting workflows.

All returns are decimal returns (0.05 means +5%). This module has no network
access and never places trades; callers must supply validated point-in-time data.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Iterable


def portfolio_snapshot(
    returns: Mapping[str, float],
    target_weights: Mapping[str, float],
    *,
    usd_tickers: Iterable[str] = (),
    usd_cnh_return: float = 0.0,
    rebalance_deadband_pp: float = 4.5,
) -> dict:
    """Calculate FX-adjusted portfolio return and normalized weight drift.

    Missing/invalid holdings fail closed rather than silently calculating a
    partial portfolio. Weight drift is reported in percentage points.
    """
    if not target_weights:
        raise ValueError("target_weights must not be empty")
    if not math.isfinite(float(usd_cnh_return)):
        raise ValueError("usd_cnh_return must be finite")
    if not math.isfinite(float(rebalance_deadband_pp)) or rebalance_deadband_pp < 0:
        raise ValueError("rebalance_deadband_pp must be finite and non-negative")

    weights = {}
    for ticker, weight in target_weights.items():
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid target weight for {ticker!r}")
        weights[ticker] = weight
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("target weights must sum to a positive value")

    missing = sorted(set(weights) - set(returns))
    if missing:
        raise ValueError(f"missing return data for holdings: {', '.join(missing)}")

    usd = set(usd_tickers)
    converted = {}
    for ticker in weights:
        value = float(returns[ticker])
        if not math.isfinite(value) or value <= -1.0:
            raise ValueError(f"invalid return for {ticker!r}")
        converted[ticker] = (
            (1.0 + value) * (1.0 + float(usd_cnh_return)) - 1.0
            if ticker in usd else value
        )

    normalized = {ticker: weight / weight_sum for ticker, weight in weights.items()}
    gross_value = sum(normalized[ticker] * (1.0 + converted[ticker]) for ticker in weights)
    if not math.isfinite(gross_value) or gross_value <= 0:
        raise ValueError("portfolio ending value must be positive and finite")

    current = {
        ticker: normalized[ticker] * (1.0 + converted[ticker]) / gross_value
        for ticker in weights
    }
    drift_pp = {ticker: (current[ticker] - normalized[ticker]) * 100.0 for ticker in weights}
    alerts = {
        ticker: drift for ticker, drift in drift_pp.items()
        if abs(drift) >= rebalance_deadband_pp
    }
    return {
        "portfolio_return": gross_value - 1.0,
        "target_weights": normalized,
        "current_weights": current,
        "drift_pp": drift_pp,
        "rebalance_alerts": alerts,
    }


def cap_buy_allocations(proposals: Mapping[str, float], max_allocate_pct: float) -> dict[str, float]:
    """Scale suggested buy allocations proportionally so their sum respects a budget."""
    budget = float(max_allocate_pct)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("max_allocate_pct must be finite and non-negative")
    amounts = {}
    for ticker, amount in proposals.items():
        amount = float(amount)
        if not math.isfinite(amount) or amount < 0:
            raise ValueError(f"invalid proposed allocation for {ticker!r}")
        amounts[ticker] = amount
    total = sum(amounts.values())
    if total <= budget or total == 0:
        return amounts
    scale = budget / total
    return {ticker: amount * scale for ticker, amount in amounts.items()}
