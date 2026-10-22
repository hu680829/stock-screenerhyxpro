"""Leakage-aware cross-sectional factor evaluation helpers.

These helpers are deliberately independent of Qlib so the lightweight screener
can evaluate candidate factors without importing the optional research stack.
Inputs must contain point-in-time factor values and forward returns computed
strictly after each signal timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_factor(
    observations: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    factor_col: str = "factor",
    forward_return_col: str = "forward_return",
    min_cross_section: int = 5,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate a cross-sectional factor using daily rank IC and top-bottom spread.

    Each row is one security at one signal date. ``forward_return`` must represent
    a future holding-period return, not a same-day return. Duplicate date/ticker
    observations are rejected to avoid silently overweighting securities.
    """
    required = {date_col, ticker_col, factor_col, forward_return_col}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if min_cross_section < 3:
        raise ValueError("min_cross_section must be at least 3")
    data = observations[list(required)].copy()
    if data.duplicated([date_col, ticker_col]).any():
        raise ValueError("duplicate date/ticker observations are not allowed")
    data[factor_col] = pd.to_numeric(data[factor_col], errors="coerce")
    data[forward_return_col] = pd.to_numeric(data[forward_return_col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[date_col, ticker_col, factor_col, forward_return_col]
    )

    rows: list[dict[str, object]] = []
    for date, group in data.groupby(date_col, sort=True):
        if len(group) < min_cross_section:
            continue
        ic = group[factor_col].rank(method="average").corr(\n            group[forward_return_col].rank(method="average")\n        )
        ranked = group.sort_values(factor_col, ascending=False, kind="mergesort")
        bucket_size = max(1, int(np.floor(len(ranked) * 0.2)))
        top = ranked.head(bucket_size)[forward_return_col].mean()
        bottom = ranked.tail(bucket_size)[forward_return_col].mean()
        rows.append({"date": date, "rank_ic": float(ic) if pd.notna(ic) else np.nan,
                     "top_bucket_return": float(top), "bottom_bucket_return": float(bottom),
                     "top_bottom_spread": float(top - bottom), "n_names": int(len(group))})

    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("no dates have enough valid cross-sectional observations")
    ic = daily["rank_ic"].dropna()
    spread = daily["top_bottom_spread"].dropna()
    metrics = {
        "evaluated_dates": float(len(daily)),
        "mean_rank_ic": float(ic.mean()) if not ic.empty else float("nan"),
        "rank_ic_ir": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) > 0 else 0.0,
        "mean_top_bottom_spread": float(spread.mean()) if not spread.empty else float("nan"),
        "positive_spread_fraction": float((spread > 0).mean()) if not spread.empty else float("nan"),
        "mean_names_per_date": float(daily["n_names"].mean()),
    }
    return daily, metrics
