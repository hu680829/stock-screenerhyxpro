"""Optional Qlib Alpha158 + LightGBM walk-forward experiment.

Requires a prepared Qlib provider dataset. No network data download or trading occurs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def _json_safe_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    """Convert NaN/inf diagnostics to JSON null for strict JSON consumers."""
    return {key: (float(value) if np.isfinite(value) else None)
            for key, value in metrics.items()}


def _daily_metrics(daily: pd.DataFrame, cost_bps: float) -> dict[str, float]:
    returns = daily["net_return"].astype(float)
    if returns.empty:
        raise ValueError("the out-of-sample segment produced no usable daily returns")
    equity = (1.0 + returns).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    years = max(len(returns) / 252.0, 1.0 / 252.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0
    drawdown = equity / equity.cummax() - 1.0
    std = returns.std(ddof=1)
    sharpe = float(returns.mean() / std * np.sqrt(252)) if len(returns) > 1 and pd.notna(std) and std > 0 else 0.0
    benchmark = daily["universe_equal_weight_return"].astype(float)
    return {
        "oos_observations": float(len(returns)),
        "oos_total_return": total,
        "oos_cagr": cagr,
        "oos_max_drawdown": float(drawdown.min()),
        "oos_annualized_volatility": float(std * np.sqrt(252)) if len(returns) > 1 else 0.0,
        "oos_sharpe_rf0": sharpe,
        "benchmark_equal_weight_total_return": float((1.0 + benchmark).prod() - 1.0),
        "total_turnover": float(daily["turnover"].sum()),
        "total_transaction_cost": float(daily["transaction_cost"].sum()),
        "cost_bps_per_traded_notional": float(cost_bps),
    }



def _build_daily_portfolio(observations: pd.DataFrame, top_k: int,
                           cost_bps: float) -> pd.DataFrame:
    """Build a daily equal-weight top-K portfolio from point-in-time predictions.

    forward_return must represent the return earned after each signal date.
    Turnover is the sum of absolute weight changes; the initial portfolio starts
    in cash, so the first day's selected weights also incur transaction costs.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    required = {"date", "ticker", "factor", "forward_return"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    data = observations.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["ticker"] = data["ticker"].astype(str)
    for col in ("factor", "forward_return"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["date", "ticker", "factor", "forward_return"])
    if data.empty:
        raise ValueError("No usable prediction/label rows")
    if data.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate date/ticker rows in predictions")

    daily_rows = []
    previous: set[str] = set()
    for date, group in data.groupby("date", sort=True):
        group = group.sort_values("factor", ascending=False, kind="mergesort")
        selected = group.head(min(top_k, len(group)))
        current = set(selected["ticker"])
        if not current:
            continue
        all_names = previous | current
        turnover = sum(
            abs((1.0 / len(current) if name in current else 0.0)
                - (1.0 / len(previous) if name in previous else 0.0))
            for name in all_names
        )
        gross = float(selected["forward_return"].mean())
        cost = turnover * cost_bps / 10000.0
        daily_rows.append({
            "date": date, "gross_return": gross, "net_return": gross - cost,
            "transaction_cost": cost, "turnover": turnover,
            "universe_equal_weight_return": float(group["forward_return"].mean()),
            "n_names": int(len(group)), "n_selected": int(len(selected)),
        })
        previous = current
    if not daily_rows:
        raise ValueError("No usable daily portfolio returns")
    return pd.DataFrame(daily_rows).set_index("date").sort_index()



def _cost_sensitivity(observations: pd.DataFrame, top_k: int,
                      cost_bps_values: list[float]) -> pd.DataFrame:
    """Re-score the same OOS predictions under several transaction-cost assumptions."""
    if not cost_bps_values:
        raise ValueError("at least one cost scenario is required")
    scenarios = sorted({float(value) for value in cost_bps_values})
    if any(not np.isfinite(value) or value < 0 for value in scenarios):
        raise ValueError("cost scenarios must be finite and non-negative")
    rows = []
    for bps in scenarios:
        daily = _build_daily_portfolio(observations, top_k=top_k, cost_bps=bps)
        metrics = _daily_metrics(daily, bps)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("cost_bps_per_traded_notional").reset_index(drop=True)


def run_experiment(provider_uri: str, train_start: str, train_end: str,
                   valid_start: str, valid_end: str, test_start: str, test_end: str,
                   output_dir: str = "reports/qlib_oos", top_k: int = 10,
                   cost_bps: float = 10.0) -> dict[str, float]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    # Parse dates instead of comparing strings. Qlib segment endpoints are
    # inclusive, so equal boundaries can leak observations across segments.
    try:
        parsed_dates = [pd.Timestamp(value) for value in
                        (train_start, train_end, valid_start, valid_end, test_start, test_end)]
    except (TypeError, ValueError) as exc:
        raise ValueError("all segment boundaries must be valid dates") from exc
    if any(pd.isna(value) for value in parsed_dates):
        raise ValueError("all segment boundaries must be valid dates")
    if not all(left < right for left, right in zip(parsed_dates, parsed_dates[1:])):
        raise ValueError(
            "date segments must be strictly chronological with gaps between inclusive "
            "train/validation/test boundaries"
        )
    try:
        import qlib
        from qlib.constant import REG_US
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies with: pip install -r requirements-research.txt") from exc

    qlib.init(provider_uri=provider_uri, region=REG_US, auto_mount=False)
    handler = {
        "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
        "kwargs": {"start_time": train_start, "end_time": test_end,
                   "fit_start_time": train_start, "fit_end_time": train_end,
                   "instruments": "all"},
    }
    dataset = DatasetH(handler=handler, segments={
        "train": (train_start, train_end),
        "valid": (valid_start, valid_end),
        "test": (test_start, test_end),
    })
    model = LGBModel(loss="mse", learning_rate=0.05, num_leaves=31,
                     max_depth=-1, num_boost_round=500,
                     early_stopping_rounds=50, lambda_l1=0.1,
                     lambda_l2=1.0, num_threads=4, seed=42,
                     feature_fraction_seed=42, bagging_seed=42,
                     deterministic=True, force_col_wise=True)
    model.fit(dataset)
    predictions = model.predict(dataset, segment="test")
    labels = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_I)
    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] == 0:
            raise ValueError("Qlib test segment returned no label column")
        labels = labels.iloc[:, 0]
    observations = pd.concat(
        [predictions.rename("factor"), labels.rename("forward_return")],
        axis=1, join="inner").dropna()
    if not isinstance(observations.index, pd.MultiIndex):
        raise ValueError("Expected Qlib prediction/label index to be (date, instrument) MultiIndex")
    observations.index = observations.index.set_names(["date", "ticker"])
    observations = observations.reset_index()
    observations["date"] = pd.to_datetime(observations["date"])
    observations["ticker"] = observations["ticker"].astype(str)
    for col in ("factor", "forward_return"):
        observations[col] = pd.to_numeric(observations[col], errors="coerce")
    observations = observations.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["date", "ticker", "factor", "forward_return"])
    if observations.empty:
        raise ValueError("No overlapping prediction/label rows in the OOS period")

    daily = _build_daily_portfolio(observations, top_k, cost_bps)
    metrics = _daily_metrics(daily, cost_bps)
    sensitivity_costs = sorted({0.0, 5.0, 10.0, 20.0, 30.0, float(cost_bps)})
    sensitivity = _cost_sensitivity(observations, top_k, sensitivity_costs)
    daily_ic = []
    for _, group in observations.groupby("date", sort=True):
        if len(group) >= 5:
            ic = group["factor"].rank(method="average").corr(
                group["forward_return"].rank(method="average"))
            daily_ic.append(float(ic) if pd.notna(ic) else np.nan)
    valid_ic = pd.Series(daily_ic).dropna()
    metrics["mean_rank_ic"] = float(valid_ic.mean()) if not valid_ic.empty else float("nan")
    metrics["rank_ic_ir"] = (float(valid_ic.mean()/valid_ic.std(ddof=1))
        if len(valid_ic) > 1 and valid_ic.std(ddof=1) > 0 else 0.0)
    metrics["positive_rank_ic_fraction"] = float((valid_ic > 0).mean()) if not valid_ic.empty else float("nan")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "oos_daily_returns.csv")
    sensitivity.to_csv(output / "oos_cost_sensitivity.csv", index=False)
    observations.to_csv(output / "oos_predictions_and_labels.csv", index=False)
    (output / "oos_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional Qlib Alpha158/LightGBM OOS evaluation")
    parser.add_argument("--provider-uri", required=True, help="Path to prepared Qlib US provider data")
    for name in ("train-start", "train-end", "valid-start", "valid-end", "test-start", "test-end"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output-dir", default="reports/qlib_oos")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    metrics = run_experiment(
        args.provider_uri, args.train_start, args.train_end, args.valid_start,
        args.valid_end, args.test_start, args.test_end, args.output_dir,
        args.top_k, args.cost_bps)
    print(json.dumps(_json_safe_metrics(metrics), indent=2, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
