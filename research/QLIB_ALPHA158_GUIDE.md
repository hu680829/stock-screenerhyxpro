# Optional Qlib / Alpha158 / LightGBM research

This experiment is separate from the production screener. It does not place orders and does not download market data itself.

## 1. Prepare Qlib data

Use the official Microsoft Qlib project and its documented data preparation workflow to obtain a US-market provider dataset. The provider directory must use Qlib's expected provider format; raw Yahoo Finance CSV files are not a Qlib provider. Keep data outside Git and do not commit large market-data files.

- Official project: https://github.com/microsoft/qlib
- Install optional dependencies in a dedicated environment: python -m pip install -r requirements-research.txt
- Confirm the provider initializes in Qlib before training.
- The script requests Qlib's Alpha158 handler and a LightGBM regression model; it does not build or download the provider dataset.

## 2. Chronological train / validation / test

Replace the provider path and dates with dates supported by your dataset:

    python research/train_qlib_alpha158.py --provider-uri /path/to/qlib_us_data --train-start 2010-01-01 --train-end 2018-12-31 --valid-start 2019-01-01 --valid-end 2020-12-31 --test-start 2021-01-01 --test-end 2025-12-31 --top-k 10 --cost-bps 10

The segments must be chronological and non-overlapping. Validation is for model selection/early stopping; the later test period is reserved for out-of-sample reporting. Do not repeatedly tune against the test period.

Qlib's current Alpha158 default label is `Ref($close, -2) / Ref($close, -1) - 1` (`LABEL0`): it measures the close-to-close return from the next session's close to the following session's close, indexed to the signal date. This is not the same as the signal-day close-to-next-day close return. Verify the installed Qlib version's handler definition and provider's trading calendar before interpreting results. The script requires strictly increasing date boundaries because Qlib segment endpoints are inclusive.

Outputs:
- oos_metrics.json: test-period return, CAGR, drawdown, volatility, Sharpe, equal-weight-universe comparison, turnover/costs, mean rank IC, IC IR, and positive-IC fraction.
- oos_daily_returns.csv: gross/net strategy returns, transaction costs, turnover, and equal-weight-universe returns.
- oos_cost_sensitivity.csv: the same OOS predictions evaluated at 0, 5, 10, 20, and 30 bps (plus the configured cost, if different); this is a sensitivity analysis, not additional independent tests.
- oos_predictions_and_labels.csv: model predictions and future labels used for evaluation.

## 3. Limitations and acceptance criteria

This is a simple daily top-K long-only simulation, not an execution simulator. Confirm the installed Qlib version's Alpha158 label convention and that the label corresponds to the intended forward holding period. The equal-weight comparison is the average future return of that day's valid universe, not an investable ETF benchmark. Costs are simplified and do not model spread, market impact, taxes, or liquidity. Survivorship and corporate-action bias may remain.

Do not promote the model based only on in-sample performance. Inspect multiple market regimes, run higher-cost sensitivity tests, compare with an investable benchmark, and review data quality and universe construction. The model is not wired into quant_engine.py or the daily workflow.
