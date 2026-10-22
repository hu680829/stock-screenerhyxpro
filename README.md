# Stock Screener HYX Pro

A transparent research-oriented daily multi-factor stock screener. It downloads daily adjusted prices, computes momentum/trend/volatility/drawdown factors, applies risk gates, and writes a CSV ranking.

## Features
- Momentum: 20, 60, and 120 trading-day returns.
- Trend: price versus SMA50 and SMA200. A long average is unavailable until enough bars exist; shorter data is never substituted.
- Risk: 20-day annualized realized volatility and 120-day peak drawdown.
- Optional volume context: average volume over 20 days divided by average volume over 60 days.
- Weighted cross-sectional score, factor coverage, and a separate risk filter. Candidates need at least 70% factor-weight coverage to pass the risk gate; incomplete-history names are not promoted on a partial score.

## Run
Python 3.10+ recommended.

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python quant_engine.py --tickers AAPL,MSFT,NVDA,SPY --period 1y

## Score weights
- 20-day momentum: 20%
- 60-day momentum: 25%
- 120-day momentum: 25%
- Trend confirmation: 20%
- Lower realized volatility: 10%

Percentiles are calculated across symbols available in the same cross-section. When fewer than two valid values exist, the percentile is missing rather than an arbitrary 50. Composite scores are renormalized over available factors and report factor_coverage. Review low-coverage scores carefully. The score is a relative ranking, not a probability of profit.

Risk gate: price must be above SMA50, 20-day return above -15%, 120-day drawdown no worse than -35%, and annualized 20-day volatility no more than 100%. Missing required risk inputs fail closed. The same 70% factor-coverage floor is enforced in the historical backtest so the tested selection rules match the screener.

## Unified historical backtest

backtest.py now uses the screener's factor definitions and weights: 20/60/120-day momentum, trend confirmation using available SMA50/SMA200, 20-day realized volatility, and the same drawdown/volatility/SMA50/20-day-return risk gate. At each scheduled rebalance, it selects the top eligible names. Signals use information available at close t; positions begin earning returns on the next bar, and turnover costs are charged on that same next-bar interval.

Example:

    import yfinance as yf
    from backtest import run_backtest

    symbols = ["AAPL", "MSFT", "NVDA", "SPY"]
    raw = yf.download(symbols, period="5y", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no historical data")
    prices = raw["Close"] if hasattr(raw.columns, "levels") else raw[["Close"]].rename(columns={"Close": symbols[0]})
    daily, metrics = run_backtest(prices, top_n=3, rebalance_every=21, cost_bps=10, benchmark="SPY")
    print(metrics)
    daily.to_csv("reports/backtest_daily_returns.csv")

This is a simplified research backtest, not proof of a profitable edge. It does not yet correct for delisted securities, point-in-time universe membership, taxes, market impact, or all corporate-action/data issues. Use multiple market regimes, realistic costs, and out-of-sample tests before drawing conclusions.

## Reusable portfolio-risk calculations

`portfolio_risk.py` contains network-free, unit-tested helpers for FX-adjusted portfolio returns, normalized current weights, rebalance drift in percentage points, and proportional buy-budget caps. Missing holdings and invalid inputs fail closed. These are reusable calculation primitives; the daily screener currently does not fetch a live personal portfolio or send portfolio alerts through this module.

## Optional Qlib-inspired research layer

See research/README.md and requirements-research.txt. Optional research dependencies are not installed by the daily workflow, and no Qlib model is used for production rankings.

## Limitations
This is a screening/research tool, not financial advice or an execution bot. It places no trades and does not promise returns. Yahoo Finance data can be delayed, adjusted, missing, or rate-limited; validate source data. Historical performance does not predict future returns.
