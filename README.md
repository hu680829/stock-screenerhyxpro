# Stock Screener HYX Pro

A transparent, research-oriented daily multi-factor stock screener. It downloads daily adjusted prices, computes momentum/trend/volatility/drawdown factors, applies explicit risk gates, and writes a CSV ranking.

## What it does
- Momentum factors: 20, 60, and 120 trading-day returns.
- Trend confirmation: price above 50-day and 200-day simple moving averages.
- Risk context: 20-day annualized realized volatility and 120-day peak drawdown.
- Optional volume context: 20-day average volume divided by 60-day average volume.
- Weighted cross-sectional score and a separate risk filter.
- Isolates per-symbol data errors and records a CSV report.

## Run locally or in GitHub Actions
Python 3.10+ recommended.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python quant_engine.py --tickers AAPL,MSFT,NVDA,SPY --period 1y
```

The default output is `reports/quant_rankings.csv`. Override symbols with the `TICKERS` environment variable or `--tickers`.

## Score weights
- 20-day momentum: 20%
- 60-day momentum: 25%
- 120-day momentum: 25%
- Trend confirmation: 20%
- Lower realized volatility: 10%

The composite score is a relative ranking within the symbols downloaded for the same run. It is not a probability of profit. The risk filter marks a row `REVIEW` only when the price is above its 50-day average, 20-day return is above -15%, 120-day drawdown is no worse than -35%, and annualized 20-day volatility is at most 100%.

## Important limitations
- This is a screening/research tool, not financial advice and not an execution bot.
- It places no trades and does not promise returns.
- Yahoo Finance data can be delayed, adjusted, missing, or rate-limited; check source data before decisions.
- A score is not a backtest. Before live use, add transaction costs, survivorship-bias controls, point-in-time data, walk-forward testing, and benchmark comparisons.
