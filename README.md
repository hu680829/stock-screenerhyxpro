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


## Experimental historical backtest

A simple long-only walk-forward **momentum prototype** is available in `backtest.py`.
It does not yet backtest the screener's full composite multi-factor score. It ranks
trailing lookback momentum, rebalances periodically, applies a one-row signal lag,
clears holdings to cash if a scheduled rebalance has no eligible candidates, and
deducts configurable turnover costs. Example:

```python
import pandas as pd
import yfinance as yf
from backtest import run_backtest

symbols = ["AAPL", "MSFT", "NVDA", "SPY"]
raw = yf.download(symbols, period="5y", auto_adjust=True, progress=False)
if raw.empty:
    raise RuntimeError("Yahoo Finance returned no historical data")
if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"]
else:
    # Fallback for single-ticker downloads that return flat columns.
    prices = raw[["Close"]].rename(columns={"Close": symbols[0]})
returns, metrics = run_backtest(prices, lookback=120, top_n=3,
                                 rebalance_every=21, cost_bps=10,
                                 benchmark="SPY")
print(metrics)
returns.to_csv("reports/backtest_daily_returns.csv")
```

**Limitations:** this is an initial research harness, not proof of an edge. It does
not yet model delisted securities, point-in-time universe membership, taxes, market
impact, or survivorship bias. Validate across multiple market regimes and compare
against buy-and-hold before relying on any result. Historical performance does not
predict future returns.
