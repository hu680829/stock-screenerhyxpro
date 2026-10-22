# Optional Qlib-inspired research layer

This directory contains research utilities that are deliberately separate from the daily screener.

## What is added

- `evaluation.py` calculates date-by-date Spearman rank IC, rank-IC information ratio, top/bottom quintile forward-return spread, and positive-spread frequency for cross-sectional factors.
- The evaluator is Qlib-inspired in its emphasis on repeatable factor/model evaluation, but it does not import Qlib and can run with the core dependencies.
- `requirements-research.txt` at the repository root declares optional `pyqlib` and `lightgbm` dependencies for a separate Qlib experiment environment. The daily GitHub Actions workflow does not install these packages.

## Evaluate a factor

Prepare a point-in-time table with one row per signal date and ticker and columns `date`, `ticker`, `factor`, and `forward_return`. The forward return must begin after the signal timestamp. Then:

```python
import pandas as pd
from research.evaluation import evaluate_factor

observations = pd.read_csv("factor_observations.csv")
daily, metrics = evaluate_factor(observations)
daily.to_csv("factor_daily_diagnostics.csv", index=False)
print(metrics)
```

The evaluator rejects duplicate date/ticker rows and skips dates with fewer than five valid securities by default. Its output is diagnostic, not a forecast or proof of profitability. Factor construction, universe membership, corporate actions, and forward-return alignment must be checked separately.

## Qlib / Alpha158 / LightGBM experiment

See [QLIB_ALPHA158_GUIDE.md](QLIB_ALPHA158_GUIDE.md) and [train_qlib_alpha158.py](train_qlib_alpha158.py). The script expects a prepared Qlib US provider dataset, trains Alpha158 + LightGBM on chronological train/validation segments, and writes a cost-adjusted top-K out-of-sample report. The optional experiment is not run in the daily workflow and does not change production rankings.

## Qlib / Alpha158 / LightGBM next step

The optional dependencies are only an environment declaration; this repository does **not** yet initialize Qlib, prepare Qlib-format market data, train a LightGBM model, or use model predictions in the production screener. Follow the official Qlib data-preparation and workflow instructions before attempting a benchmark. Keep train/validation/test periods chronologically separated, reserve a genuinely untouched test period, include transaction costs, and compare against a benchmark. Do not promote a model into the daily scan unless it passes reproducible tests and out-of-sample evaluation.

Official project: https://github.com/microsoft/qlib

## Which Microsoft components are actually being used?

Qlib is a quantitative research platform, not a single stock-picking large language model. This repository currently uses its Alpha158 feature handler and LightGBM model API in an optional experiment. The useful Qlib practices adopted here are chronological train/validation/test splits, cross-sectional signal evaluation (rank IC), portfolio-level evaluation, transaction-cost sensitivity, and reproducible research outputs.

Microsoft's separate RD-Agent-Quant project focuses on agent-driven factor mining and model/data iteration. It is not automatically included by installing Qlib, and blindly allowing an agent to rewrite factors without strict out-of-sample gates would risk overfitting. Treat RD-Agent integration as a later isolated experiment: candidate factors/models must pass walk-forward tests, multiple cost assumptions, benchmark comparisons, and reproducibility checks before promotion. No agent or LLM currently changes production rankings.
