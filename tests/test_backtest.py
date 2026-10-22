import unittest

import numpy as np
import pandas as pd

from backtest import run_backtest


class BacktestTests(unittest.TestCase):
    def test_returns_metrics_and_costs(self):
        idx = pd.date_range("2020-01-01", periods=260, freq="B")
        t = np.arange(len(idx), dtype=float)
        prices = pd.DataFrame({
            "AAA": 50 * (1.003 ** t),
            "BBB": 50 * (1.001 ** t),
            "SPY": 100 * (1.0008 ** t),
        }, index=idx)
        result, metrics = run_backtest(
            prices, lookback=60, top_n=1, rebalance_every=20,
            cost_bps=10, benchmark="SPY"
        )
        self.assertIn("strategy_return", result.columns)
        self.assertIn("benchmark_total_return", metrics)
        self.assertLessEqual(metrics["max_drawdown"], 0.0)
        self.assertGreater(metrics["observations"], 0)
        self.assertGreater(metrics["cost_bps_one_way"], 0)

    def test_rejects_invalid_cost(self):
        prices = pd.DataFrame({"AAA": [10, 11, 12, 13]})
        with self.assertRaises(ValueError):
            run_backtest(prices, lookback=2, cost_bps=-1)

    def test_requires_sufficient_history(self):
        prices = pd.DataFrame({"AAA": [10, 11, 12, 13]})
        with self.assertRaises(ValueError):
            run_backtest(prices, lookback=20)


if __name__ == "__main__":
    unittest.main()
