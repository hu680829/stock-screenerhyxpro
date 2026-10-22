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

    def test_scheduled_rebalance_with_no_valid_candidates_clears_old_holdings(self):
        idx = pd.date_range("2020-01-01", periods=12, freq="B")
        # AAA is eligible at the first rebalance, then has a missing price at
        # the next scheduled rebalance so its momentum score is unavailable.
        prices = pd.DataFrame({
            "AAA": [10, 11, 12, 13, 14, 15, 16, np.nan, np.nan, np.nan, np.nan, np.nan],
            "BBB": [10, 10.5, 11, 11.5, 12, 12.5, 13, np.nan, np.nan, np.nan, np.nan, np.nan],
        }, index=idx)
        result, _ = run_backtest(
            prices, lookback=2, top_n=1, rebalance_every=3,
            cost_bps=0, benchmark="SPY"
        )
        # The second scheduled rebalance has no eligible momentum observations;
        # returns after that point must not keep earning stale AAA/BBB exposure.
        self.assertTrue((result.loc[idx[7]:, "gross_return"] == 0.0).all())

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
