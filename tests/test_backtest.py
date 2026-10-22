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
        # A gap makes momentum unavailable on the scheduled rebalance at idx[10],
        # while prices recover afterward so stale exposure would create returns.
        prices = pd.DataFrame({
            "AAA": [10, 11, 12, 13, 14, 15, 16, 17, np.nan, 19, 20, 21],
            "BBB": [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, np.nan, 15, 16, 17],
        }, index=idx)
        result, _ = run_backtest(
            prices, lookback=2, top_n=1, rebalance_every=4,
            cost_bps=0, benchmark="SPY"
        )
        # The scheduled rebalance at idx[10] has no valid momentum scores.
        # Holdings are cleared at close idx[10], so idx[11] earns no exposure.
        self.assertEqual(result.loc[idx[11], "gross_return"], 0.0)

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
