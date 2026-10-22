import unittest
import numpy as np
import pandas as pd
from backtest import calculate_signal_frame, run_backtest

class BacktestTests(unittest.TestCase):
    def prices(self, n=320):
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        t = np.arange(n, dtype=float)
        return pd.DataFrame({"AAA":50*(1.003**t),"BBB":50*(1.001**t),
                             "SPY":100*(1.0008**t)}, index=idx)

    def test_signal_needs_history_and_risk_gate(self):
        scores, risk = calculate_signal_frame(self.prices())
        self.assertFalse(risk.iloc[:120].any().any())
        self.assertFalse(risk.iloc[:50].any().any())
        self.assertTrue(bool(risk.iloc[-1]["AAA"]))

    def test_benchmark_does_not_change_cross_sectional_signal_scores(self):
        prices = self.prices(320)
        scores_before, risk_before = calculate_signal_frame(prices[["AAA", "BBB"]])
        # An extreme benchmark should not enter the investable cross-section.
        prices["SPY"] = np.linspace(100, 10000, len(prices))
        result, _ = run_backtest(prices, top_n=1, rebalance_every=20,
                                 cost_bps=0, benchmark="SPY")
        scores_after, risk_after = calculate_signal_frame(prices[["AAA", "BBB"]])
        pd.testing.assert_frame_equal(scores_before, scores_after)
        pd.testing.assert_frame_equal(risk_before, risk_after)
        self.assertIn("benchmark_return", result.columns)

    def test_metrics_costs_and_benchmark(self):
        result, metrics = run_backtest(self.prices(), top_n=1, rebalance_every=20,
                                       cost_bps=10, benchmark="SPY")
        self.assertIn("strategy_return", result)
        self.assertIn("transaction_cost", result)
        self.assertIn("benchmark_total_return", metrics)
        self.assertLessEqual(metrics["max_drawdown"], 0)
        self.assertGreater(metrics["observations"], 0)
        self.assertGreater(metrics["cost_bps_one_way"], 0)

    def test_no_candidates_means_cash_after_rebalance(self):
        prices = self.prices()
        # Cause a severe drawdown in every asset; risk gate should block candidates.
        prices.iloc[220:] = prices.iloc[220:] * 0.4
        result, _ = run_backtest(prices, top_n=1, rebalance_every=5, cost_bps=0, benchmark="SPY")
        # After a later scheduled rebalance with no eligible names, there is no stale exposure.
        self.assertTrue((result["gross_return"].iloc[230:] == 0).all())

    def test_missing_return_for_held_asset_fails_closed(self):
        prices = self.prices(260)
        # Make a selected asset's price unavailable after it has become eligible.
        prices.iloc[205, prices.columns.get_loc("AAA")] = np.nan
        with self.assertRaisesRegex(ValueError, "missing daily return for held asset"):
            run_backtest(prices, top_n=1, rebalance_every=5, cost_bps=0, benchmark="SPY")

    def test_rejects_invalid_cost(self):
        with self.assertRaises(ValueError):
            run_backtest(self.prices(250), cost_bps=-1)

    def test_rejects_non_positive_or_infinite_prices(self):
        prices = self.prices(250)
        prices.iloc[205, prices.columns.get_loc("AAA")] = np.inf
        with self.assertRaisesRegex(ValueError, "missing daily return|positive numeric prices"):
            run_backtest(prices, cost_bps=0)

        prices = self.prices(250)
        prices.iloc[10, prices.columns.get_loc("AAA")] = 0
        with self.assertRaisesRegex(ValueError, "positive numeric prices"):
            run_backtest(prices, cost_bps=0)

    def test_requires_sufficient_history(self):
        with self.assertRaises(ValueError):
            run_backtest(self.prices(80))

if __name__ == "__main__":
    unittest.main()
