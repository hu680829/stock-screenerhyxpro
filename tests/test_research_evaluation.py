import unittest

import pandas as pd

from research.evaluation import evaluate_factor


class FactorEvaluationTests(unittest.TestCase):
    def test_perfect_monotonic_factor_has_positive_ic_and_spread(self):
        rows = []
        for date in pd.date_range("2024-01-01", periods=3, freq="B"):
            for ticker, factor in zip("ABCDE", range(5)):
                rows.append({"date": date, "ticker": ticker, "factor": factor,
                             "forward_return": factor / 100.0})
        daily, metrics = evaluate_factor(pd.DataFrame(rows))
        self.assertEqual(len(daily), 3)
        self.assertAlmostEqual(metrics["mean_rank_ic"], 1.0)
        self.assertGreater(metrics["mean_top_bottom_spread"], 0)

    def test_rejects_duplicate_date_ticker(self):
        frame = pd.DataFrame([
            {"date": "2024-01-01", "ticker": "A", "factor": 1, "forward_return": .1},
            {"date": "2024-01-01", "ticker": "A", "factor": 2, "forward_return": .2},
        ])
        with self.assertRaises(ValueError):
            evaluate_factor(frame)

    def test_rejects_duplicate_after_date_and_ticker_normalization(self):
        frame = pd.DataFrame([
            {"date": "2024-01-01", "ticker": "A", "factor": 1, "forward_return": .1},
            {"date": pd.Timestamp("2024-01-01"), "ticker": " A ", "factor": 2, "forward_return": .2},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate date/ticker"):
            evaluate_factor(frame)

    def test_drops_invalid_dates_and_nonfinite_values(self):
        rows = [
            {"date": "2024-01-01", "ticker": ticker, "factor": i,
             "forward_return": i / 100.0}
            for i, ticker in enumerate("ABCDE")
        ]
        rows.append({"date": "not-a-date", "ticker": "Z", "factor": 9, "forward_return": .9})
        daily, metrics = evaluate_factor(pd.DataFrame(rows))
        self.assertEqual(len(daily), 1)
        self.assertEqual(metrics["mean_names_per_date"], 5.0)

    def test_requires_enough_cross_section(self):
        frame = pd.DataFrame([
            {"date": "2024-01-01", "ticker": "A", "factor": 1, "forward_return": .1},
            {"date": "2024-01-01", "ticker": "B", "factor": 2, "forward_return": .2},
        ])
        with self.assertRaises(ValueError):
            evaluate_factor(frame)


if __name__ == "__main__":
    unittest.main()
