import unittest
import pandas as pd
from research.train_qlib_alpha158 import (
    run_experiment, _build_daily_portfolio, _cost_sensitivity, _json_safe_metrics,
    _validate_split_dates,
)


class QlibResearchTests(unittest.TestCase):
    def test_rejects_non_chronological_splits_before_importing_qlib(self):
        with self.assertRaisesRegex(ValueError, "chronological"):
            run_experiment(
                provider_uri="/unused",
                train_start="2020-01-01", train_end="2021-01-01",
                valid_start="2019-01-01", valid_end="2020-01-01",
                test_start="2022-01-01", test_end="2023-01-01",
            )

    def test_rejects_equal_inclusive_segment_boundaries(self):
        with self.assertRaisesRegex(ValueError, "strictly chronological"):
            _validate_split_dates("2010-01-01", "2018-12-31", "2018-12-31",
                                  "2020-12-31", "2021-01-10", "2024-01-01")

    def test_rejects_invalid_date_strings(self):
        with self.assertRaisesRegex(ValueError, "valid dates"):
            _validate_split_dates("not-a-date", "2018-12-31", "2019-01-10",
                                  "2020-12-31", "2021-01-10", "2024-01-01")

    def test_rejects_too_short_train_validation_embargo(self):
        with self.assertRaisesRegex(ValueError, "train/validation split requires"):
            _validate_split_dates("2010-01-01", "2018-12-31", "2019-01-01",
                                  "2020-12-31", "2021-01-10", "2024-01-01")

    def test_rejects_too_short_validation_test_embargo(self):
        with self.assertRaisesRegex(ValueError, "validation/test split requires"):
            _validate_split_dates("2010-01-01", "2018-12-31", "2019-01-10",
                                  "2020-12-31", "2021-01-01", "2024-01-01")

    def test_accepts_chronological_splits_with_embargo(self):
        dates = _validate_split_dates("2010-01-01", "2018-12-31", "2019-01-10",
                                      "2020-12-31", "2021-01-10", "2024-01-01")
        self.assertEqual(len(dates), 6)

    def test_rejects_invalid_top_k(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            run_experiment(
                provider_uri="/unused",
                train_start="2010-01-01", train_end="2018-12-31",
                valid_start="2019-01-10", valid_end="2020-12-31",
                test_start="2021-01-10", test_end="2024-01-01", top_k=0,
            )

    def test_rejects_negative_cost(self):
        with self.assertRaisesRegex(ValueError, "cost_bps"):
            run_experiment(
                provider_uri="/unused",
                train_start="2010-01-01", train_end="2018-12-31",
                valid_start="2019-01-10", valid_end="2020-12-31",
                test_start="2021-01-10", test_end="2024-01-01", cost_bps=-1,
            )

    def test_portfolio_turnover_and_costs_are_calculated(self):
        observations = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "AAA", "factor": 2.0, "forward_return": 0.02},
            {"date": "2024-01-02", "ticker": "BBB", "factor": 1.0, "forward_return": 0.01},
            {"date": "2024-01-03", "ticker": "AAA", "factor": 3.0, "forward_return": 0.03},
            {"date": "2024-01-03", "ticker": "BBB", "factor": 1.0, "forward_return": -0.01},
        ])
        daily = _build_daily_portfolio(observations, top_k=1, cost_bps=10.0)
        self.assertAlmostEqual(daily.iloc[0]["turnover"], 1.0)
        self.assertAlmostEqual(daily.iloc[0]["transaction_cost"], 0.001)
        self.assertAlmostEqual(daily.iloc[0]["net_return"], 0.019)
        self.assertAlmostEqual(daily.iloc[1]["turnover"], 0.0)
        self.assertAlmostEqual(daily.iloc[1]["net_return"], 0.03)

    def test_rejects_return_below_minus_100_percent(self):
        observations = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "AAA", "factor": 2.0, "forward_return": -1.01},
        ])
        with self.assertRaisesRegex(ValueError, "below -100%"):
            _build_daily_portfolio(observations, top_k=1, cost_bps=10.0)

    def test_cost_sensitivity_reuses_same_gross_returns_and_net_falls_with_cost(self):
        observations = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "AAA", "factor": 2.0, "forward_return": 0.02},
            {"date": "2024-01-02", "ticker": "BBB", "factor": 1.0, "forward_return": 0.01},
            {"date": "2024-01-03", "ticker": "AAA", "factor": 3.0, "forward_return": 0.03},
            {"date": "2024-01-03", "ticker": "BBB", "factor": 1.0, "forward_return": -0.01},
        ])
        result = _cost_sensitivity(observations, top_k=1, cost_bps_values=[20, 0, 10])
        self.assertEqual(result["cost_bps_per_traded_notional"].tolist(), [0.0, 10.0, 20.0])
        self.assertAlmostEqual(result.iloc[0]["oos_total_return"], 0.0506)
        self.assertLess(result.iloc[1]["oos_total_return"], result.iloc[0]["oos_total_return"])
        self.assertLess(result.iloc[2]["oos_total_return"], result.iloc[1]["oos_total_return"])

    def test_rejects_nonfinite_cost_before_qlib_import(self):
        with self.assertRaisesRegex(ValueError, "cost_bps"):
            run_experiment(
                provider_uri="/unused",
                train_start="2010-01-01", train_end="2018-12-31",
                valid_start="2019-01-10", valid_end="2020-12-31",
                test_start="2021-01-10", test_end="2024-01-01", cost_bps=float("nan"),
            )

    def test_portfolio_rejects_nonfinite_cost(self):
        observations = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "AAA", "factor": 2.0, "forward_return": 0.02},
        ])
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            _build_daily_portfolio(observations, top_k=1, cost_bps=float("inf"))

    def test_cost_sensitivity_rejects_invalid_costs(self):
        observations = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "AAA", "factor": 2.0, "forward_return": 0.02},
        ])
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            _cost_sensitivity(observations, top_k=1, cost_bps_values=[float("nan")])

    def test_portfolio_rejects_duplicate_date_ticker_rows(self):
        observations = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "AAA", "factor": 2.0, "forward_return": 0.02},
            {"date": "2024-01-02", "ticker": "AAA", "factor": 1.0, "forward_return": 0.01},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate date/ticker"):
            _build_daily_portfolio(observations, top_k=1, cost_bps=10.0)

    def test_portfolio_rejects_duplicate_tickers_after_normalization(self):
        observations = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "AAA", "factor": 2.0, "forward_return": 0.02},
            {"date": "2024-01-02", "ticker": " AAA ", "factor": 1.0, "forward_return": 0.01},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate date/ticker"):
            _build_daily_portfolio(observations, top_k=1, cost_bps=10.0)

    def test_portfolio_rejects_empty_usable_rows(self):
        observations = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "AAA", "factor": float("nan"), "forward_return": 0.02},
        ])
        with self.assertRaisesRegex(ValueError, "No usable"):
            _build_daily_portfolio(observations, top_k=1, cost_bps=10.0)


if __name__ == "__main__":
    unittest.main()
