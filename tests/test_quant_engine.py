import unittest

import numpy as np
import pandas as pd

from quant_engine import calculate_factors, rank_universe


class FactorEngineTests(unittest.TestCase):
    def test_calculate_factors_on_rising_series(self):
        idx = pd.date_range("2024-01-01", periods=260, freq="B")
        close = pd.Series(np.linspace(50, 150, len(idx)), index=idx)
        frame = pd.DataFrame({"Close": close, "Volume": 1_000_000}, index=idx)
        factors = calculate_factors(frame)
        self.assertGreater(factors["return_20d"], 0)
        self.assertGreater(factors["return_60d"], 0)
        self.assertTrue(factors["above_sma50"])
        self.assertAlmostEqual(factors["drawdown_120d"], 0.0)

    def test_requires_enough_data(self):
        frame = pd.DataFrame({"Close": [10, 11, 12]})
        with self.assertRaises(ValueError):
            calculate_factors(frame)

    def test_missing_momentum_gets_neutral_score(self):
        scores = pd.Series([0.10, np.nan, -0.10], index=["A", "B", "C"])
        from quant_engine import _percentile_score
        result = _percentile_score(scores)
        self.assertEqual(result["B"], 50.0)

    def test_missing_risk_inputs_fail_closed(self):
        frame = pd.DataFrame([
            {"ticker": "AAA", "return_20d": .08, "return_60d": .20,
             "return_120d": .30, "volatility_20d_ann": np.nan,
             "drawdown_120d": -.05, "above_sma50": True, "above_sma200": True}
        ])
        result = rank_universe(frame)
        self.assertFalse(bool(result.iloc[0]["risk_pass"]))
        self.assertEqual(result.iloc[0]["recommendation"], "RISK_FILTERED")

    def test_risk_filter_and_ranking(self):
        frame = pd.DataFrame([
            {"ticker": "AAA", "return_20d": .08, "return_60d": .20,
             "return_120d": .30, "volatility_20d_ann": .25,
             "drawdown_120d": -.05, "above_sma50": True, "above_sma200": True},
            {"ticker": "BBB", "return_20d": -.20, "return_60d": -.10,
             "return_120d": -.05, "volatility_20d_ann": 1.20,
             "drawdown_120d": -.40, "above_sma50": False, "above_sma200": False},
        ])
        result = rank_universe(frame)
        indexed = result.set_index("ticker")
        self.assertTrue(bool(indexed.loc["AAA", "risk_pass"]))
        self.assertFalse(bool(indexed.loc["BBB", "risk_pass"]))
        self.assertEqual(indexed.loc["AAA", "recommendation"], "REVIEW")
        self.assertEqual(indexed.loc["BBB", "recommendation"], "RISK_FILTERED")


if __name__ == "__main__":
    unittest.main()
