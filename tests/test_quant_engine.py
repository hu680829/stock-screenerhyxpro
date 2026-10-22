import unittest
import numpy as np
import pandas as pd
from quant_engine import calculate_factors, rank_universe, _percentile_score

class FactorEngineTests(unittest.TestCase):
    def test_calculate_factors_on_rising_series(self):
        idx = pd.date_range("2024-01-01", periods=260, freq="B")
        frame = pd.DataFrame({"Close": np.linspace(50, 150, len(idx)), "Volume": 1000000}, index=idx)
        f = calculate_factors(frame)
        self.assertGreater(f["return_20d"], 0)
        self.assertTrue(f["above_sma50"])
        self.assertTrue(f["above_sma200"])
        self.assertAlmostEqual(f["drawdown_120d"], 0.0)

    def test_short_history_does_not_fake_long_averages(self):
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        f = calculate_factors(pd.DataFrame({"Close": np.linspace(50, 100, len(idx))}, index=idx))
        self.assertTrue(np.isnan(f["return_120d"]))
        self.assertTrue(np.isnan(f["above_sma200"]))
        self.assertTrue(f["sma50_available"])
        self.assertFalse(f["sma200_available"])

    def test_requires_enough_data(self):
        with self.assertRaises(ValueError):
            calculate_factors(pd.DataFrame({"Close": [10, 11, 12]}))

    def test_percentile_requires_cross_section(self):
        self.assertTrue(_percentile_score(pd.Series([.1, np.nan])).isna().all())

    def test_missing_risk_inputs_fail_closed(self):
        frame = pd.DataFrame([{"ticker":"AAA","return_20d":.08,"return_60d":.2,
            "return_120d":.3,"volatility_20d_ann":np.nan,"drawdown_120d":-.05,
            "above_sma50":True,"above_sma200":True}])
        result = rank_universe(frame)
        self.assertFalse(bool(result.iloc[0]["risk_pass"]))
        self.assertEqual(result.iloc[0]["recommendation"], "RISK_FILTERED")

    def test_risk_filter_and_score_coverage(self):
        frame = pd.DataFrame([
          {"ticker":"AAA","return_20d":.08,"return_60d":.2,"return_120d":.3,
           "volatility_20d_ann":.25,"drawdown_120d":-.05,"above_sma50":True,"above_sma200":True},
          {"ticker":"BBB","return_20d":-.2,"return_60d":-.1,"return_120d":-.05,
           "volatility_20d_ann":1.2,"drawdown_120d":-.4,"above_sma50":False,"above_sma200":False}])
        result = rank_universe(frame).set_index("ticker")
        self.assertTrue(bool(result.loc["AAA","risk_pass"]))
        self.assertFalse(bool(result.loc["BBB","risk_pass"]))
        self.assertGreater(result.loc["AAA","factor_coverage"], 0)

if __name__ == "__main__":
    unittest.main()
