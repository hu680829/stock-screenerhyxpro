import unittest
from research.train_qlib_alpha158 import run_experiment


class QlibResearchTests(unittest.TestCase):
    def test_rejects_non_chronological_splits_before_importing_qlib(self):
        with self.assertRaisesRegex(ValueError, "chronological"):
            run_experiment(
                provider_uri="/unused",
                train_start="2020-01-01", train_end="2021-01-01",
                valid_start="2019-01-01", valid_end="2020-01-01",
                test_start="2022-01-01", test_end="2023-01-01",
            )

    def test_rejects_invalid_top_k(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            run_experiment(
                provider_uri="/unused",
                train_start="2010-01-01", train_end="2018-01-01",
                valid_start="2018-01-02", valid_end="2020-01-01",
                test_start="2020-01-02", test_end="2024-01-01",
                top_k=0,
            )

    def test_rejects_negative_cost(self):
        with self.assertRaisesRegex(ValueError, "cost_bps"):
            run_experiment(
                provider_uri="/unused",
                train_start="2010-01-01", train_end="2018-01-01",
                valid_start="2018-01-02", valid_end="2020-01-01",
                test_start="2020-01-02", test_end="2024-01-01",
                cost_bps=-1,
            )


if __name__ == "__main__":
    unittest.main()
