import unittest

from portfolio_risk import cap_buy_allocations, portfolio_snapshot


class PortfolioRiskTests(unittest.TestCase):
    def test_fx_adjusted_return_and_weight_drift(self):
        result = portfolio_snapshot(
            {"US": 0.10, "CN": 0.05},
            {"US": 0.8, "CN": 0.2},
            usd_tickers={"US"},
            usd_cnh_return=0.02,
        )
        self.assertAlmostEqual(result["portfolio_return"], 0.8 * 0.122 + 0.2 * 0.05)
        self.assertAlmostEqual(sum(result["current_weights"].values()), 1.0)
        self.assertGreater(result["drift_pp"]["US"], 0)
        self.assertLess(result["drift_pp"]["CN"], 0)

    def test_missing_holding_data_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing return data"):
            portfolio_snapshot({"US": 0.01}, {"US": 0.5, "CN": 0.5})

    def test_invalid_fx_and_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "usd_cnh_return"):
            portfolio_snapshot({"US": 0.01}, {"US": 1.0}, usd_cnh_return=float("nan"))
        with self.assertRaisesRegex(ValueError, "target weights"):
            portfolio_snapshot({"US": 0.01}, {"US": 0.0})

    def test_buy_allocations_are_capped_proportionally(self):
        result = cap_buy_allocations({"AAA": 12.0, "BBB": 8.0}, 10.0)
        self.assertAlmostEqual(sum(result.values()), 10.0)
        self.assertAlmostEqual(result["AAA"] / result["BBB"], 1.5)

    def test_invalid_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_allocate_pct"):
            cap_buy_allocations({"AAA": 1.0}, float("inf"))


if __name__ == "__main__":
    unittest.main()
