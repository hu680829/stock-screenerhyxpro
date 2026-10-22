import unittest

import pandas as pd

from portfolio_monitor import calculate_snapshot


class PortfolioMonitorTests(unittest.TestCase):
    def test_snapshot_uses_latest_shared_sessions_and_fx(self):
        idx = pd.to_datetime(["2026-10-19", "2026-10-20", "2026-10-21"])
        prices = pd.DataFrame({
            "AAPL": [100, 101, 102],
            "NVDA": [200, 202, 204],
            "BTC-USD": [1000, 1010, 1020],
            "000300.SS": [4000, 4020, 4040],
            "SGOV": [100, 100.1, 100.2],
            "USDCNH=X": [7.0, 7.05, 7.1],
        }, index=idx)
        result = calculate_snapshot(prices)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["previous_session"], "2026-10-20")
        self.assertEqual(result["latest_session"], "2026-10-21")
        self.assertGreater(result["usd_cnh_return"], 0)
        self.assertAlmostEqual(sum(result["current_weights"].values()), 1.0)


    def test_duplicate_session_labels_are_deduplicated(self):
        idx = pd.to_datetime([
            "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-21"
        ])
        prices = pd.DataFrame({
            "AAPL": [100, 101, 102, 103],
            "NVDA": [200, 202, 204, 206],
            "BTC-USD": [1000, 1010, 1020, 1030],
            "000300.SS": [4000, 4020, 4040, 4060],
            "SGOV": [100, 100.1, 100.2, 100.3],
            "USDCNH=X": [7.0, 7.05, 7.1, 7.15],
        }, index=idx)
        result = calculate_snapshot(prices)
        self.assertEqual(result["previous_session"], "2026-10-20")
        self.assertEqual(result["latest_session"], "2026-10-21")
        self.assertGreater(result["portfolio_return"], 0)

    def test_missing_shared_session_fails_closed(self):
        idx = pd.to_datetime(["2026-10-20", "2026-10-21", "2026-10-22"])
        prices = pd.DataFrame({
            "AAPL": [100, 101, 102],
            "NVDA": [None, None, 204],
            "BTC-USD": [1000, 1010, 1020],
            "000300.SS": [4000, 4020, 4040],
            "SGOV": [100, 100.1, 100.2],
            "USDCNH=X": [7.0, 7.05, 7.1],
        }, index=idx)
        with self.assertRaisesRegex(ValueError, "shared valid sessions"):
            calculate_snapshot(prices)


if __name__ == "__main__":
    unittest.main()
