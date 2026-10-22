import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import notify_pushplus


class PushPlusNotificationTests(unittest.TestCase):
    def write_csv(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False)
        writer = csv.DictWriter(handle, fieldnames=[
            "rank", "ticker", "composite_score", "risk_pass", "recommendation"
        ])
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_build_message_contains_rankings_and_risk_disclaimer(self):
        path = self.write_csv([{
            "rank": "1", "ticker": "AAPL", "composite_score": "82.5",
            "risk_pass": "True", "recommendation": "REVIEW"
        }])
        title, body = notify_pushplus.build_message(path)
        self.assertIn("通过风控 1 只", title)
        self.assertIn("AAPL", body)
        self.assertIn("不是买卖指令", body)

    def test_build_message_warns_when_no_symbol_passes_risk_gate(self):
        path = self.write_csv([{
            "rank": "1", "ticker": "NVDA", "composite_score": "55",
            "risk_pass": "False", "recommendation": "RISK_FILTERED"
        }])
        _, body = notify_pushplus.build_message(path)
        self.assertIn("没有标的通过风控", body)

    def test_empty_report_fails_closed(self):
        path = self.write_csv([])
        with self.assertRaises(ValueError):
            notify_pushplus.build_message(path)

    @patch("notify_pushplus.urlopen")
    def test_pushplus_api_rejection_raises(self, mocked_urlopen):
        response = MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({"code": 900, "msg": "bad token"}).encode()
        response.__enter__.return_value = response
        mocked_urlopen.return_value = response
        with self.assertRaises(RuntimeError):
            notify_pushplus.send_pushplus("not-a-real-token", "test", "test body")


if __name__ == "__main__":
    unittest.main()
