import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import usage_tracker


class UsageTrackerTests(unittest.TestCase):
    def test_local_counter_increments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "counter.json"
            with patch.object(usage_tracker, "LOCAL_COUNTER_PATH", path):
                self.assertEqual(usage_tracker._increment_local(), 1)
                self.assertEqual(usage_tracker._increment_local(), 2)
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["total_uses"], 2)
                self.assertTrue(saved["last_used_at"])

    def test_rpc_total_parser(self):
        self.assertEqual(usage_tracker._parse_rpc_total(7), 7)
        self.assertEqual(usage_tracker._parse_rpc_total([8]), 8)
        self.assertEqual(usage_tracker._parse_rpc_total({"total_uses": 9}), 9)
        self.assertEqual(usage_tracker._parse_rpc_total("10"), 10)


if __name__ == "__main__":
    unittest.main()
