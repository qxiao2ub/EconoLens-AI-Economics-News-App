from __future__ import annotations

import unittest

from econolens_core import run_self_checks


class EconoLensCoreTests(unittest.TestCase):
    def test_builtin_self_checks(self) -> None:
        checks = run_self_checks()
        failed = [name for name, passed in checks.items() if not passed]
        self.assertFalse(failed, f"Failed checks: {failed}")


if __name__ == "__main__":
    unittest.main()
