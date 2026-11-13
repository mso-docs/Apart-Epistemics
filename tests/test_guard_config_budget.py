import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from apart_epistemics import budget, guard
from apart_epistemics.config import ROOT, load_config, validate_config

ET = ZoneInfo("America/New_York")


class GuardTest(unittest.TestCase):
    def test_weekday_before_5pm_refused(self):
        ok, why = guard.allowed(datetime(2026, 11, 13, 16, 59, tzinfo=ET))            # Friday
        self.assertFalse(ok)
        self.assertIn("refusing", why)
        self.assertTrue(guard.allowed(datetime(2026, 11, 13, 17, 0, tzinfo=ET))[0])
        self.assertFalse(guard.allowed(datetime(2026, 11, 13, 8, 0, tzinfo=ET))[0])
        self.assertFalse(guard.allowed(datetime(2026, 11, 13, 3, 0, tzinfo=ET))[0])

    def test_weekend_always_allowed(self):
        self.assertTrue(guard.allowed(datetime(2026, 11, 14, 9, 0, tzinfo=ET))[0])     # Saturday
        self.assertTrue(guard.allowed(datetime(2026, 11, 15, 13, 0, tzinfo=ET))[0])

    def test_timezone_conversion(self):
        # 20:30 UTC Friday = 15:30 ET: refused; 22:30 UTC = 17:30 ET: allowed
        utc = ZoneInfo("UTC")
        self.assertFalse(guard.allowed(datetime(2026, 11, 13, 20, 30, tzinfo=utc))[0])
        self.assertTrue(guard.allowed(datetime(2026, 11, 13, 22, 30, tzinfo=utc))[0])

    def test_require_raises(self):
        with self.assertRaises(guard.OutsideRunWindow):
            guard.require_window({"guard": {}}, datetime(2026, 11, 13, 9, 0, tzinfo=ET))


class ConfigTest(unittest.TestCase):
    def test_default_and_extends_valid(self):
        for name in ("default", "dev", "full", "synthetic"):
            cfg = load_config(ROOT / "configs" / f"{name}.yaml")
            self.assertEqual(validate_config(cfg), [], name)
        dev = load_config(ROOT / "configs" / "dev.yaml")
        self.assertEqual(dev["datasets"]["squad_v2"]["n_per_class"], 100)
        self.assertEqual(dev["datasets"]["squad_v2"]["split"], "dev")       # inherited via extends

    def test_invalid_configs_report_problems(self):
        cfg = load_config(ROOT / "configs" / "default.yaml")
        cfg["splits"] = {"train": 0.5, "val": 0.2, "test": 0.2}
        cfg["model"]["quantization"] = "fp4"
        cfg["evalboard"]["url"] = "http://example.com:8767"
        cfg["judge"] = {"backend": "openrouter"}
        errs = " | ".join(validate_config(cfg))
        for needle in ("sum to 1", "quantization", "loopback", "judge.model"):
            self.assertIn(needle, errs)

    def test_batch_size_fixed(self):
        cfg = load_config(ROOT / "configs" / "default.yaml")
        cfg["model"]["batch_size"] = 8
        self.assertTrue(any("batch_size" in e for e in validate_config(cfg)))


class BudgetTest(unittest.TestCase):
    def test_paid_calls_off_by_default(self):
        old = os.environ.pop("APART_ALLOW_PAID_CALLS", None)
        try:
            with self.assertRaises(budget.PaidCallsDisabled):
                budget.guarded_chat([{"role": "user", "content": "x"}], "m")
        finally:
            if old is not None:
                os.environ["APART_ALLOW_PAID_CALLS"] = old

    def test_guard_module_wired_and_cap_is_30(self):
        try:
            st = budget.status()
        except FileNotFoundError:
            self.skipTest("openrouter_budget.py not on this machine")
        self.assertEqual(st["cap_usd"], 30.0)
        self.assertFalse(st["paid_calls_allowed"])


if __name__ == "__main__":
    unittest.main()
