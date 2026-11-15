"""Startup configuration parsing tests; no external services are contacted."""
import importlib
import os
import unittest
from unittest.mock import patch


_TEST_ENV = {
    "TELEGRAM_BOT_TOKEN": "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",
    "OPENAI_API_KEY": "stockmind-test-key",
    "TAPETIDE_TOKEN": "stockmind-test-tapetide-token",
    "MAX_CHART_IMAGE_BYTES": "5242880",
    "MAX_HISTORY_BYTES": "24576",
    "RESEARCH_TURN_TIMEOUT_SECONDS": "120",
    "USER_LOCK_WAIT_TIMEOUT_SECONDS": "30",
    "CHART_CAPACITY_WAIT_TIMEOUT_SECONDS": "30",
    "STOCKMIND_PREFERENCES_DB": ":memory:",
}
with patch.dict(os.environ, _TEST_ENV):
    config = importlib.import_module("stockmind_telegram_mvp.app.config")


class PositiveIntegerSettingTests(unittest.TestCase):
    def test_uses_default_when_setting_is_absent(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config._positive_int_setting("TEST_LIMIT", 4096), 4096)

    def test_accepts_positive_integer_setting(self):
        with patch.dict(os.environ, {"TEST_LIMIT": "8192"}, clear=True):
            self.assertEqual(config._positive_int_setting("TEST_LIMIT", 4096), 8192)

    def test_rejects_zero_negative_and_non_integer_values(self):
        for raw_value in ("0", "-1", "1.5", "not-a-number"):
            with self.subTest(raw_value=raw_value):
                with patch.dict(os.environ, {"TEST_LIMIT": raw_value}, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "positive integer"):
                        config._positive_int_setting("TEST_LIMIT", 4096)


class PositiveFiniteFloatSettingTests(unittest.TestCase):
    def test_uses_default_when_setting_is_absent(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                config._positive_finite_float_setting("TEST_TIMEOUT", 120.0),
                120.0,
            )

    def test_accepts_positive_finite_timeout(self):
        with patch.dict(os.environ, {"TEST_TIMEOUT": "0.25"}, clear=True):
            self.assertEqual(
                config._positive_finite_float_setting("TEST_TIMEOUT", 120.0),
                0.25,
            )

    def test_rejects_zero_negative_non_finite_and_invalid_values(self):
        for raw_value in ("0", "-1", "nan", "inf", "-inf", "not-a-number"):
            with self.subTest(raw_value=raw_value):
                with patch.dict(os.environ, {"TEST_TIMEOUT": raw_value}, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "positive finite number"):
                        config._positive_finite_float_setting("TEST_TIMEOUT", 120.0)


class ConfigurationWiringTests(unittest.TestCase):
    def test_image_and_timeout_settings_use_validated_parsers(self):
        self.assertEqual(config.MAX_CHART_IMAGE_BYTES, 5 * 1024 * 1024)
        self.assertEqual(config.MAX_HISTORY_BYTES, 24 * 1024)
        self.assertEqual(config.RESEARCH_TURN_TIMEOUT_SECONDS, 120.0)
        self.assertEqual(config.USER_LOCK_WAIT_TIMEOUT_SECONDS, 30.0)
        self.assertEqual(config.CHART_CAPACITY_WAIT_TIMEOUT_SECONDS, 30.0)


if __name__ == "__main__":
    unittest.main()
