import unittest

from stockmind_telegram_mvp.app.localization import (
    LANGUAGE_NAMES,
    normalize_language,
    resolve_language,
    translate,
)


class LocalizationTests(unittest.TestCase):
    def test_normalizes_telegram_locale_variants_and_names(self):
        self.assertEqual(normalize_language("hi-IN"), "hi")
        self.assertEqual(normalize_language("gu_IN"), "gu")
        self.assertEqual(normalize_language(" Gujarati "), "gu")
        self.assertEqual(normalize_language("हिन्दी"), "hi")
        self.assertIsNone(normalize_language("fr-FR"))
        self.assertIsNone(normalize_language(None))

    def test_explicit_preference_overrides_telegram_locale(self):
        self.assertEqual(resolve_language("gu", "hi-IN"), "gu")

    def test_unsupported_preference_falls_back_to_telegram_locale_then_english(self):
        self.assertEqual(resolve_language("fr", "hi-IN"), "hi")
        self.assertEqual(resolve_language("fr", "fr-FR"), "en")

    def test_translation_uses_english_for_unknown_language(self):
        self.assertEqual(
            translate("fr", "daily_limit"),
            translate("en", "daily_limit"),
        )

    def test_translated_timeout_interpolates_seconds_without_losing_unicode(self):
        message = translate("hi", "chart_timeout", seconds=12.5)
        self.assertIn("12.5", message)
        self.assertIn("चार्ट", message)

    def test_all_supported_languages_have_core_beta_messages(self):
        required = (
            "language_help", "language_invalid", "language_changed",
            "language_changed_not_saved",
            "welcome", "beta_full", "daily_limit", "quota_storage_unavailable",
            "progress_chart",
            "progress_research", "reset_done", "reset_lock_unavailable", "unsupported_message",
        )
        for language in ("en", "hi", "gu"):
            for key in required:
                with self.subTest(language=language, key=key):
                    self.assertTrue(translate(language, key, language_name=LANGUAGE_NAMES["en"]))


if __name__ == "__main__":
    unittest.main()
