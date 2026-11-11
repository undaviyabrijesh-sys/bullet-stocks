import tempfile
import unittest
from pathlib import Path

from stockmind_telegram_mvp.app.user_preferences import LanguagePreferenceStore


class LanguagePreferenceStoreTests(unittest.TestCase):
    def test_preference_survives_a_new_store_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "preferences.sqlite3"
            first = LanguagePreferenceStore(database_path)
            second = None
            try:
                first.set(123456789, "Gujarati")
                second = LanguagePreferenceStore(database_path)
                self.assertEqual(second.get(123456789), "gu")
            finally:
                if second is not None:
                    second.close()
                first.close()

    def test_updates_replace_the_existing_language(self):
        store = LanguagePreferenceStore(":memory:")
        try:
            store.set(123456789, "gu")
            store.set(123456789, "hi-IN")
            self.assertEqual(store.get(123456789), "hi")
        finally:
            store.close()

    def test_missing_preference_returns_none(self):
        store = LanguagePreferenceStore(":memory:")
        try:
            self.assertIsNone(store.get(123456789))
        finally:
            store.close()

    def test_rejects_unsupported_language(self):
        store = LanguagePreferenceStore(":memory:")
        try:
            with self.assertRaises(ValueError):
                store.set(123456789, "fr")
        finally:
            store.close()

    def test_rejects_invalid_user_ids(self):
        store = LanguagePreferenceStore(":memory:")
        try:
            for user_id in (0, -1, True, "123"):
                with self.subTest(user_id=user_id):
                    with self.assertRaises(ValueError):
                        store.get(user_id)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
