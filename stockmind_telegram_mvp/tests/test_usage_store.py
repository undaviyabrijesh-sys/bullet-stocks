import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from stockmind_telegram_mvp.app.usage_store import AdmissionStore


class AdmissionStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.directory.name) / "stockmind.sqlite3"
        self.first = AdmissionStore(self.database_path)
        self.second = AdmissionStore(self.database_path)

    def tearDown(self):
        self.second.close()
        self.first.close()
        self.directory.cleanup()

    def test_beta_membership_survives_a_new_store_instance(self):
        self.assertTrue(self.first.admit(101, 2))
        self.assertTrue(self.second.admit(102, 2))
        self.assertFalse(self.second.admit(103, 2))
        self.assertEqual(self.first.beta_user_count(), 2)
        self.assertIsNone(self.second.check_status(101, "2026-11-11", 2, 10))

    def test_daily_usage_persists_and_resets_on_a_new_date(self):
        self.assertEqual(
            self.first.admit_and_consume(101, "2026-11-11", 2, 1), "ok"
        )
        self.assertEqual(
            self.second.check_status(101, "2026-11-11", 2, 1), "daily_limit"
        )
        self.assertIsNone(self.second.check_status(101, "2026-11-12", 2, 1))
        self.assertEqual(
            self.second.admit_and_consume(101, "2026-11-12", 2, 1), "ok"
        )
        self.assertEqual(self.first.daily_usage(101, "2026-11-12"), 1)

    def test_preflight_does_not_consume_a_request_slot(self):
        self.assertTrue(self.first.admit(101, 2))
        self.assertIsNone(self.second.check_status(101, "2026-11-11", 2, 1))
        self.assertIsNone(self.second.check_status(101, "2026-11-11", 2, 1))
        self.assertEqual(self.first.daily_usage(101, "2026-11-11"), 0)

    def test_atomic_beta_cap_across_independent_connections(self):
        stores = [self.first, self.second]
        user_ids = list(range(1000, 1012))

        def admit(index_and_user):
            index, user_id = index_and_user
            return stores[index % 2].admit(user_id, 3)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(admit, enumerate(user_ids)))
        self.assertEqual(sum(results), 3)
        self.assertEqual(self.first.beta_user_count(), 3)

    def test_atomic_daily_limit_across_independent_connections(self):
        self.assertTrue(self.first.admit(101, 2))

        def reserve(index):
            store = self.first if index % 2 == 0 else self.second
            return store.admit_and_consume(101, "2026-11-11", 2, 2)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(reserve, range(12)))
        self.assertEqual(results.count("ok"), 2)
        self.assertEqual(results.count("daily_limit"), 10)
        self.assertEqual(self.first.daily_usage(101, "2026-11-11"), 2)

    def test_rejects_invalid_user_ids_dates_and_limits(self):
        with self.assertRaises(ValueError):
            self.first.admit(True, 2)
        with self.assertRaises(ValueError):
            self.first.check_status(101, "2026-02-30", 2, 10)
        with self.assertRaises(ValueError):
            self.first.admit(101, 0)
        with self.assertRaises(ValueError):
            self.first.admit_and_consume(101, "2026-11-11", 2, False)


if __name__ == "__main__":
    unittest.main()
