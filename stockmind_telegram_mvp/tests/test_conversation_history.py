import tempfile
import unittest
from pathlib import Path

from stockmind_telegram_mvp.app.conversation_history import ConversationHistoryStore


class ConversationHistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.directory.name) / "stockmind.sqlite3"
        self.first = ConversationHistoryStore(self.database_path)
        self.second = ConversationHistoryStore(self.database_path)

    def tearDown(self):
        self.second.close()
        self.first.close()
        self.directory.cleanup()

    def test_history_survives_a_new_store_instance(self):
        messages = [
            {"role": "user", "content": "Analyse BTC"},
            {"role": "assistant", "content": "Visible structure is uncertain."},
        ]
        self.first.append(101, messages, 12)
        self.assertEqual(self.second.get(101, 12), messages)

    def test_history_is_trimmed_to_the_newest_configured_message_count(self):
        self.first.append(
            101,
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
            3,
        )
        self.assertEqual(
            self.second.get(101, 12),
            [
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
        )

    def test_users_have_isolated_histories(self):
        self.first.append(101, [{"role": "user", "content": "private A"}], 12)
        self.first.append(202, [{"role": "user", "content": "private B"}], 12)
        self.assertEqual(
            self.second.get(101, 12),
            [{"role": "user", "content": "private A"}],
        )
        self.assertEqual(
            self.second.get(202, 12),
            [{"role": "user", "content": "private B"}],
        )

    def test_reset_deletes_only_the_selected_users_history(self):
        self.first.append(101, [{"role": "user", "content": "A"}], 12)
        self.first.append(202, [{"role": "user", "content": "B"}], 12)
        self.assertEqual(self.first.reset(101), 1)
        self.assertEqual(self.second.get(101, 12), [])
        self.assertEqual(self.second.get(202, 12), [{"role": "user", "content": "B"}])

    def test_rejects_invalid_user_ids_limits_and_messages(self):
        with self.assertRaises(ValueError):
            self.first.get(True, 12)
        with self.assertRaises(ValueError):
            self.first.get(101, False)
        with self.assertRaises(ValueError):
            self.first.append(101, [{"role": "system", "content": "no"}], 12)
        with self.assertRaises(ValueError):
            self.first.append(101, [{"role": "user", "content": 123}], 12)
        with self.assertRaises(ValueError):
            self.first.append(101, [{"role": [], "content": "bad"}], 12)


if __name__ == "__main__":
    unittest.main()
