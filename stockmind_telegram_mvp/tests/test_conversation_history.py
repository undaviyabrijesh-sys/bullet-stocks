import tempfile
import unittest
from pathlib import Path

from stockmind_telegram_mvp.app.conversation_history import (
    ConversationHistoryStore,
    bound_history,
)


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


    def test_append_enforces_total_utf8_byte_budget_and_keeps_newest_context(self):
        messages = [
            {"role": "user", "content": "first " * 80},
            {"role": "assistant", "content": "second " * 80},
            {"role": "user", "content": "third " * 80},
            {"role": "assistant", "content": "fourth " * 80},
        ]
        self.first.append(101, messages, 12, max_bytes=256)

        history = self.second.get(101, 12, max_bytes=256)
        total_bytes = sum(len(message["content"].encode("utf-8")) for message in history)
        stored_bytes = self.first._connection.execute(
            "SELECT COALESCE(SUM(LENGTH(CAST(content AS BLOB))), 0) "
            "FROM conversation_messages WHERE user_id = ?",
            (101,),
        ).fetchone()[0]

        self.assertLessEqual(total_bytes, 256)
        self.assertLessEqual(stored_bytes, 256)
        self.assertEqual([message["role"] for message in history], ["user", "assistant"])
        self.assertTrue(all("[message shortened" in item["content"] for item in history))

    def test_unicode_truncation_preserves_valid_utf8_and_byte_budget(self):
        messages = [
            {"role": "user", "content": "ગુજરાતી ભાષા " * 80},
            {"role": "assistant", "content": "हिंदी उत्तर " * 80},
            {"role": "user", "content": "છેલ્લો પ્રશ્ન " * 80},
            {"role": "assistant", "content": "अंतिम उत्तर " * 80},
        ]
        self.first.append(101, messages, 12, max_bytes=240)

        history = self.second.get(101, 12, max_bytes=240)
        total_bytes = sum(len(message["content"].encode("utf-8")) for message in history)

        self.assertLessEqual(total_bytes, 240)
        self.assertEqual([message["role"] for message in history], ["user", "assistant"])
        for message in history:
            encoded = message["content"].encode("utf-8")
            self.assertEqual(encoded.decode("utf-8"), message["content"])
            self.assertLessEqual(len(encoded), 120)

    def test_get_bounds_legacy_oversized_rows_before_model_context(self):
        with self.first._lock:
            self.first._connection.executemany(
                """
                INSERT INTO conversation_messages (user_id, role, content)
                VALUES (?, ?, ?)
                """,
                [
                    (101, "user", "old " * 500),
                    (101, "assistant", "older answer " * 500),
                    (101, "user", "latest " * 500),
                    (101, "assistant", "latest answer " * 500),
                ],
            )

        history = self.second.get(101, 12, max_bytes=160)
        total_bytes = sum(len(message["content"].encode("utf-8")) for message in history)

        self.assertLessEqual(total_bytes, 100)
        self.assertEqual([message["role"] for message in history], ["user", "assistant"])
        self.assertTrue(all("[message shortened" in item["content"] for item in history))

    def test_bound_history_applies_same_limit_to_process_local_cache(self):
        messages = [
            {"role": "user", "content": "earlier " * 100},
            {"role": "assistant", "content": "answer " * 100},
            {"role": "user", "content": "latest " * 100},
            {"role": "assistant", "content": "final " * 100},
        ]
        history = bound_history(messages, 12, 200)

        self.assertLessEqual(
            sum(len(message["content"].encode("utf-8")) for message in history),
            200,
        )
        self.assertEqual([message["role"] for message in history], ["user", "assistant"])

    def test_rejects_non_positive_byte_budgets(self):
        with self.assertRaises(ValueError):
            self.first.get(101, 12, max_bytes=0)
        with self.assertRaises(ValueError):
            self.first.append(
                101, [{"role": "user", "content": "question"}], 12, max_bytes=-1
            )


if __name__ == "__main__":
    unittest.main()
