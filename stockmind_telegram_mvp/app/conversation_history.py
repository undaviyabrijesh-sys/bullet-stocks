"""Durable, bounded text-only conversation history for Telegram users."""

import sqlite3
import threading
from pathlib import Path


class ConversationHistoryStore:
    """Persist recent user/assistant messages in the local StockMind SQLite DB.

    Image bytes and provider payloads are deliberately not stored. The caller
    supplies a message-count limit so persisted context follows the bot config.
    This is single-host SQLite storage, not a distributed conversation lock.
    """

    def __init__(self, database_path):
        raw_path = str(database_path).strip()
        if not raw_path:
            raise ValueError("A non-empty SQLite database path is required.")
        if raw_path != ":memory:":
            path = Path(raw_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            raw_path = str(path)

        self.database_path = raw_path
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            timeout=5.0,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL CHECK (user_id > 0),
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_user_id_id
            ON conversation_messages (user_id, id DESC)
            """
        )

    @staticmethod
    def _validate_user_id(user_id):
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("Telegram user_id must be a positive integer.")
        return user_id

    @staticmethod
    def _validate_limit(max_messages):
        if (
            isinstance(max_messages, bool)
            or not isinstance(max_messages, int)
            or max_messages <= 0
        ):
            raise ValueError("max_messages must be a positive integer.")
        return max_messages

    @staticmethod
    def _validate_messages(messages):
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("messages must be a non-empty list or tuple.")
        validated = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each conversation message must be a dictionary.")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or role not in {"user", "assistant"}:
                raise ValueError("Conversation message role must be user or assistant.")
            if not isinstance(content, str):
                raise ValueError("Conversation message content must be text.")
            validated.append((role, content))
        return validated

    def get(self, user_id, max_messages):
        """Return up to max_messages in chronological order."""
        user_id = self._validate_user_id(user_id)
        max_messages = self._validate_limit(max_messages)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT role, content
                FROM conversation_messages
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, max_messages),
            ).fetchall()
        rows.reverse()
        return [{"role": role, "content": content} for role, content in rows]

    def append(self, user_id, messages, max_messages):
        """Append one or more messages and atomically retain only the newest N."""
        user_id = self._validate_user_id(user_id)
        max_messages = self._validate_limit(max_messages)
        validated = self._validate_messages(messages)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.executemany(
                    """
                    INSERT INTO conversation_messages (user_id, role, content)
                    VALUES (?, ?, ?)
                    """,
                    [(user_id, role, content) for role, content in validated],
                )
                self._connection.execute(
                    """
                    DELETE FROM conversation_messages
                    WHERE user_id = ?
                      AND id NOT IN (
                          SELECT id
                          FROM conversation_messages
                          WHERE user_id = ?
                          ORDER BY id DESC
                          LIMIT ?
                      )
                    """,
                    (user_id, user_id, max_messages),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def reset(self, user_id):
        """Delete all persisted conversation messages for one user."""
        user_id = self._validate_user_id(user_id)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._connection.execute(
                    "DELETE FROM conversation_messages WHERE user_id = ?",
                    (user_id,),
                )
                self._connection.commit()
                return cursor.rowcount
            except Exception:
                self._connection.rollback()
                raise

    def clear(self):
        """Clear all rows; intended for isolated tests."""
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute("DELETE FROM conversation_messages")
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def close(self):
        with self._lock:
            self._connection.close()
