"""Minimal durable storage for explicit Telegram UI-language preferences."""

import sqlite3
import threading
from pathlib import Path

from .localization import normalize_language


class LanguagePreferenceStore:
    """Store only a Telegram user ID and a supported interface-language code.

    SQLite makes explicit language choices survive process restarts. This store
    is intended for the single-instance MVP on a persistent local volume; it is
    not a substitute for shared storage in a multi-replica deployment.
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
        )
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_language_preferences (
                user_id INTEGER PRIMARY KEY CHECK (user_id > 0),
                language TEXT NOT NULL CHECK (language IN ('en', 'hi', 'gu')),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()

    @staticmethod
    def _validate_user_id(user_id):
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("Telegram user_id must be a positive integer.")
        return user_id

    def get(self, user_id):
        user_id = self._validate_user_id(user_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT language FROM user_language_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        # Treat an unexpected/corrupt value as absent rather than trusting it.
        return normalize_language(row[0])

    def set(self, user_id, language):
        user_id = self._validate_user_id(user_id)
        normalized = normalize_language(language)
        if normalized is None:
            raise ValueError("Unsupported StockMind interface language.")
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO user_language_preferences (user_id, language, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    language = excluded.language,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, normalized),
            )
            self._connection.commit()

    def clear(self):
        """Clear all rows; intended for isolated tests and explicit local resets."""
        with self._lock:
            self._connection.execute("DELETE FROM user_language_preferences")
            self._connection.commit()

    def close(self):
        with self._lock:
            self._connection.close()
