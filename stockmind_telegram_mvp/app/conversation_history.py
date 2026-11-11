"""Durable, bounded text-only conversation history for Telegram users."""

import sqlite3
import threading
from pathlib import Path

DEFAULT_MAX_HISTORY_BYTES = 24 * 1024
_TRUNCATION_MARKER = "\n[message shortened to fit the conversation context budget]"


def _validate_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


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


def _truncate_utf8(content, max_bytes):
    """Shorten oversized text without splitting UTF-8 or hiding the omission."""
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content

    marker = _TRUNCATION_MARKER.encode("utf-8")
    if max_bytes <= len(marker):
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    available = max_bytes - len(marker)
    head_budget = (available * 2) // 3
    tail_budget = available - head_budget
    head = encoded[:head_budget].decode("utf-8", errors="ignore")
    tail = encoded[-tail_budget:].decode("utf-8", errors="ignore") if tail_budget else ""
    return head + _TRUNCATION_MARKER + tail


def _fit_recent_rows(rows_newest_first, max_messages, max_bytes, include_id=False):
    """Keep a contiguous newest suffix within both message and UTF-8 byte caps."""
    selected = []
    used_bytes = 0
    per_message_limit = max(1, max_bytes // 2)

    for row in rows_newest_first[:max_messages]:
        if include_id:
            row_id, role, content = row
        else:
            role, content = row

        bounded_content = _truncate_utf8(content, per_message_limit)
        content_bytes = len(bounded_content.encode("utf-8"))
        if selected and used_bytes + content_bytes > max_bytes:
            break
        if used_bytes + content_bytes > max_bytes:
            bounded_content = _truncate_utf8(content, max_bytes - used_bytes)
            content_bytes = len(bounded_content.encode("utf-8"))

        if include_id:
            selected.append((row_id, role, bounded_content))
        else:
            selected.append((role, bounded_content))
        used_bytes += content_bytes

    return selected


def bound_history(messages, max_messages, max_bytes=DEFAULT_MAX_HISTORY_BYTES):
    """Return the newest chronological messages within count and UTF-8 byte caps."""
    max_messages = _validate_positive_int(max_messages, "max_messages")
    max_bytes = _validate_positive_int(max_bytes, "max_bytes")
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages must be a list or tuple.")
    if not messages:
        return []

    validated = _validate_messages(messages)
    newest_first = list(reversed(validated[-max_messages:]))
    selected = _fit_recent_rows(newest_first, max_messages, max_bytes)
    selected.reverse()
    return [{"role": role, "content": content} for role, content in selected]


class ConversationHistoryStore:
    """Persist recent user/assistant messages in the local StockMind SQLite DB.

    Image bytes and provider payloads are deliberately not stored. History is
    bounded by both message count and UTF-8 bytes to cap storage and model
    context. This is single-host SQLite storage, not a distributed conversation
    lock.
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
        return _validate_positive_int(max_messages, "max_messages")

    @staticmethod
    def _validate_byte_limit(max_bytes):
        return _validate_positive_int(max_bytes, "max_bytes")

    def get(self, user_id, max_messages, max_bytes=DEFAULT_MAX_HISTORY_BYTES):
        """Return the newest bounded messages in chronological order."""
        user_id = self._validate_user_id(user_id)
        max_messages = self._validate_limit(max_messages)
        max_bytes = self._validate_byte_limit(max_bytes)
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
        selected = _fit_recent_rows(rows, max_messages, max_bytes)
        selected.reverse()
        return [{"role": role, "content": content} for role, content in selected]

    def append(
        self,
        user_id,
        messages,
        max_messages,
        max_bytes=DEFAULT_MAX_HISTORY_BYTES,
    ):
        """Append messages and atomically retain the newest count/byte-bounded suffix."""
        user_id = self._validate_user_id(user_id)
        max_messages = self._validate_limit(max_messages)
        max_bytes = self._validate_byte_limit(max_bytes)
        validated = _validate_messages(messages)
        per_message_limit = max(1, max_bytes // 2)
        bounded_new = [
            (role, _truncate_utf8(content, per_message_limit))
            for role, content in validated
        ]

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.executemany(
                    """
                    INSERT INTO conversation_messages (user_id, role, content)
                    VALUES (?, ?, ?)
                    """,
                    [(user_id, role, content) for role, content in bounded_new],
                )
                rows = self._connection.execute(
                    """
                    SELECT id, role, content
                    FROM conversation_messages
                    WHERE user_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (user_id, max_messages),
                ).fetchall()
                selected = _fit_recent_rows(
                    rows, max_messages, max_bytes, include_id=True
                )

                for row_id, _role, content in selected:
                    self._connection.execute(
                        "UPDATE conversation_messages SET content = ? WHERE id = ?",
                        (content, row_id),
                    )

                keep_ids = [row_id for row_id, _role, _content in selected]
                if keep_ids:
                    placeholders = ",".join("?" for _ in keep_ids)
                    self._connection.execute(
                        f"""
                        DELETE FROM conversation_messages
                        WHERE user_id = ? AND id NOT IN ({placeholders})
                        """,
                        (user_id, *keep_ids),
                    )
                else:
                    self._connection.execute(
                        "DELETE FROM conversation_messages WHERE user_id = ?",
                        (user_id,),
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
