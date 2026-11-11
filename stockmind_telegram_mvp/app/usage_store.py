"""Durable beta admission and per-day research quota accounting."""

import sqlite3
import threading
from datetime import date
from pathlib import Path


class AdmissionStore:
    """Persist beta membership and daily request slots in SQLite.

    The final admission + quota reservation is one BEGIN IMMEDIATE transaction,
    so separate bot processes sharing the same local database cannot overbook
    the beta cap or a user's daily quota. SQLite should live on persistent local
    storage; it is not a multi-host database or a distributed user lock.
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
            self.database_path, timeout=5.0, check_same_thread=False,
            isolation_level=None,
        )
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS beta_admissions (
                user_id INTEGER PRIMARY KEY CHECK (user_id > 0),
                admitted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_usage (
                user_id INTEGER NOT NULL CHECK (user_id > 0),
                usage_day TEXT NOT NULL,
                request_count INTEGER NOT NULL CHECK (request_count >= 0),
                PRIMARY KEY (user_id, usage_day)
            )
            """
        )

    @staticmethod
    def _validate_user_id(user_id):
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("Telegram user_id must be a positive integer.")
        return user_id

    @staticmethod
    def _validate_limit(limit, name):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return limit

    @staticmethod
    def _validate_day(usage_day):
        if not isinstance(usage_day, str):
            raise ValueError("usage_day must be an ISO-8601 calendar date.")
        try:
            parsed = date.fromisoformat(usage_day)
        except ValueError as exc:
            raise ValueError("usage_day must be an ISO-8601 calendar date.") from exc
        if parsed.isoformat() != usage_day:
            raise ValueError("usage_day must use canonical YYYY-MM-DD format.")
        return usage_day

    def check_status(self, user_id, usage_day, beta_limit, daily_limit):
        """Return beta_full/daily_limit, or None when a request may proceed.

        This is a non-consuming preflight. Call admit_and_consume after image
        validation for the authoritative atomic reservation.
        """
        user_id = self._validate_user_id(user_id)
        usage_day = self._validate_day(usage_day)
        beta_limit = self._validate_limit(beta_limit, "beta_limit")
        daily_limit = self._validate_limit(daily_limit, "daily_limit")
        with self._lock:
            admitted = self._connection.execute(
                "SELECT 1 FROM beta_admissions WHERE user_id = ?", (user_id,)
            ).fetchone()
            if admitted is None:
                count = self._connection.execute(
                    "SELECT COUNT(*) FROM beta_admissions"
                ).fetchone()[0]
                if count >= beta_limit:
                    return "beta_full"
            row = self._connection.execute(
                "SELECT request_count FROM daily_usage WHERE user_id = ? AND usage_day = ?",
                (user_id, usage_day),
            ).fetchone()
            if row is not None and row[0] >= daily_limit:
                return "daily_limit"
        return None

    def admit(self, user_id, beta_limit):
        """Persist beta membership without consuming a research request."""
        user_id = self._validate_user_id(user_id)
        beta_limit = self._validate_limit(beta_limit, "beta_limit")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._connection.execute(
                    "SELECT 1 FROM beta_admissions WHERE user_id = ?", (user_id,)
                ).fetchone()
                if existing is not None:
                    self._connection.commit()
                    return True
                count = self._connection.execute(
                    "SELECT COUNT(*) FROM beta_admissions"
                ).fetchone()[0]
                if count >= beta_limit:
                    self._connection.rollback()
                    return False
                self._connection.execute(
                    "INSERT INTO beta_admissions (user_id) VALUES (?)", (user_id,)
                )
                self._connection.commit()
                return True
            except Exception:
                self._connection.rollback()
                raise

    def admit_and_consume(self, user_id, usage_day, beta_limit, daily_limit):
        """Atomically enforce beta capacity and reserve one daily request slot.

        Returns exactly one of: ok, beta_full, daily_limit.
        """
        user_id = self._validate_user_id(user_id)
        usage_day = self._validate_day(usage_day)
        beta_limit = self._validate_limit(beta_limit, "beta_limit")
        daily_limit = self._validate_limit(daily_limit, "daily_limit")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                admitted = self._connection.execute(
                    "SELECT 1 FROM beta_admissions WHERE user_id = ?", (user_id,)
                ).fetchone()
                if admitted is None:
                    count = self._connection.execute(
                        "SELECT COUNT(*) FROM beta_admissions"
                    ).fetchone()[0]
                    if count >= beta_limit:
                        self._connection.rollback()
                        return "beta_full"
                row = self._connection.execute(
                    "SELECT request_count FROM daily_usage WHERE user_id = ? AND usage_day = ?",
                    (user_id, usage_day),
                ).fetchone()
                if row is not None and row[0] >= daily_limit:
                    self._connection.rollback()
                    return "daily_limit"
                if admitted is None:
                    self._connection.execute(
                        "INSERT INTO beta_admissions (user_id) VALUES (?)", (user_id,)
                    )
                self._connection.execute(
                    """
                    INSERT INTO daily_usage (user_id, usage_day, request_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(user_id, usage_day) DO UPDATE SET
                        request_count = daily_usage.request_count + 1
                    """,
                    (user_id, usage_day),
                )
                self._connection.commit()
                return "ok"
            except Exception:
                self._connection.rollback()
                raise

    def consume_daily_query(self, user_id, usage_day, daily_limit):
        """Atomically reserve one daily slot (for an already admitted user)."""
        user_id = self._validate_user_id(user_id)
        usage_day = self._validate_day(usage_day)
        daily_limit = self._validate_limit(daily_limit, "daily_limit")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT request_count FROM daily_usage WHERE user_id = ? AND usage_day = ?",
                    (user_id, usage_day),
                ).fetchone()
                if row is not None and row[0] >= daily_limit:
                    self._connection.rollback()
                    return False
                self._connection.execute(
                    """
                    INSERT INTO daily_usage (user_id, usage_day, request_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(user_id, usage_day) DO UPDATE SET
                        request_count = daily_usage.request_count + 1
                    """,
                    (user_id, usage_day),
                )
                self._connection.commit()
                return True
            except Exception:
                self._connection.rollback()
                raise

    def daily_usage(self, user_id, usage_day):
        user_id = self._validate_user_id(user_id)
        usage_day = self._validate_day(usage_day)
        with self._lock:
            row = self._connection.execute(
                "SELECT request_count FROM daily_usage WHERE user_id = ? AND usage_day = ?",
                (user_id, usage_day),
            ).fetchone()
        return 0 if row is None else row[0]

    def beta_user_count(self):
        with self._lock:
            return self._connection.execute(
                "SELECT COUNT(*) FROM beta_admissions"
            ).fetchone()[0]

    def clear(self):
        """Clear admission and quota rows; intended for isolated tests."""
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute("DELETE FROM daily_usage")
                self._connection.execute("DELETE FROM beta_admissions")
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def close(self):
        with self._lock:
            self._connection.close()
