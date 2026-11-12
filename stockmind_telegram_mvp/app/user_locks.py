"""Per-user request locks for one process or shared local SQLite processes.

The in-memory registry is suitable for tests and single-process deployments.
SQLiteUserLockRegistry coordinates separate processes that share the same
persistent local database file. It uses expiring leases and heartbeats so a
crashed process cannot leave a user permanently locked. It is not a multi-host
distributed lock; do not place the SQLite database on a network filesystem.
"""

import asyncio
import logging
import math
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Hashable

logger = logging.getLogger(__name__)

LEASE_LOST_CANCEL_MESSAGE = "stockmind-user-lock-lease-lost"


async def _await_shielded_to_completion(task):
    """Finish a worker/cleanup future even if the caller is cancelled repeatedly.

    asyncio.to_thread cannot stop its worker thread. If cancellation abandons a
    shielded acquisition or release, that thread may still mutate the lock row.
    Consume additional cancellation requests while waiting, then let the caller
    re-raise cancellation after the critical cleanup has completed.
    """
    cancellation = None
    while True:
        try:
            return await asyncio.shield(task), cancellation
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
            current_task = asyncio.current_task()
            if current_task is not None:
                current_task.uncancel()
            if task.done():
                return task.result(), cancellation


class UserLockBusy(RuntimeError):
    """Raised when another request holds this user's lock beyond the wait budget."""


class UserLockUnavailable(RuntimeError):
    """Raised when the shared lock store cannot safely grant or maintain a lock."""


class _LockEntry:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.references = 0


class UserLockRegistry:
    """Serialize work per key inside one asyncio event loop."""

    def __init__(self, *, wait_timeout_seconds: float = 30.0) -> None:
        if (
            isinstance(wait_timeout_seconds, bool)
            or not isinstance(wait_timeout_seconds, (int, float))
            or not math.isfinite(wait_timeout_seconds)
            or wait_timeout_seconds <= 0
        ):
            raise ValueError("wait_timeout_seconds must be a positive finite number")
        self.wait_timeout_seconds = float(wait_timeout_seconds)
        self._entries: dict[Hashable, _LockEntry] = {}

    @asynccontextmanager
    async def hold(self, key: Hashable) -> AsyncIterator[None]:
        entry = self._entries.get(key)
        if entry is None:
            entry = _LockEntry()
            self._entries[key] = entry

        # Count holders and waiters before the first await so cleanup cannot
        # remove an entry while another task is queued on the same lock.
        entry.references += 1
        acquired = False
        try:
            try:
                await asyncio.wait_for(
                    entry.lock.acquire(),
                    timeout=self.wait_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise UserLockBusy(
                    "Timed out waiting for the per-user request lock"
                ) from exc
            acquired = True
            try:
                yield
            finally:
                entry.lock.release()
        finally:
            entry.references -= 1
            if entry.references == 0 and self._entries.get(key) is entry:
                del self._entries[key]

    def clear(self) -> None:
        """Clear an idle registry; primarily useful for isolated tests."""
        if any(entry.references for entry in self._entries.values()):
            raise RuntimeError("cannot clear UserLockRegistry while locks are in use")
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def close(self) -> None:
        """Compatibility with the SQLite-backed registry; no resources are held."""


class SQLiteUserLockRegistry:
    """Serialize same-user requests across processes sharing one local SQLite DB.

    Each holder owns a random token and a renewable lease. A stale lease can be
    taken over after expiry; release is conditional on the token so an expired
    former owner cannot delete a newer owner's lock. If a heartbeat discovers
    that ownership was lost, it cancels the protected task rather than allowing
    two requests for the same user to continue concurrently.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        lease_seconds: float = 180.0,
        heartbeat_interval: float | None = None,
        poll_interval: float = 0.1,
        wait_timeout_seconds: float = 30.0,
    ) -> None:
        raw_path = str(database_path).strip()
        if not raw_path or raw_path == ":memory:":
            raise ValueError("SQLiteUserLockRegistry requires a persistent database path")
        for value, name in (
            (lease_seconds, "lease_seconds"),
            (poll_interval, "poll_interval"),
            (wait_timeout_seconds, "wait_timeout_seconds"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a positive finite number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")

        if heartbeat_interval is None:
            heartbeat_interval = min(30.0, float(lease_seconds) / 3.0)
        if (
            isinstance(heartbeat_interval, bool)
            or not isinstance(heartbeat_interval, (int, float))
            or not math.isfinite(heartbeat_interval)
            or heartbeat_interval <= 0
            or heartbeat_interval >= lease_seconds
        ):
            raise ValueError("heartbeat_interval must be positive and shorter than lease_seconds")

        path = Path(raw_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = str(path)
        self.lease_seconds = float(lease_seconds)
        self.heartbeat_interval = float(heartbeat_interval)
        self.poll_interval = float(poll_interval)
        self.wait_timeout_seconds = float(wait_timeout_seconds)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=5.0,
            isolation_level=None,
        )
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_request_locks (
                    user_id INTEGER PRIMARY KEY CHECK (user_id > 0),
                    owner_token TEXT NOT NULL,
                    lease_expires_at REAL NOT NULL
                )
                """
            )
        finally:
            connection.close()

    @staticmethod
    def _validate_user_id(user_id: int) -> int:
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("Telegram user_id must be a positive integer")
        return user_id

    def _try_acquire(self, user_id: int, owner_token: str) -> bool:
        now = time.time()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO user_request_locks (user_id, owner_token, lease_expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    owner_token = excluded.owner_token,
                    lease_expires_at = excluded.lease_expires_at
                WHERE user_request_locks.lease_expires_at <= ?
                """,
                (user_id, owner_token, now + self.lease_seconds, now),
            )
            acquired = cursor.rowcount == 1
            connection.commit()
            return acquired
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _renew(self, user_id: int, owner_token: str) -> bool:
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                UPDATE user_request_locks
                SET lease_expires_at = ?
                WHERE user_id = ? AND owner_token = ?
                """,
                (time.time() + self.lease_seconds, user_id, owner_token),
            )
            return cursor.rowcount == 1
        finally:
            connection.close()

    def _release(self, user_id: int, owner_token: str) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "DELETE FROM user_request_locks WHERE user_id = ? AND owner_token = ?",
                (user_id, owner_token),
            )
        finally:
            connection.close()

    async def _acquire(self, user_id: int, owner_token: str) -> None:
        deadline = time.monotonic() + self.wait_timeout_seconds
        while True:
            if time.monotonic() >= deadline:
                raise UserLockBusy("Timed out waiting for the per-user request lock")
            attempt = asyncio.create_task(
                asyncio.to_thread(self._try_acquire, user_id, owner_token)
            )
            try:
                acquired = await asyncio.shield(attempt)
            except asyncio.CancelledError:
                # to_thread cannot be force-stopped. Wait for its result and
                # release a lock if the SQLite operation won the race. Repeated
                # cancellation must not abandon that worker before cleanup ends.
                try:
                    acquired, _ = await _await_shielded_to_completion(attempt)
                    if acquired:
                        release_task = asyncio.create_task(
                            asyncio.to_thread(self._release, user_id, owner_token)
                        )
                        await _await_shielded_to_completion(release_task)
                except Exception:
                    logger.exception("Could not clean up a cancelled StockMind lock acquisition")
                raise
            except Exception as exc:
                raise UserLockUnavailable("Could not acquire the shared user lock") from exc

            if acquired:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise UserLockBusy("Timed out waiting for the per-user request lock")
            await asyncio.sleep(min(self.poll_interval, remaining))

    async def _heartbeat(self, user_id: int, owner_token: str, owner_task: asyncio.Task) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                renewed = await asyncio.to_thread(self._renew, user_id, owner_token)
            except Exception:
                logger.exception("Could not renew a StockMind per-user lock lease")
                owner_task.cancel(LEASE_LOST_CANCEL_MESSAGE)
                return
            if not renewed:
                logger.error("StockMind per-user lock lease was lost for user_id=%s", user_id)
                owner_task.cancel(LEASE_LOST_CANCEL_MESSAGE)
                return

    @asynccontextmanager
    async def hold(self, user_id: int) -> AsyncIterator[None]:
        user_id = self._validate_user_id(user_id)
        owner_token = uuid.uuid4().hex
        await self._acquire(user_id, owner_token)

        owner_task = asyncio.current_task()
        if owner_task is None:
            await asyncio.to_thread(self._release, user_id, owner_token)
            raise RuntimeError("SQLiteUserLockRegistry requires an asyncio task")

        heartbeat = asyncio.create_task(
            self._heartbeat(user_id, owner_token, owner_task)
        )
        try:
            yield
        finally:
            heartbeat.cancel()
            cleanup_cancellation = None
            try:
                _, heartbeat_cancellation = await _await_shielded_to_completion(
                    asyncio.gather(heartbeat, return_exceptions=True)
                )
                cleanup_cancellation = heartbeat_cancellation
            except Exception:
                logger.exception("Could not finish StockMind per-user lock heartbeat cleanup")

            release_task = asyncio.create_task(
                asyncio.to_thread(self._release, user_id, owner_token)
            )
            try:
                _, release_cancellation = await _await_shielded_to_completion(release_task)
                cleanup_cancellation = cleanup_cancellation or release_cancellation
            except Exception:
                logger.exception("Could not release a StockMind per-user lock")

            # If cancellation arrived while cleanup was running, propagate it
            # only after the worker has completed its conditional release.
            if cleanup_cancellation is not None:
                raise cleanup_cancellation

    def clear(self) -> None:
        """Remove expired rows only; active leases cannot be cleared."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT COUNT(*) FROM user_request_locks WHERE lease_expires_at > ?",
                (time.time(),),
            ).fetchone()[0]
            if active:
                connection.rollback()
                raise RuntimeError("cannot clear SQLiteUserLockRegistry while locks are active")
            connection.execute(
                "DELETE FROM user_request_locks WHERE lease_expires_at <= ?",
                (time.time(),),
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def __len__(self) -> int:
        connection = self._connect()
        try:
            return connection.execute(
                "SELECT COUNT(*) FROM user_request_locks WHERE lease_expires_at > ?",
                (time.time(),),
            ).fetchone()[0]
        finally:
            connection.close()

    def close(self) -> None:
        """Compatibility with other stores; this registry opens per-operation connections."""


def create_user_lock_registry(database_path: str | Path, **kwargs):
    """Use SQLite for persistent files and the in-memory registry for tests."""
    raw_path = str(database_path).strip()
    if not raw_path:
        raise ValueError("A non-empty SQLite database path is required")
    if raw_path == ":memory:":
        return UserLockRegistry(
            wait_timeout_seconds=kwargs.get("wait_timeout_seconds", 30.0)
        )
    return SQLiteUserLockRegistry(raw_path, **kwargs)


def is_lease_lost_cancellation(exc: asyncio.CancelledError) -> bool:
    """Distinguish lock-loss cancellation from normal bot shutdown cancellation."""
    return exc.args == (LEASE_LOST_CANCEL_MESSAGE,)
