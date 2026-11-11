"""Cross-instance tests for SQLite-backed per-user request leases."""

import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path

from stockmind_telegram_mvp.app.user_locks import (
    LEASE_LOST_CANCEL_MESSAGE,
    SQLiteUserLockRegistry,
    UserLockRegistry,
    create_user_lock_registry,
)


class SQLiteUserLockRegistryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "shared.sqlite3"
        self.first = SQLiteUserLockRegistry(
            self.database_path,
            lease_seconds=0.3,
            heartbeat_interval=0.05,
            poll_interval=0.005,
        )
        self.second = SQLiteUserLockRegistry(
            self.database_path,
            lease_seconds=0.3,
            heartbeat_interval=0.05,
            poll_interval=0.005,
        )

    def tearDown(self):
        self.first.close()
        self.second.close()
        self.temp_dir.cleanup()

    async def test_same_user_is_serialized_across_independent_registry_instances(self):
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_entered = asyncio.Event()
        order = []

        async def first_request():
            async with self.first.hold(1001):
                order.append("first-enter")
                first_entered.set()
                await release_first.wait()
                order.append("first-exit")

        async def second_request():
            await first_entered.wait()
            async with self.second.hold(1001):
                order.append("second-enter")
                second_entered.set()

        first_task = asyncio.create_task(first_request())
        await asyncio.wait_for(first_entered.wait(), timeout=1)
        second_task = asyncio.create_task(second_request())
        await asyncio.sleep(0.04)

        self.assertFalse(second_entered.is_set())
        self.assertEqual(len(self.first), 1)

        release_first.set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=2)
        self.assertEqual(order, ["first-enter", "first-exit", "second-enter"])
        self.assertEqual(len(self.first), 0)

    async def test_different_users_can_run_concurrently_across_instances(self):
        both_entered = asyncio.Event()
        entered = set()

        async def request(registry, user_id):
            async with registry.hold(user_id):
                entered.add(user_id)
                if len(entered) == 2:
                    both_entered.set()
                await both_entered.wait()

        await asyncio.wait_for(
            asyncio.gather(
                request(self.first, 1001),
                request(self.second, 1002),
            ),
            timeout=2,
        )
        self.assertEqual(entered, {1001, 1002})
        self.assertEqual(len(self.first), 0)

    async def test_heartbeat_prevents_takeover_during_a_long_request(self):
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_entered = asyncio.Event()

        async def first_request():
            async with self.first.hold(1001):
                first_entered.set()
                await release_first.wait()

        async def second_request():
            await first_entered.wait()
            async with self.second.hold(1001):
                second_entered.set()

        first_task = asyncio.create_task(first_request())
        await asyncio.wait_for(first_entered.wait(), timeout=1)
        second_task = asyncio.create_task(second_request())

        # Wait beyond the original 0.3-second lease. Heartbeats must keep the
        # first owner alive, so the second registry must remain queued.
        await asyncio.sleep(0.38)
        self.assertFalse(second_entered.is_set())

        release_first.set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=2)
        self.assertTrue(second_entered.is_set())

    async def test_expired_crashed_owner_lease_can_be_recovered(self):
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                """
                INSERT INTO user_request_locks (user_id, owner_token, lease_expires_at)
                VALUES (?, ?, 0)
                """,
                (1001, "stale-process-token"),
            )
            connection.commit()
        finally:
            connection.close()

        async with self.second.hold(1001):
            self.assertEqual(len(self.first), 1)

        self.assertEqual(len(self.first), 0)

    async def test_lost_owner_is_cancelled_and_cannot_release_new_owners_lock(self):
        first_entered = asyncio.Event()
        first_finished = asyncio.Event()
        second_entered = asyncio.Event()
        release_second = asyncio.Event()

        # Use a slower heartbeat here so the forced expiry and takeover are
        # deterministic before the former owner's next renewal attempt.
        first_registry = SQLiteUserLockRegistry(
            self.database_path,
            lease_seconds=1.0,
            heartbeat_interval=0.5,
            poll_interval=0.005,
        )
        second_registry = SQLiteUserLockRegistry(
            self.database_path,
            lease_seconds=1.0,
            heartbeat_interval=0.5,
            poll_interval=0.005,
        )

        async def first_request():
            try:
                async with first_registry.hold(1001):
                    first_entered.set()
                    await asyncio.sleep(5)
            finally:
                first_finished.set()

        async def second_request():
            async with second_registry.hold(1001):
                second_entered.set()
                await release_second.wait()

        first_task = asyncio.create_task(first_request())
        await asyncio.wait_for(first_entered.wait(), timeout=1)

        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                "UPDATE user_request_locks SET lease_expires_at = 0 WHERE user_id = ?",
                (1001,),
            )
            connection.commit()
        finally:
            connection.close()

        second_task = asyncio.create_task(second_request())
        await asyncio.wait_for(second_entered.wait(), timeout=1)
        with self.assertRaises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(first_task, timeout=1)
        self.assertEqual(caught.exception.args, (LEASE_LOST_CANCEL_MESSAGE,))
        self.assertTrue(first_finished.is_set())

        # The former owner's conditional release must not delete the new lease.
        self.assertEqual(len(second_registry), 1)
        release_second.set()
        await asyncio.wait_for(second_task, timeout=1)
        self.assertEqual(len(second_registry), 0)

    async def test_cancelled_waiter_does_not_release_active_owner(self):
        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        async def first_request():
            async with self.first.hold(1001):
                first_entered.set()
                await release_first.wait()

        async def waiting_request():
            async with self.second.hold(1001):
                self.fail("cancelled waiter must not acquire the active user's lock")

        first_task = asyncio.create_task(first_request())
        await asyncio.wait_for(first_entered.wait(), timeout=1)
        waiter_task = asyncio.create_task(waiting_request())
        await asyncio.sleep(0.03)
        waiter_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter_task

        self.assertEqual(len(self.first), 1)
        release_first.set()
        await asyncio.wait_for(first_task, timeout=1)
        self.assertEqual(len(self.first), 0)

    def test_rejects_invalid_user_ids_and_lease_configuration(self):
        for invalid_user_id in (True, 0, -1, "1001"):
            with self.subTest(user_id=invalid_user_id):
                with self.assertRaises(ValueError):
                    self.first._validate_user_id(invalid_user_id)

        with self.assertRaisesRegex(ValueError, "heartbeat_interval"):
            SQLiteUserLockRegistry(
                self.database_path,
                lease_seconds=1,
                heartbeat_interval=1,
            )

    def test_factory_keeps_in_memory_test_mode_process_local(self):
        self.assertIsInstance(create_user_lock_registry(":memory:"), UserLockRegistry)


if __name__ == "__main__":
    unittest.main()
