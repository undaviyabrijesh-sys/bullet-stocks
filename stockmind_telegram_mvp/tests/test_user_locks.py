"""Tests for per-user async lock lifecycle and isolation."""

import asyncio
import unittest

from stockmind_telegram_mvp.app.user_locks import UserLockRegistry


class UserLockRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_key_waiters_share_entry_until_all_exit(self):
        registry = UserLockRegistry()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_started = asyncio.Event()

        async def first():
            async with registry.hold("user-a"):
                first_entered.set()
                await release_first.wait()

        async def second():
            second_started.set()
            async with registry.hold("user-a"):
                pass

        first_task = asyncio.create_task(first())
        await asyncio.wait_for(first_entered.wait(), timeout=1)
        second_task = asyncio.create_task(second())
        await asyncio.wait_for(second_started.wait(), timeout=1)
        await asyncio.sleep(0)

        self.assertEqual(len(registry), 1)
        self.assertEqual(registry._entries["user-a"].references, 2)

        release_first.set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=1)
        self.assertEqual(len(registry), 0)

    async def test_different_keys_can_run_concurrently(self):
        registry = UserLockRegistry()
        both_entered = asyncio.Event()
        entered = set()

        async def work(key):
            async with registry.hold(key):
                entered.add(key)
                if len(entered) == 2:
                    both_entered.set()
                await both_entered.wait()

        await asyncio.wait_for(
            asyncio.gather(work("user-a"), work("user-b")),
            timeout=1,
        )
        self.assertEqual(entered, {"user-a", "user-b"})
        self.assertEqual(len(registry), 0)

    async def test_cancelled_waiter_releases_its_registry_reference(self):
        registry = UserLockRegistry()
        holder_entered = asyncio.Event()
        release_holder = asyncio.Event()

        async def holder():
            async with registry.hold("user-a"):
                holder_entered.set()
                await release_holder.wait()

        async def waiter():
            async with registry.hold("user-a"):
                self.fail("cancelled waiter must not acquire the lock")

        holder_task = asyncio.create_task(holder())
        await asyncio.wait_for(holder_entered.wait(), timeout=1)
        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        self.assertEqual(registry._entries["user-a"].references, 2)

        waiter_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter_task
        self.assertEqual(registry._entries["user-a"].references, 1)

        release_holder.set()
        await asyncio.wait_for(holder_task, timeout=1)
        self.assertEqual(len(registry), 0)

    async def test_clearing_an_active_registry_is_rejected(self):
        registry = UserLockRegistry()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def work():
            async with registry.hold("user-a"):
                entered.set()
                await release.wait()

        task = asyncio.create_task(work())
        await asyncio.wait_for(entered.wait(), timeout=1)
        with self.assertRaises(RuntimeError):
            registry.clear()

        release.set()
        await asyncio.wait_for(task, timeout=1)
        registry.clear()
        self.assertEqual(len(registry), 0)


if __name__ == "__main__":
    unittest.main()
