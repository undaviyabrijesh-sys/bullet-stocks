"""Process-local per-key async locks with automatic idle-entry cleanup.

The registry is intended for use from one asyncio event loop (the bot's polling
loop). Entry bookkeeping contains no await points, so acquisition references
are updated atomically relative to other tasks on that loop.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Hashable


class _LockEntry:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.references = 0


class UserLockRegistry:
    """Serialize work per key without retaining idle locks indefinitely."""

    def __init__(self) -> None:
        self._entries: dict[Hashable, _LockEntry] = {}

    @asynccontextmanager
    async def hold(self, key: Hashable) -> AsyncIterator[None]:
        entry = self._entries.get(key)
        if entry is None:
            entry = _LockEntry()
            self._entries[key] = entry

        # Count both the holder and tasks waiting to acquire this same lock.
        # This increment happens before the first await, so no waiter can be
        # missed when the active holder releases the entry.
        entry.references += 1
        try:
            async with entry.lock:
                yield
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
