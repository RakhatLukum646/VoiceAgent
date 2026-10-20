"""Single-process admission control: reserve before creating a room or opening ASR."""
import asyncio


class LinesBusy(Exception):
    pass


class ChannelPool:
    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("MAX_CONCURRENT_CALLS must be positive")
        self.limit = limit
        self._active: set[str] = set()
        self._lock = asyncio.Lock()

    async def acquire(self, call_id: str):
        async with self._lock:
            if call_id in self._active:
                raise ValueError("Duplicate channel reservation")
            if len(self._active) >= self.limit:
                raise LinesBusy("Все линии заняты. Попробуйте немного позже.")
            self._active.add(call_id)

    async def release(self, call_id: str):
        async with self._lock:
            self._active.discard(call_id)

    def snapshot(self):
        return {"limit": self.limit, "active": len(self._active), "available": self.limit - len(self._active)}
