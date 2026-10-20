from abc import ABC, abstractmethod


class CallTransport(ABC):
    @abstractmethod
    async def event(self, payload: dict) -> None: ...

    @abstractmethod
    async def audio(self, pcm: bytes, generation: int) -> None: ...


class BrowserWebSocketTransport(CallTransport):
    def __init__(self, websocket):
        self.websocket = websocket
        self._send_lock = __import__("asyncio").Lock()

    async def event(self, payload: dict) -> None:
        async with self._send_lock:
            await self.websocket.send_json(payload)

    async def audio(self, pcm: bytes, generation: int) -> None:
        # Binary wire format: 4-byte little endian generation followed by PCM16 16 kHz mono.
        async with self._send_lock:
            await self.websocket.send_bytes(generation.to_bytes(4, "little") + pcm)
