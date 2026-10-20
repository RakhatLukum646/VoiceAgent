import asyncio
import base64
import json
import websockets
from .base import TTSProvider


class DeepdubTTSProvider(TTSProvider):
    """Official streaming contract: https://docs.deepdub.ai/api-reference/websocket/streaming

    One WebSocket is opened lazily and kept for the lifetime of the provider
    (one call), configured once, then reused turn after turn (`stream-text` +
    `end-stream`, await `isFinal`) instead of reconnecting every utterance.
    A lock makes turn ownership exclusive: only one logical turn is ever in
    flight on the wire. A cancelled turn is drained to `isCancelled` before the
    lock is released, so its trailing audio cannot bleed into the next turn,
    and the connection survives a cancel to be reused by the next turn.
    """

    def __init__(self, api_key: str, voice_id: str, base_url: str, locale: str = "ru-RU"):
        self.api_key, self.voice_id, self.base_url, self.locale = api_key, voice_id, base_url, locale
        self._conn = None
        self._ws = None
        self._lock = asyncio.Lock()

    async def _connected(self):
        if self._ws is not None:
            return self._ws
        if not all((self.api_key, self.voice_id, self.base_url)):
            raise RuntimeError("Deepdub requires DEEPDUB_API_KEY, DEEPDUB_VOICE_ID and DEEPDUB_BASE_URL")
        conn = websockets.connect(self.base_url, additional_headers={"x-api-key": self.api_key}, max_size=8 * 1024 * 1024)
        ws = await conn.__aenter__()
        try:
            welcome = json.loads(await asyncio.wait_for(ws.recv(), 10))
            if welcome.get("action") != "status" or welcome.get("message") != "connected":
                raise RuntimeError(f"Deepdub handshake failed: {welcome}")
            await ws.send(json.dumps({"action": "stream-config", "data": {"model": "dd-etts-3.0", "voicePromptId": self.voice_id, "locale": self.locale, "format": "s16le", "sampleRate": 16000, "realtime": True}}))
        except Exception:
            await conn.__aexit__(None, None, None)
            raise
        self._conn, self._ws = conn, ws
        return ws

    async def synthesize_stream(self, text: str):
        async with self._lock:
            try:
                ws = await self._connected()
                await ws.send(json.dumps({"action": "stream-text", "data": {"text": text}}))
                await ws.send(json.dumps({"action": "end-stream"}))
            except (websockets.exceptions.ConnectionClosed, OSError):
                await self._reset()
                raise
            try:
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 40))
                    if msg.get("action") == "ping":
                        await ws.send(json.dumps({"action": "pong"}))
                        continue
                    if msg.get("error") or msg.get("action") == "error":
                        raise RuntimeError(f"Deepdub error: {msg.get('error') or msg.get('message')}")
                    data = msg.get("data")
                    if isinstance(data, str) and data:
                        yield base64.b64decode(data, validate=True)
                    if msg.get("isFinal"):
                        break
            except asyncio.CancelledError:
                await self._cancel_turn(ws)
                raise
            except (websockets.exceptions.ConnectionClosed, OSError, RuntimeError):
                await self._reset()
                raise

    async def _cancel_turn(self, ws):
        # A clean `isCancelled` drain does not prove the server's turn state is
        # safe to reuse (observed: next turn on the same socket got an
        # immediate empty/isFinal response after a cancel). Always reconnect
        # after any cancellation rather than trusting the post-cancel socket.
        try:
            await asyncio.wait_for(ws.send(json.dumps({"action": "cancel"})), 1)
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 2))
                if msg.get("isCancelled") or msg.get("isFinal"):
                    break
        except Exception:
            pass
        finally:
            await self._reset()

    async def _reset(self):
        conn, self._conn, self._ws = self._conn, None, None
        if conn is not None:
            try:
                await conn.__aexit__(None, None, None)
            except Exception:
                pass

    async def close(self):
        async with self._lock:
            await self._reset()
