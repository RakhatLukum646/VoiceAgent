import asyncio
import json
from time import perf_counter
from urllib.parse import urlencode
import websockets
from .base import ASRProvider


class DeepgramASR(ASRProvider):
    """Live API: https://developers.deepgram.com/reference/speech-to-text/listen-streaming"""

    def __init__(self, api_key: str, model: str = "nova-3", language: str = "ru"):
        self.api_key, self.model, self.language = api_key, model, language
        self.ws = None
        self.reader: asyncio.Task | None = None
        self.segments: list[str] = []
        self.keepalive: asyncio.Task | None = None
        self.last_audio = perf_counter()
        self.on_error = None

    async def start(self, on_partial, on_final):
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY required")
        query = urlencode({"model": self.model, "language": self.language, "encoding": "linear16", "sample_rate": 16000, "channels": 1, "interim_results": "true", "endpointing": 300, "utterance_end_ms": 1000})
        self.ws = await websockets.connect(f"wss://api.deepgram.com/v1/listen?{query}", additional_headers={"Authorization": f"Token {self.api_key}"}, max_size=4 * 1024 * 1024)
        self.reader = asyncio.create_task(self._read(on_partial, on_final))
        self.keepalive = asyncio.create_task(self._keepalive())

    async def _keepalive(self):
        try:
            while self.ws is not None:
                await asyncio.sleep(3)
                if self.ws is not None and perf_counter() - self.last_audio > 3:
                    await self.ws.send(json.dumps({"type": "KeepAlive"}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.on_error:
                await self.on_error(exc)

    async def _read(self, on_partial, on_final):
        try:
            async for raw in self.ws:
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "Error":
                    raise RuntimeError(f"Deepgram: {message}")
                if kind == "Results":
                    text = message.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "").strip()
                    if text and message.get("is_final"):
                        self.segments.append(text)
                    elif text:
                        await on_partial(" ".join([*self.segments, text]))
                    if message.get("speech_final") or message.get("from_finalize"):
                        await self._emit(on_final)
                elif kind == "UtteranceEnd":
                    await self._emit(on_final)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.on_error:
                await self.on_error(exc)
        finally:
            self.ws = None

    async def _emit(self, callback):
        text = " ".join(self.segments).strip()
        self.segments.clear()
        if text:
            await callback(text)

    async def send_audio(self, pcm: bytes):
        if self.ws is None:
            raise RuntimeError("Deepgram stream unavailable")
        self.last_audio = perf_counter()
        await self.ws.send(pcm)

    async def finalize(self):
        if self.ws is not None:
            await self.ws.send(json.dumps({"type": "Finalize"}))

    async def close(self):
        if self.keepalive:
            self.keepalive.cancel()
            await asyncio.gather(self.keepalive, return_exceptions=True)
        if self.ws is not None:
            await self.ws.send(json.dumps({"type": "CloseStream"}))
            await self.ws.close()
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
