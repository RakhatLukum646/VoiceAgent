"""Gemini Live transcription over documented PCM WebSocket protocol."""
import asyncio
import base64
import json
from urllib.parse import quote
import websockets
from .base import ASRProvider


LIVE_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"


class GoogleLiveASR(ASRProvider):
    """https://ai.google.dev/gemini-api/docs/live-api/live-transcribe"""

    def __init__(self, api_key: str, model: str = "gemini-3.5-transcribe-live", language: str = "ru-RU"):
        language = {"ru": "ru-RU", "kk": "kk-KZ"}.get(language, language)
        self.api_key, self.model, self.language = api_key, model, language
        self.ws = None
        self.reader: asyncio.Task | None = None
        self.on_error = None
        self.on_partial = None
        self.on_final = None
        self.listening = False
        self.finalize_requested = False
        self.finalized_once = False
        self.pending_finals: list[str] = []
        self._send_lock = asyncio.Lock()
        self._final_timer: asyncio.Task | None = None
        self._final_watchdog: asyncio.Task | None = None
        self._active_callback_tasks: set[asyncio.Task] = set()
        self._turn_complete = asyncio.Event()
        self._turn_complete.set()
        self._audio_buffer = bytearray()
        self._turn_seq = 0
        self._needs_reconnect = False
        self._closing = False
        self._discard_until_audio = False

    async def start(self, on_partial, on_final):
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY required for live transcription")
        self.on_partial, self.on_final = on_partial, on_final
        self._closing = False
        url = f"{LIVE_URL}?key={quote(self.api_key, safe='')}"
        try:
            self.ws = await websockets.connect(url, open_timeout=10, max_size=4 * 1024 * 1024)
            setup = {"setup": {"model": f"models/{self.model}", "generationConfig": {"responseModalities": ["TEXT"]}, "inputAudioTranscription": {"languageCodes": [self.language] if self.language else []}}}
            await self.ws.send(json.dumps(setup))
            response = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=10))
            if "setupComplete" not in response:
                raise RuntimeError("Live transcription setup was rejected")
        except Exception:
            if self.ws is not None:
                await self.ws.close()
            self.ws = None
            raise RuntimeError("Google Live transcription connection or setup failed") from None
        self.reader = asyncio.create_task(self._read())

    async def begin_utterance(self):
        if self.listening:
            return
        try:
            await asyncio.wait_for(self._turn_complete.wait(), timeout=5.5)
        except TimeoutError:
            self._needs_reconnect = True
        if self._needs_reconnect:
            await self._restart()
        self._turn_seq += 1
        self.listening = True
        self.finalize_requested = False
        self.finalized_once = False
        self.pending_finals.clear()
        self._audio_buffer.clear()
        self._discard_until_audio = True
        self._turn_complete.clear()

    async def _restart(self):
        partial, final = self.on_partial, self.on_final
        await self.close()
        await self.start(partial, final)
        self._needs_reconnect = False

    async def send_audio(self, pcm: bytes):
        if not self.listening:
            return
        if self.ws is None:
            raise RuntimeError("Google Live transcription stream unavailable")
        self._audio_buffer.extend(pcm)
        self._discard_until_audio = False
        if len(self._audio_buffer) < 3200:
            return
        await self._send_buffer()

    async def _send_buffer(self):
        if not self._audio_buffer:
            return
        message = {"realtimeInput": {"audio": {"data": base64.b64encode(self._audio_buffer).decode("ascii"), "mimeType": "audio/pcm;rate=16000"}}}
        self._audio_buffer.clear()
        async with self._send_lock:
            await self.ws.send(json.dumps(message))

    async def finalize(self):
        if not self.listening or self.finalize_requested:
            return
        self.finalize_requested = True
        self.listening = False
        if self.ws is None:
            return
        await self._send_buffer()
        async with self._send_lock:
            await self.ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
        if self.pending_finals:
            self._schedule_final()
        else:
            self._final_watchdog = asyncio.create_task(self._wait_for_final(self._turn_seq))

    async def _wait_for_final(self, turn_seq: int):
        await asyncio.sleep(5)
        if turn_seq == self._turn_seq and not self._turn_complete.is_set():
            self._needs_reconnect = True
            try:
                await self.on_final("")
            finally:
                if turn_seq == self._turn_seq:
                    self._turn_complete.set()

    def _schedule_final(self):
        if self._final_timer and not self._final_timer.done() and self._final_timer not in self._active_callback_tasks:
            self._final_timer.cancel()
        self._final_timer = asyncio.create_task(self._settle_final(self._turn_seq))

    async def _settle_final(self, turn_seq: int):
        await asyncio.sleep(0.3)
        task = asyncio.current_task()
        self._active_callback_tasks.add(task)
        try:
            await self._emit_final(turn_seq)
        finally:
            self._active_callback_tasks.discard(task)

    async def _emit_final(self, turn_seq: int):
        if turn_seq != self._turn_seq or self.finalized_once:
            return
        text = " ".join(self.pending_finals).strip()
        self.pending_finals.clear()
        if self._final_watchdog and not self._final_watchdog.done():
            self._final_watchdog.cancel()
        try:
            if text:
                self.finalized_once = True
                self._turn_complete.set()
                await self.on_final(text)
        finally:
            if turn_seq == self._turn_seq:
                self._turn_complete.set()

    async def _read(self):
        failure = None
        try:
            async for raw in self.ws:
                response = json.loads(raw)
                if response.get("error"):
                    raise RuntimeError("Google Live transcription provider error")
                content = response.get("serverContent") or {}
                interim = content.get("interimInputTranscription") or {}
                if interim.get("text") and not self._discard_until_audio:
                    await self.on_partial(interim["text"])
                final = content.get("inputTranscription") or {}
                if final.get("text") and not self._discard_until_audio and not self.finalized_once:
                    self.pending_finals.append(final["text"])
                    if self.finalize_requested:
                        self._schedule_final()
            if not self._closing:
                failure = RuntimeError("Google Live transcription stream closed")
        except asyncio.CancelledError:
            raise
        except Exception:
            failure = RuntimeError("Google Live transcription stream failed")
        finally:
            if failure:
                self.listening = False
                self._needs_reconnect = True
                failed_socket = self.ws
                self.ws = None
                self._turn_complete.set()
                if failed_socket is not None:
                    try:
                        await failed_socket.close()
                    except Exception:
                        pass
                if self.on_error:
                    await self.on_error(failure)

    async def close(self):
        self._closing = True
        if self._final_watchdog:
            self._final_watchdog.cancel()
            await asyncio.gather(self._final_watchdog, return_exceptions=True)
        if self._final_timer:
            self._final_timer.cancel()
            await asyncio.gather(self._final_timer, return_exceptions=True)
        for task in self._active_callback_tasks:
            task.cancel()
        await asyncio.gather(*self._active_callback_tasks, return_exceptions=True)
        self._active_callback_tasks.clear()
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.ws:
            await self.ws.close()
        self.ws = None
        self._turn_complete.set()
