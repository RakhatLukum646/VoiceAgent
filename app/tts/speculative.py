"""Private bounded PCM. A candidate must be explicitly approved before iteration.

The gate belongs to exactly one response task; scope identity includes call,
turn, cancellation generation and candidate digest. No transport exists here.
"""
import asyncio
import hashlib
from time import perf_counter

_SYNTHESIS_SLOTS = asyncio.Semaphore(3)


class CandidateAudio:
    def __init__(self, tts, call_id, turn_id, generation, max_bytes=384000):
        self.tts = tts
        self.identity = (call_id, turn_id, generation)
        self.max_bytes = max_bytes
        self.queue = asyncio.Queue(maxsize=max(1, max_bytes // 3200))
        self.task = None
        self.key = None
        self.approved = False
        self.error = None
        self.produced = 0
        self.first_audio_ms = None
        self._started = 0.0

    def binding(self, text, attempt):
        return self.identity + (attempt, hashlib.sha256(text.encode()).hexdigest())

    async def start(self, text, attempt):
        await self.discard()
        self.key = self.binding(text, attempt)
        self.approved = False
        self.error = None
        self.produced = 0
        self.first_audio_ms = None
        self._started = perf_counter()
        self.task = asyncio.create_task(self._produce(text))
        return self.key

    async def _produce(self, text):
        try:
            async with asyncio.timeout(35):
                async with _SYNTHESIS_SLOTS:
                    async for chunk in self.tts.synthesize_stream(text):
                        if not chunk:
                            continue
                        if self.first_audio_ms is None:
                            self.first_audio_ms = round((perf_counter() - self._started) * 1000, 2)
                        for offset in range(0, len(chunk), 3200):
                            part = chunk[offset:offset + 3200]
                            self.produced += len(part)
                            if self.produced > 16000 * 2 * 35:
                                raise RuntimeError("Speculative speech exceeds duration cap")
                            await self.queue.put(part)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = exc
        finally:
            # No blocking sentinel put: consumer polls task completion too.
            pass

    def approve(self, key, text):
        if key != self.key or key[-1] != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError("Candidate audio approval mismatch")
        self.approved = True

    async def stream(self, text, identity):
        if not self.approved or identity != self.identity or not self.key or self.key[-1] != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError("Unapproved candidate audio")
        while True:
            if not self.queue.empty():
                yield self.queue.get_nowait()
                continue
            if self.task.done():
                if self.error:
                    raise RuntimeError("Speculative TTS failed") from self.error
                break
            try:
                yield await asyncio.wait_for(self.queue.get(), .1)
            except asyncio.TimeoutError:
                continue

    async def discard(self):
        self.approved = False
        if self.task and not self.task.done():
            self.task.cancel()
        if self.task:
            await asyncio.gather(self.task, return_exceptions=True)
        self.task = None
        while not self.queue.empty():
            self.queue.get_nowait()
        self.key = None
