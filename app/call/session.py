import asyncio
import uuid
from collections import deque
from time import perf_counter
from app.audio.vad import RMSVAD
from app.observability.latency import TurnMetrics
from app.observability.logging import event
from app.observability.errors import safe_error_fields
from app.call.utterances import is_contentful, is_filler, is_stop
from app.tts.speculative import CandidateAudio


class CallSession:
    ACKNOWLEDGEMENT = "Сейчас проверю."
    CLARIFICATION = "Пожалуйста, уточните ваш вопрос об оформлении или получении Kaspi Gold."
    CORRECTION_BRIDGE = "Секунду, уточню. "
    def __init__(self, transport, asr, orchestrator, tts, on_metrics=None, speak_first=False):
        self.call_id = str(uuid.uuid4())
        self.transport, self.asr, self.orchestrator, self.tts = transport, asr, orchestrator, tts
        self.on_metrics = on_metrics
        # Opt-in only: default False keeps the existing never-speak-unverified
        # guarantee (and tests exercising it) unchanged for any caller that
        # does not explicitly request the speak-first/correct-after trade-off.
        self.speak_first = speak_first
        self.vad = RMSVAD()
        self.turn_id = 0
        self.generation = 0
        self.history: list[dict] = []
        self.task: asyncio.Task | None = None
        self.state = "connected"
        self.closed = False
        self.terminal = False
        self.speech_start: float | None = None
        self.vad_end: float | None = None
        self.first_partial: float | None = None
        self._pending_turn: dict | None = None
        self._barge_pending = False
        self._last_partial_text = ""
        self._prefetch: tuple[str, object] | None = None
        self.playback_done = asyncio.Event()
        self._interrupt_lock = asyncio.Lock()
        self.max_bytes = 16000 * 2 * 30
        self.turn_bytes = 0
        self.preroll: deque[bytes] = deque(maxlen=12)
        self._asr_begin_task: asyncio.Task | None = None
        self._asr_audio_lock = asyncio.Lock()
        self._asr_pending_audio = bytearray()
        self._asr_pending_finalize = False
        self._asr_ready = False
        self._asr_pending_limit = 16000 * 2 * 30

    async def start(self):
        self.asr.on_error = self._asr_error
        await self.asr.start(self._partial, self._final)
        if hasattr(self.tts, "warm"):
            # Opens the TTS connection while the call is still being set up,
            # so the very first utterance doesn't pay handshake latency on
            # top of synthesis time.
            asyncio.create_task(self.tts.warm())
        await self.transport.event({"type": "connected", "call_id": self.call_id, "generation": self.generation})

    async def _asr_error(self, exc: Exception):
        event("asr_error", self.call_id, self.turn_id, at_monotonic=perf_counter(), **safe_error_fields(exc))
        if not self.closed:
            self.vad.speaking = False
            self.vad.silence_ms = 0
            self._pending_turn = None
            self._barge_pending = False
            await self.transport.event({"type": "error", "message": "Сервис распознавания речи недоступен"})
            await self.interrupt()

    async def feed_audio(self, pcm: bytes):
        if self.closed or self.terminal:
            return
        if len(pcm) > 32000 or len(pcm) % 2:
            raise ValueError("Invalid PCM frame")
        was_speaking = self.vad.speaking
        if not was_speaking:
            self.preroll.append(pcm)
        started, ended = self.vad.feed(pcm)
        if started:
            if hasattr(self.asr, "begin_utterance"):
                if self._asr_begin_task is None or self._asr_begin_task.done():
                    self._asr_ready = False
                    self._asr_begin_task = asyncio.create_task(self._begin_asr_and_flush())
                elif self._asr_pending_finalize:
                    # The prior queued utterance ended before ASR opened. Keep the
                    # newest speech prefix instead of merging separate turns.
                    self._asr_pending_audio.clear()
                    event("asr_pending_replaced", self.call_id, self.turn_id, generation=self.generation)
                self._asr_pending_finalize = False
            self.speech_start = perf_counter()
            self.first_partial = None
            self._last_partial_text = ""
            self.turn_bytes = 0
            self._barge_pending = self.task is not None and not self.task.done()
            if not self._barge_pending:
                await self._state("listening")
        if self.vad.speaking:
            self.turn_bytes += len(pcm)
        if self.turn_bytes > self.max_bytes:
            await self._finalize_asr()
            self.turn_bytes = 0
            self.vad.speaking = False
            self.vad.silence_ms = 0
            self.preroll.clear()
        if started and hasattr(self.asr, "begin_utterance"):
            for buffered in self.preroll:
                await self._send_asr(buffered)
            self.preroll.clear()
        else:
            await self._send_asr(pcm)
        if ended:
            self.vad_end = perf_counter()
            self._pending_turn = {"speech_start": self.speech_start, "vad_end": self.vad_end, "last_voice_at": self.vad.last_voice_at, "first_partial": self.first_partial}
            if not (self.task and not self.task.done()):
                await self._state("finalizing")
            self._start_prefetch()
            await self._finalize_asr()

    def _start_prefetch(self):
        """Speculatively warm retrieval (embedding + Qdrant search) the moment
        VAD confirms the caller stopped talking, using the latest ASR partial
        text instead of waiting for the ASR provider's own finalization
        round-trip (which `asr_finalization` metrics show as real dead time).
        Read-only and side-effect-free: `prepare()` only starts background
        search tasks, never touches turn/generation bookkeeping or audio, so a
        wrong guess just gets closed and discarded in `_final` with no risk to
        the spoken answer.
        """
        if self.task and not self.task.done():
            return
        text = self._last_partial_text.strip()
        if not text or not is_contentful(text) or is_stop(text) or is_filler(text):
            return
        if not hasattr(self.orchestrator, "prepare"):
            return
        if self._prefetch is not None:
            asyncio.create_task(self._prefetch[1].close())
            self._prefetch = None
        try:
            prepared = self.orchestrator.prepare(text, self.history.copy())
        except Exception:
            return
        if prepared is not None:
            self._prefetch = (text, prepared)

    async def _begin_asr_and_flush(self):
        try:
            await self.asr.begin_utterance()
            async with self._asr_audio_lock:
                self._asr_ready = True
                while self._asr_pending_audio:
                    chunk = bytes(self._asr_pending_audio[:3200])
                    del self._asr_pending_audio[:3200]
                    await self.asr.send_audio(chunk)
                if self._asr_pending_finalize:
                    self._asr_pending_finalize = False
                    self._asr_ready = False
                    await self.asr.finalize()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._asr_ready = False
            self._asr_pending_audio.clear()
            await self._asr_error(exc)

    async def _send_asr(self, pcm: bytes):
        if not hasattr(self.asr, "begin_utterance"):
            await self.asr.send_audio(pcm)
            return
        async with self._asr_audio_lock:
            if not self._asr_ready:
                if self._asr_begin_task is None or self._asr_begin_task.done():
                    return
                if self._asr_pending_finalize:
                    return
                if len(self._asr_pending_audio) + len(pcm) > self._asr_pending_limit:
                    available = self._asr_pending_limit - len(self._asr_pending_audio)
                    self._asr_pending_audio.extend(pcm[:available & ~1])
                    self._asr_pending_finalize = True
                    event("asr_pending_audio_overflow", self.call_id, self.turn_id, generation=self.generation, buffered_bytes=len(self._asr_pending_audio))
                    return
                self._asr_pending_audio.extend(pcm)
            else:
                await self.asr.send_audio(pcm)

    async def _finalize_asr(self):
        if not hasattr(self.asr, "begin_utterance"):
            await self.asr.finalize()
            return
        async with self._asr_audio_lock:
            if self._asr_ready:
                self._asr_ready = False
                await self.asr.finalize()
            else:
                self._asr_pending_finalize = True

    async def _partial(self, transcript: str):
        if self.closed or self.terminal:
            return
        self._last_partial_text = transcript
        meaningful = is_contentful(transcript)
        if meaningful and self.first_partial is None:
            self.first_partial = perf_counter()
        if self._barge_pending and meaningful:
            self._barge_pending = False
            if self.task and not self.task.done():
                await self.interrupt()
                await self._state("listening")
        await self.transport.event({"type": "partial", "text": transcript, "generation": self.generation})

    async def _final(self, transcript: str):
        if self.closed or self.terminal:
            return
        prefetched = None
        if self._prefetch is not None:
            prefetch_text, prefetch_prepared = self._prefetch
            self._prefetch = None
            if " ".join(prefetch_text.split()) == " ".join(transcript.split()):
                prefetched = prefetch_prepared
            else:
                asyncio.create_task(prefetch_prepared.close())
        if not transcript.strip():
            if prefetched:
                asyncio.create_task(prefetched.close())
            if self._pending_turn:
                had_meaningful_partial = self._pending_turn["first_partial"] is not None
                self._pending_turn = None
                self._barge_pending = False
                if not (self.task and not self.task.done()) and (had_meaningful_partial or self.turn_id == 0):
                    await self.transport.event({"type": "error", "generation": self.generation, "message": "Речь не распознана, повторите вопрос"})
                    await self._state("listening")
            return
        if is_stop(transcript):
            if prefetched:
                asyncio.create_task(prefetched.close())
            self._pending_turn = None
            self._barge_pending = False
            await self.interrupt()
            await self._state("listening")
            return
        if is_filler(transcript):
            if prefetched:
                asyncio.create_task(prefetched.close())
            self._pending_turn = None
            self._barge_pending = False
            await self.transport.event({"type": "discard_partial", "generation": self.generation})
            if not (self.task and not self.task.done()):
                self.playback_done = asyncio.Event()
                self.task = asyncio.create_task(self._speak_clarification(self.generation))
            return
        # ASR reader never waits for RAG/TTS, so speech remains detectable during playback.
        prior = self._pending_turn or {"speech_start": self.speech_start, "vad_end": self.vad_end, "last_voice_at": self.vad.last_voice_at, "first_partial": self.first_partial}
        self._pending_turn = None
        self._barge_pending = False
        ended_at = prior["last_voice_at"] or prior["vad_end"] or perf_counter()
        await self.interrupt()
        self.turn_id += 1
        turn_id, generation = self.turn_id, self.generation
        metrics = TurnMetrics(self.call_id, turn_id)
        now = perf_counter()
        if prior["first_partial"] and prior["speech_start"]:
            metrics.durations_ms["asr_first_partial"] = round((prior["first_partial"] - prior["speech_start"]) * 1000, 2)
        if prior["vad_end"]:
            metrics.durations_ms["asr_finalization"] = round((now - prior["vad_end"]) * 1000, 2)
            if prior["last_voice_at"]:
                metrics.durations_ms["vad_end_of_turn"] = round(max(0, (prior["vad_end"] - prior["last_voice_at"]) * 1000), 2)
        if not self.vad.speaking:
            self.vad.silence_ms = 0
            self.speech_start = self.vad_end = self.first_partial = None
        await self.transport.event({"type": "final", "turn_id": turn_id, "generation": generation, "text": transcript})
        self.task = asyncio.create_task(self._respond(transcript, turn_id, generation, metrics, ended_at, prefetched=prefetched))

    async def _respond(self, question: str, turn_id: int, generation: int, metrics: TurnMetrics, ended_at: float, prefetched=None):
        answer_task = None
        ack_task = None
        transfer = False
        prepared = prefetched
        verify_task = None
        speak_first_mode = self.speak_first and hasattr(self.orchestrator, "answer_speak_first")
        candidate_audio = None if speak_first_mode else CandidateAudio(self.tts, self.call_id, turn_id, generation)
        ack_text = self.acknowledgement_for(question)
        ack_audio = CandidateAudio(self.tts, self.call_id, turn_id, generation)
        await ack_audio.start(ack_text, 0)
        # Approved and played immediately rather than gated behind router/retrieval:
        # the ACK text carries no unverified claim (safety gate is for the real
        # answer, not this filler), so the only cost of speaking it before we know
        # the route is TRANSFER/REDIRECT is an occasional "Секунду, проверяю..."
        # right before a transfer/redirect line. Traded deliberately for ~1.4s off
        # time-to-first-sound on the dominant RAG path.
        ack_audio.approve(ack_audio.key, ack_text)
        ack_task = asyncio.create_task(self._acknowledge_bounded(ack_audio, ack_text, turn_id, generation, metrics, ended_at))

        def start_answer():
            if speak_first_mode:
                kwargs = {"prepared": prepared} if prepared else {}
                return asyncio.create_task(self.orchestrator.answer_speak_first(question, self.history.copy(), metrics, state, **kwargs))
            kwargs = {"prepared": prepared, "candidate_audio": candidate_audio} if prepared else {}
            return asyncio.create_task(self.orchestrator.answer(question, self.history.copy(), metrics, state, **kwargs))

        try:
            async def state(value: str):
                await self._state(value, turn_id, generation)
            route = None
            if prepared is None and hasattr(self.orchestrator, "prepare"):
                prepared = self.orchestrator.prepare(question, self.history.copy())
            if hasattr(self.orchestrator, "route"):
                await state("routing")
                route_task = asyncio.create_task(self.orchestrator.route(question, self.history.copy()))
                if prepared is not None:
                    # Speculative: retrieval is already running via `prepared`, so start
                    # generation/checking before intent routing resolves instead of after.
                    # Discarded unused below if the turn is not actually RAG.
                    answer_task = start_answer()
                started = perf_counter()
                route = await asyncio.wait_for(route_task, 15)
                metrics.elapsed("routing", started)
            if generation != self.generation or self.closed:
                if answer_task and not answer_task.done():
                    answer_task.cancel()
                if answer_task:
                    await asyncio.gather(answer_task, return_exceptions=True)
                if ack_task and not ack_task.done():
                    ack_task.cancel()
                    await asyncio.gather(ack_task, return_exceptions=True)
                await ack_audio.discard()
                return
            if route and route.action != "RAG":
                if answer_task:
                    if not answer_task.done():
                        answer_task.cancel()
                    await asyncio.gather(answer_task, return_exceptions=True)
                    answer_task = None
                if prepared:
                    await prepared.close()
                # The ACK may already be mid-playback (started before routing
                # resolved); cut it off before speaking the transfer/redirect line.
                if ack_task and not ack_task.done():
                    ack_task.cancel()
                    await asyncio.gather(ack_task, return_exceptions=True)
                ack_task = None
                await ack_audio.discard()
                answer, hits = route.text, []
                transfer = route.action == "TRANSFER"
                if transfer:
                    # Terminal policy is latched before producing the final phrase.
                    # ASR cannot cancel it or start a second transfer.
                    self.terminal = True
            else:
                if answer_task is None:
                    answer_task = start_answer()
                result = await asyncio.wait_for(answer_task, 40)
                if speak_first_mode:
                    answer, hits, verify_task = result
                else:
                    answer, hits = result
            if ack_task and not ack_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(ack_task), timeout=1.0)
                except asyncio.TimeoutError:
                    ack_task.cancel()
            if ack_task:
                await asyncio.gather(ack_task, return_exceptions=True)
            if generation != self.generation:
                return
            if prepared:
                metrics.durations_ms.update(prepared.timings)
            if candidate_audio is not None and candidate_audio.first_audio_ms is not None:
                metrics.durations_ms["speculative_tts_first_audio"] = candidate_audio.first_audio_ms
            spoken = await self._speak_audio(answer, hits, turn_id, generation, metrics, ended_at, route)
            if not spoken:
                return
            if verify_task is not None:
                # Checker runs in the background while the candidate is already
                # being spoken (dual-agent: speak-first, verify-in-parallel).
                # By the time the candidate finishes playing the verdict is
                # usually already in; if it rejects, correct immediately rather
                # than silently leaving an unverified claim as the final word.
                correction = await verify_task
                verify_task = None
                if generation != self.generation:
                    return
                if correction:
                    event("speak_first_correction", self.call_id, turn_id, at_monotonic=perf_counter())
                    # Spoken as one bridged utterance so the correction reads as
                    # a natural self-correction, not an abrupt unexplained cut.
                    # History keeps the clean corrected text, not the bridge.
                    bridge = self.CORRECTION_BRIDGE + correction
                    spoken = await self._speak_audio(bridge, [], turn_id, generation, metrics, ended_at, route, tts_total_key="tts_total_correction")
                    if not spoken:
                        return
                    answer, hits = correction, []
            if hasattr(self.transport, "drain"):
                await self.transport.drain(generation)
            await self.transport.event({"type": "audio_end", "generation": generation, "turn_id": turn_id})
            await asyncio.wait_for(self.playback_done.wait(), timeout=30)
            self.history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
            await self.transport.event({"type": "metrics", "turn_id": turn_id, "generation": generation, "history_turns": len(self.history) // 2, **metrics.as_event()})
            if self.on_metrics:
                self.on_metrics(metrics.as_event())
            event("turn_complete", self.call_id, turn_id, metrics_ms=metrics.durations_ms)
            if transfer:
                await self.transport.hangup("transfer")
            else:
                await self._state("listening", turn_id, generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            event("turn_error", self.call_id, turn_id, **safe_error_fields(exc))
            if generation == self.generation:
                await self.transport.event({"type": "error", "turn_id": turn_id, "generation": generation, "message": "Не удалось завершить ответ. Попробуйте ещё раз."})
                if transfer:
                    await self.transport.hangup("transfer_audio_failed")
                else:
                    await self._state("listening", turn_id, generation)
        finally:
            for task in (answer_task, ack_task, verify_task):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (answer_task, ack_task, verify_task) if task), return_exceptions=True)
            if candidate_audio is not None:
                await candidate_audio.discard()
            await ack_audio.discard()
            if prepared:
                await prepared.close()

    async def _speak_audio(self, answer: str, hits: list[dict], turn_id: int, generation: int, metrics: TurnMetrics, ended_at: float, route, tts_total_key: str = "tts_total") -> bool:
        """Streams `answer` audio to the transport. Returns False if the turn
        was interrupted mid-stream (generation changed; caller must stop, not
        proceed to drain/commit). Caller is responsible for drain, history
        commit and terminal events once it knows the final (possibly
        speak-first-corrected) text — this only speaks one utterance.
        """
        sources = {h["article"]: {"article": h["article"], "score": round(h["score"], 3), "source": h["source"]} for h in hits}
        await self.transport.event({"type": "answer", "turn_id": turn_id, "generation": generation, "text": answer, "sources": list(sources.values()), "action": route.action if route else "RAG"})
        await self._state("speaking", turn_id, generation)
        tts_started = perf_counter()
        first = True
        async for pcm in self.tts.synthesize_stream(answer):
            if generation != self.generation:
                return False
            if first:
                first = False
                metrics.durations_ms["tts_first_audio"] = round((perf_counter() - tts_started) * 1000, 2)
                if "end_of_turn_to_first_substantive_audio" not in metrics.durations_ms:
                    metrics.durations_ms["end_of_turn_to_first_substantive_audio"] = round((perf_counter() - ended_at) * 1000, 2)
                    metrics.durations_ms["end_of_turn_to_first_audio"] = metrics.durations_ms["end_of_turn_to_first_substantive_audio"]
                event("tts_first_audio", self.call_id, turn_id, latency_ms=metrics.durations_ms["tts_first_audio"], at_monotonic=perf_counter())
            await self.transport.audio(pcm, generation)
        metrics.elapsed(tts_total_key, tts_started)
        event("tts_total", self.call_id, turn_id, latency_ms=metrics.durations_ms[tts_total_key], at_monotonic=perf_counter())
        if first:
            raise RuntimeError("Deepdub returned no audio")
        return generation == self.generation

    @classmethod
    def acknowledgement_for(cls, question: str) -> str:
        concise = " ".join(question.split())
        return f"Проверяю ваш вопрос: {concise}" if 12 <= len(concise) <= 75 else cls.ACKNOWLEDGEMENT

    async def _acknowledge_bounded(self, ack_audio: CandidateAudio, spoken: str, turn_id: int, generation: int, metrics: TurnMetrics, ended_at: float):
        try:
            await asyncio.wait_for(self._acknowledge(ack_audio, spoken, turn_id, generation, metrics, ended_at), timeout=4.0)
        except asyncio.TimeoutError:
            event("acknowledgement_timeout", self.call_id, turn_id)

    async def _acknowledge(self, ack_audio: CandidateAudio, spoken: str, turn_id: int, generation: int, metrics: TurnMetrics, ended_at: float):
        started = perf_counter()
        try:
            first = True
            async for pcm in ack_audio.stream(spoken, (self.call_id, turn_id, generation)):
                if generation != self.generation:
                    return
                if not pcm:
                    continue
                if first:
                    first = False
                    metrics.durations_ms["end_of_turn_to_first_acknowledgement_audio"] = round((perf_counter() - ended_at) * 1000, 2)
                    metrics.durations_ms["ack_tts_first_audio"] = round((perf_counter() - started) * 1000, 2)
                    await self.transport.event({"type": "acknowledgement", "turn_id": turn_id, "generation": generation, "text": spoken})
                await self.transport.audio(pcm, generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            event("acknowledgement_error", self.call_id, turn_id, error_type=type(exc).__name__)

    async def _speak_clarification(self, generation: int):
        try:
            await self.transport.event({"type": "clarification", "generation": generation, "text": self.CLARIFICATION})
            await self._state("speaking", generation=generation)
            async for pcm in self.tts.synthesize_stream(self.CLARIFICATION):
                if generation != self.generation:
                    return
                await self.transport.audio(pcm, generation)
            if generation == self.generation:
                if hasattr(self.transport, "drain"):
                    await self.transport.drain(generation)
                await self.transport.event({"type": "audio_end", "generation": generation, "turn_id": self.turn_id})
                await asyncio.wait_for(self.playback_done.wait(), timeout=30)
                await self._state("listening", generation=generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            event("clarification_error", self.call_id, self.turn_id, error_type=type(exc).__name__)

    async def _state(self, name: str, turn_id: int | None = None, generation: int | None = None):
        if self.closed or (generation is not None and generation != self.generation):
            return
        self.state = name
        await self.transport.event({"type": "state", "state": name, "turn_id": turn_id or self.turn_id, "generation": self.generation if generation is None else generation})

    async def interrupt(self):
        async with self._interrupt_lock:
            if self.terminal:
                return
            self.generation += 1
            self.playback_done = asyncio.Event()
            old = self.task
            self.task = None
            # Flush scheduled browser audio before waiting for provider cancellation.
            await self.transport.event({"type": "interrupt", "generation": self.generation, "turn_id": self.turn_id})
            event("interruption", self.call_id, self.turn_id, generation=self.generation, at_monotonic=perf_counter())
            if old and not old.done() and old is not asyncio.current_task():
                old.cancel()
                await asyncio.gather(old, return_exceptions=True)
            if hasattr(self.transport, "clear_audio"):
                await self.transport.clear_audio(self.generation)

    def acknowledge_playback(self, generation: int):
        if generation == self.generation:
            self.playback_done.set()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self._asr_begin_task and not self._asr_begin_task.done():
            self._asr_begin_task.cancel()
            await asyncio.gather(self._asr_begin_task, return_exceptions=True)
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.asr.close()
        if hasattr(self.tts, "close"):
            await self.tts.close()
