import asyncio
import uuid
from collections import deque
from time import perf_counter
from app.audio.vad import RMSVAD
from app.observability.latency import TurnMetrics
from app.observability.logging import event
from app.observability.errors import safe_error_fields
from app.call.utterances import is_contentful, is_filler, is_stop, same_utterance
from app.tts.speculative import CandidateAudio
from app.agents.prompts import FALLBACK


async def _noop_state(value):
    pass


class CallSession:
    ACKNOWLEDGEMENT = "Сейчас проверю."
    CLARIFICATION = "Пожалуйста, уточните ваш вопрос об оформлении или получении Kaspi Gold."
    CORRECTION_BRIDGE = "Секунду, уточню. "
    PREFETCH_DEBOUNCE_SECONDS = 0.35

    def __init__(self, transport, asr, orchestrator, tts, on_metrics=None, speak_first=False, ack_tts=None, end_silence_ms: int | None = None, speculative_prefetch: bool = False):
        self.call_id = str(uuid.uuid4())
        self.transport, self.asr, self.orchestrator, self.tts = transport, asr, orchestrator, tts
        # A dedicated connection for the ACK filler, so it never contends
        # with the real answer's synthesis. Defaults to the same `tts`
        # instance so a caller that only has one fake TTS (most tests) is
        # unaffected.
        self.ack_tts = ack_tts or tts
        self._ack_pcm: bytes | None = None
        self._ack_render_task: asyncio.Task | None = None
        # ACK started at VAD end-of-speech (not after ASR finalization, which
        # measured 650-1000ms later); handed to the turn's `_respond` on a
        # contentful `_final`, cut on anything else.
        self._early_ack: asyncio.Task | None = None
        self._early_ack_started: float | None = None
        self.on_metrics = on_metrics
        # Opt-in only: default False keeps the existing never-speak-unverified
        # guarantee (and tests exercising it) unchanged for any caller that
        # does not explicitly request the speak-first/correct-after trade-off.
        self.speak_first = speak_first
        self.vad = RMSVAD(end_silence_ms=end_silence_ms) if end_silence_ms is not None else RMSVAD()
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
        self.history_revision = 0
        # Opt-in: speculatively runs route+answer(+audio) on a stable ASR
        # partial, before VAD end-of-speech. Default False keeps the
        # existing VAD-end-only retrieval prefetch (`_start_prefetch`)
        # unchanged and exclusive -- exactly one of the two ever runs.
        self.speculative_prefetch = speculative_prefetch
        self._speculative_turn: "SpeculativeTurn | None" = None
        self._prefetch_debounce: asyncio.Task | None = None
        self._pending_disposals: set[asyncio.Task] = set()
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
        if self.ack_tts is not self.tts and hasattr(self.ack_tts, "warm"):
            asyncio.create_task(self.ack_tts.warm())
        self._ack_render_task = asyncio.create_task(self._render_ack_pcm())
        await self.transport.event({"type": "connected", "call_id": self.call_id, "generation": self.generation})

    async def _render_ack_pcm(self):
        """Renders the static ACK phrase to PCM once, over `ack_tts`, and
        caches the bytes. From then on, playing the ACK is pure memory -- no
        TTS round-trip, nothing to contend for, nothing to wait on."""
        try:
            chunks = []
            async for pcm in self.ack_tts.synthesize_stream(self.ACKNOWLEDGEMENT):
                if pcm:
                    chunks.append(pcm)
            self._ack_pcm = b"".join(chunks)
        except Exception as exc:
            event("ack_render_error", self.call_id, 0, error_type=type(exc).__name__)

    def _ack_pcm_ready(self) -> bytes | None:
        # Never blocks: the ACK is optional filler, not something the real
        # answer should ever wait on. If rendering (started at call `start()`,
        # normally long done before any turn's `_final` lands) hasn't
        # finished yet, this turn simply has no ACK.
        return self._ack_pcm

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
            self._start_early_ack()
            if self.speculative_prefetch:
                await self._maybe_start_speculative_turn(self._last_partial_text)
            else:
                await self._start_prefetch()
            await self._finalize_asr()

    async def _start_prefetch(self):
        """Speculatively warms retrieval (embedding + Qdrant search) the
        moment VAD confirms the caller stopped talking, using the latest ASR
        partial text instead of waiting for the ASR provider's own
        finalization round-trip (`asr_finalization` metrics showed 700-900ms
        of real dead time here). The ACK itself needs no prefetching: it's a
        static phrase rendered once to PCM at call start (`_render_ack_pcm`),
        so there is nothing per-turn to warm for it.

        Safe to guess wrong: `prepare()` only starts background search tasks
        (no turn/generation/audio side effects) until `_final` confirms the
        text actually matches.
        """
        if self.task and not self.task.done():
            return
        text = self._last_partial_text.strip()
        if not text or not is_contentful(text) or is_stop(text) or is_filler(text):
            return
        if self._prefetch is not None:
            stale_text, stale_prepared = self._prefetch
            self._prefetch = None
            if stale_prepared is not None:
                asyncio.create_task(stale_prepared.close())
        prepared = None
        if hasattr(self.orchestrator, "prepare"):
            try:
                prepared = self.orchestrator.prepare(text, self.history.copy())
            except Exception:
                prepared = None
        self._prefetch = (text, prepared)

    def _start_early_ack(self):
        """Plays the cached ACK the moment VAD confirms end-of-speech, gated
        on the same latest-partial checks as `_start_prefetch` so a stop word
        or filler doesn't get "Сейчас проверю". Only when nothing is already
        playing: during an answer this is a barge-in, and `_respond` keeps
        its own ACK for that case."""
        if self.task and not self.task.done():
            return
        if self._early_ack and not self._early_ack.done():
            return
        text = self._last_partial_text.strip()
        if not text or not is_contentful(text) or is_stop(text) or is_filler(text):
            return
        pcm = self._ack_pcm_ready()
        if not pcm:
            return
        self._early_ack_started = perf_counter()
        self._early_ack = asyncio.create_task(self._play_ack(pcm, self.turn_id + 1, self.generation))

    async def _cut_early_ack(self):
        ack, self._early_ack = self._early_ack, None
        if ack is None:
            return
        was_running = not ack.done()
        if was_running:
            ack.cancel()
        await asyncio.gather(ack, return_exceptions=True)
        if was_running and hasattr(self.transport, "clear_audio"):
            await self.transport.clear_audio(self.generation)

    def _schedule_speculative_debounce(self):
        if self._prefetch_debounce and not self._prefetch_debounce.done():
            self._prefetch_debounce.cancel()
        self._prefetch_debounce = asyncio.create_task(self._debounced_speculative_turn(self._last_partial_text))

    async def _debounced_speculative_turn(self, text: str):
        try:
            await asyncio.sleep(self.PREFETCH_DEBOUNCE_SECONDS)
        except asyncio.CancelledError:
            return
        if self._last_partial_text == text and self.vad.speaking:
            await self._maybe_start_speculative_turn(text)

    def _current_corpus_version(self):
        """Best-effort, side-effect-free read of the active corpus version,
        for the staleness check in `_final` -- not every retriever exposes
        this (test doubles especially don't), so `None` just means that
        check is skipped rather than enforced."""
        retriever = getattr(self.orchestrator, "retriever", None)
        if retriever is None:
            retriever = getattr(getattr(self.orchestrator, "rag", None), "retriever", None)
        manifest = getattr(retriever, "manifest", None)
        if manifest is None:
            return None
        try:
            _, scope = manifest()
            return getattr(scope, "version", None)
        except Exception:
            return None

    def _schedule_disposal(self, turn: "SpeculativeTurn"):
        task = asyncio.create_task(turn.cancel())
        self._pending_disposals.add(task)
        task.add_done_callback(lambda t: self._pending_disposals.discard(t))

    async def _maybe_start_speculative_turn(self, text: str):
        """Starts route+answer(+audio) speculatively for `text`, cancellable,
        gated entirely behind `speculative_prefetch` -- with the flag off
        this is a no-op and the VAD-end-only retrieval prefetch in
        `_start_prefetch` is the only thing that ever runs.

        Disposal of a superseded turn is tracked (`_pending_disposals`), not
        awaited inline: `DeepdubTTSProvider`'s cancel-drain is a real
        protocol round-trip, and this runs on the ASR-partial callback path,
        which must stay responsive. The new turn's own TTS usage is still
        safe to start immediately regardless -- `DeepdubTTSProvider`'s own
        lock serializes the two on the wire; nothing here adds a second lock.
        """
        if not self.speculative_prefetch:
            return
        if self.task and not self.task.done():
            return
        text = text.strip()
        if not text or not is_contentful(text) or is_stop(text) or is_filler(text):
            return
        if self._speculative_turn is not None and same_utterance(self._speculative_turn.text, text):
            return
        old = self._speculative_turn
        self._speculative_turn = SpeculativeTurn(self, text, self.history_revision, self._current_corpus_version())
        if old is not None:
            self._schedule_disposal(old)

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
        if self.speculative_prefetch and meaningful and self.vad.speaking and not (self.task and not self.task.done()):
            self._schedule_speculative_debounce()
        await self.transport.event({"type": "partial", "text": transcript, "generation": self.generation})

    async def _final(self, transcript: str):
        if self.closed or self.terminal:
            return
        if self._prefetch_debounce and not self._prefetch_debounce.done():
            self._prefetch_debounce.cancel()
        prefetched = None
        if self._prefetch is not None:
            prefetch_text, prefetch_prepared = self._prefetch
            self._prefetch = None
            reused = same_utterance(prefetch_text, transcript)
            event("prefetch_match", self.call_id, self.turn_id + 1, kind="retrieval", reused=reused, partial_words=len(prefetch_text.split()), final_words=len(transcript.split()))
            if reused:
                prefetched = prefetch_prepared
            elif prefetch_prepared is not None:
                asyncio.create_task(prefetch_prepared.close())
        prefetched_turn = None
        if self._speculative_turn is not None:
            candidate = self._speculative_turn
            self._speculative_turn = None
            normalized_match = same_utterance(candidate.text, transcript)
            event("prefetch_match", self.call_id, self.turn_id + 1, kind="speculative", reused=normalized_match, partial_words=len(candidate.text.split()), final_words=len(transcript.split()))
            history_match = candidate.history_revision == self.history_revision
            fresh_corpus_version = self._current_corpus_version()
            corpus_match = candidate.corpus_version is None or fresh_corpus_version is None or candidate.corpus_version == fresh_corpus_version
            if normalized_match and history_match and corpus_match:
                prefetched_turn = candidate
            else:
                self._schedule_disposal(candidate)
        if not transcript.strip() or is_stop(transcript) or is_filler(transcript):
            # No real question after all: whatever early ACK VAD end started
            # must not keep saying "Сейчас проверю" over nothing.
            await self._cut_early_ack()
        if not transcript.strip():
            if prefetched:
                asyncio.create_task(prefetched.close())
            if prefetched_turn:
                self._schedule_disposal(prefetched_turn)
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
            if prefetched_turn:
                self._schedule_disposal(prefetched_turn)
            self._pending_turn = None
            self._barge_pending = False
            await self.interrupt()
            await self._state("listening")
            return
        if is_filler(transcript):
            if prefetched:
                asyncio.create_task(prefetched.close())
            if prefetched_turn:
                self._schedule_disposal(prefetched_turn)
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
        early_ack, self._early_ack = self._early_ack, None
        if early_ack is not None and not (self.task and not self.task.done()):
            # Nothing is playing to interrupt, and interrupting would flush
            # the early ACK (plus add the transport's post-interrupt resume
            # gap) -- keep its generation, only reset what interrupt() would
            # have reset for the new turn.
            self.playback_done = asyncio.Event()
        else:
            if early_ack is not None:
                early_ack.cancel()
                await asyncio.gather(early_ack, return_exceptions=True)
                early_ack = None
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
        if early_ack is not None:
            metrics.durations_ms["end_of_turn_to_first_acknowledgement_audio"] = round((self._early_ack_started - ended_at) * 1000, 2)
            metrics.durations_ms["ack_tts_first_audio"] = 0.0
        self.task = asyncio.create_task(self._respond(transcript, turn_id, generation, metrics, ended_at, prefetched=prefetched, prefetched_turn=prefetched_turn, early_ack=early_ack))

    async def _run_pipeline(self, question: str, prepared, metrics: TurnMetrics, state, candidate_audio=None, is_stale=lambda: False, register_task=lambda t: None):
        """Route+answer(+speak-first verify), shared by the live turn path
        (`_respond`) and speculative prefetch (`SpeculativeTurn`). `state` is
        a no-op for speculative calls so UI state events are never emitted
        before a turn is confirmed real. `register_task` is called with every
        child task created (`route_task`, `answer_task`) so a caller that
        needs to own cancellation explicitly (`SpeculativeTurn.cancel`) can,
        rather than relying on cancellation propagating through
        `asyncio.wait_for` alone.

        Returns `(answer, hits, route, verify_task, status)` on success,
        `None` if `is_stale()` becomes true after routing resolves (the
        caller owns the turn's own ack/audio cleanup in that case, not this
        method). `status` is one of "verified" (checker PASS), "fallback"
        (the FALLBACK constant -- safe to speak, but not a verified claim),
        "route" (non-RAG fixed text), or "draft" (speak-first, returned
        before verification).
        """
        speak_first_mode = self.speak_first and hasattr(self.orchestrator, "answer_speak_first")

        # Preferred path: classification and drafting merged into one model
        # call instead of a separately-awaited router call before it -- the
        # two never depended on each other's output. Orchestrators that
        # don't implement this (older test doubles, simpler setups) fall
        # through unchanged to the two-call path below.
        combined_name = "route_and_answer_speak_first" if speak_first_mode else "route_and_answer"
        if hasattr(self.orchestrator, combined_name):
            kwargs = {"candidate_audio": candidate_audio}
            if prepared:
                kwargs["prepared"] = prepared
            task = asyncio.create_task(getattr(self.orchestrator, combined_name)(question, self.history.copy(), metrics, state, **kwargs))
            register_task(task)
            answer, hits, route, verify_task, status = await asyncio.wait_for(task, 40)
            if is_stale():
                if verify_task and not verify_task.done():
                    verify_task.cancel()
                if verify_task:
                    await asyncio.gather(verify_task, return_exceptions=True)
                return None
            return answer, hits, route, verify_task, status

        answer_task = None

        def start_answer():
            kwargs = {"candidate_audio": candidate_audio}
            if prepared:
                kwargs["prepared"] = prepared
            if speak_first_mode:
                task = asyncio.create_task(self.orchestrator.answer_speak_first(question, self.history.copy(), metrics, state, **kwargs))
            else:
                task = asyncio.create_task(self.orchestrator.answer(question, self.history.copy(), metrics, state, **kwargs))
            register_task(task)
            return task

        route = None
        if hasattr(self.orchestrator, "route"):
            await state("routing")
            route_task = asyncio.create_task(self.orchestrator.route(question, self.history.copy()))
            register_task(route_task)
            if prepared is not None:
                # Speculative: retrieval is already running via `prepared`, so start
                # generation/checking before intent routing resolves instead of after.
                # Discarded unused below if the turn is not actually RAG.
                answer_task = start_answer()
            started = perf_counter()
            route = await asyncio.wait_for(route_task, 15)
            metrics.elapsed("routing", started)
        if is_stale():
            if answer_task and not answer_task.done():
                answer_task.cancel()
            if answer_task:
                await asyncio.gather(answer_task, return_exceptions=True)
            return None
        if route and route.action != "RAG":
            if answer_task:
                if not answer_task.done():
                    answer_task.cancel()
                await asyncio.gather(answer_task, return_exceptions=True)
                answer_task = None
            if prepared:
                await prepared.close()
            answer, hits, verify_task, status = route.text, [], None, "route"
        else:
            if answer_task is None:
                answer_task = start_answer()
            result = await asyncio.wait_for(answer_task, 40)
            if speak_first_mode:
                answer, hits, verify_task = result
                status = "draft"
            else:
                answer, hits = result
                verify_task = None
                status = "fallback" if answer == FALLBACK else "verified"
        return answer, hits, route, verify_task, status

    async def _respond(self, question: str, turn_id: int, generation: int, metrics: TurnMetrics, ended_at: float, prefetched=None, prefetched_turn=None, early_ack: asyncio.Task | None = None):
        ack_task = None
        transfer = False
        prepared = prefetched
        verify_task = None
        route = None
        # Built for both modes: speak-first pre-synthesizes+approves the draft
        # the instant it's extracted (inside answer_speak_first), non-speak-first
        # pre-synthesizes during checking and approves on PASS (inside answer).
        # Either way, `_speak_audio` below streams from this buffer instead of
        # calling the TTS a second time for the same text. Replaced below if a
        # reused `prefetched_turn` brings its own (already-transferred) one.
        candidate_audio = CandidateAudio(self.tts, self.call_id, turn_id, generation)
        # Pre-rendered once at call start (`_render_ack_pcm`) -- played
        # immediately rather than gated behind router/retrieval: the ACK text
        # carries no unverified claim (safety gate is for the real answer,
        # not this filler), so the only cost of speaking it before we know
        # the route is TRANSFER/REDIRECT is an occasional "Сейчас проверю"
        # right before a transfer/redirect line. If rendering hasn't finished
        # yet (e.g. the very first turn landing unusually fast), there is
        # simply no ACK this turn rather than blocking on it.
        # Normally already playing since VAD end (`_start_early_ack`); started
        # here only for turns that didn't get one (barge-in, ACK not rendered
        # yet at VAD end, ASR without a contentful partial).
        if early_ack is not None:
            ack_task = early_ack
        else:
            ack_pcm = self._ack_pcm_ready()
            ack_task = asyncio.create_task(self._acknowledge_bounded(ack_pcm, turn_id, generation, metrics, ended_at)) if ack_pcm else None

        async def cut_ack():
            nonlocal ack_task
            if ack_task is None:
                return
            # Only flush the transport's queue if the ACK was actually still
            # producing/playing when we cut it -- if it had already finished
            # naturally, there is nothing queued to flush, and calling
            # `clear_audio` unconditionally on every turn (most turns: the
            # ACK is short and usually done well before the answer is ready)
            # was clearing the LiveKit AudioSource's queue right before the
            # real answer's own frames were about to be pushed into it,
            # which could suppress them depending on the native source's
            # state -- narrowing this to the actually-interrupted case is
            # both more correct and avoids that risk entirely.
            was_running = not ack_task.done()
            if was_running:
                ack_task.cancel()
            await asyncio.gather(ack_task, return_exceptions=True)
            if was_running and hasattr(self.transport, "clear_audio"):
                await self.transport.clear_audio(generation)
            ack_task = None

        try:
            async def state(value: str):
                await self._state(value, turn_id, generation)

            if prefetched_turn is not None:
                # Races `result_ready` against `done` rather than waiting on
                # `result_ready` alone: if the pipeline raised before ever
                # producing a result, only `done` fires, and waiting on
                # `result_ready` alone would block the full 40s for nothing.
                ready_wait = asyncio.create_task(prefetched_turn.result_ready.wait())
                done_wait = asyncio.create_task(prefetched_turn.done.wait())
                try:
                    await asyncio.wait_for(asyncio.wait({ready_wait, done_wait}, return_when=asyncio.FIRST_COMPLETED), 40)
                finally:
                    for t in (ready_wait, done_wait):
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(ready_wait, done_wait, return_exceptions=True)
                usable = (prefetched_turn.result_ready.is_set() and prefetched_turn.failure is None
                          and generation == self.generation and not self.closed)
                if not usable:
                    # Torn down immediately, not deferred to this method's own
                    # `finally`: an unusable turn is still holding a TTS
                    # connection via its `candidate_audio`, and the live path
                    # below is about to want one too.
                    await asyncio.wait_for(prefetched_turn.cancel(), 2.0)
                    prefetched_turn = None

            if prefetched_turn is not None:
                answer, hits, route, verify_task = prefetched_turn.answer, prefetched_turn.hits, prefetched_turn.route, prefetched_turn.verify_task
                status = prefetched_turn.outcome
                metrics.durations_ms.update(prefetched_turn.metrics.durations_ms)
                prepared = prefetched_turn.prepared
                if prefetched_turn.candidate_audio.approved:
                    prefetched_turn.candidate_audio.transfer((self.call_id, -1, -1), (self.call_id, turn_id, generation), answer)
                    candidate_audio = prefetched_turn.candidate_audio
            else:
                if prepared is None and hasattr(self.orchestrator, "prepare"):
                    prepared = self.orchestrator.prepare(question, self.history.copy())
                result = await self._run_pipeline(question, prepared, metrics, state, candidate_audio=candidate_audio,
                                                   is_stale=lambda: generation != self.generation or self.closed)
                if result is None:
                    await cut_ack()
                    return
                answer, hits, route, verify_task, status = result

            if status == "route":
                transfer = route.action == "TRANSFER"
                if transfer:
                    # Terminal policy is latched before producing the final phrase.
                    # ASR cannot cancel it or start a second transfer.
                    self.terminal = True
            # The real answer is ready: give it priority over the ACK rather
            # than waiting for the ACK to finish "naturally" (that wait used
            # to be up to 1s). Cancel the producer, then flush anything
            # already queued-but-not-yet-sent for this generation so it
            # doesn't play ahead of the answer. This is a server-side
            # guarantee only: whatever ACK audio had already left the server
            # before this point still plays out client-side -- cancelling
            # the task doesn't retroactively un-send it.
            await cut_ack()
            if generation != self.generation:
                return
            if prepared:
                metrics.durations_ms.update(prepared.timings)
            if candidate_audio.first_audio_ms is not None:
                metrics.durations_ms["speculative_tts_first_audio"] = candidate_audio.first_audio_ms
            ready_audio = candidate_audio if candidate_audio.approved else None
            spoken = await self._speak_audio(answer, hits, turn_id, generation, metrics, ended_at, route, ready_audio=ready_audio)
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
            self.history_revision += 1
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
            if prefetched_turn is not None:
                # Safe even after a successful reuse: every child task it
                # touches is checked with `if not task.done()` before
                # cancelling, so one that already finished is simply left
                # alone. Covers an error or a barge-in that happens after
                # the hit was accepted but before the turn finished speaking.
                await prefetched_turn.cancel()
            for task in (ack_task, verify_task):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (ack_task, verify_task) if task), return_exceptions=True)
            if candidate_audio is not None:
                await candidate_audio.discard()
            if prepared:
                await prepared.close()

    async def _speak_audio(self, answer: str, hits: list[dict], turn_id: int, generation: int, metrics: TurnMetrics, ended_at: float, route, tts_total_key: str = "tts_total", ready_audio: CandidateAudio | None = None) -> bool:
        """Streams `answer` audio to the transport. Returns False if the turn
        was interrupted mid-stream (generation changed; caller must stop, not
        proceed to drain/commit). Caller is responsible for drain, history
        commit and terminal events once it knows the final (possibly
        speak-first-corrected) text — this only speaks one utterance.

        `ready_audio`, when given and already `.approved`, is an already
        (or still being) synthesized private buffer for this exact `answer`
        text -- streamed from instead of calling the TTS a second time. If it
        raises mid-stream, that propagates like any other synthesis failure;
        there is no fallback branch here, so a failure after streaming has
        started never triggers a silent second synthesis attempt.
        """
        sources = {h["article"]: {"article": h["article"], "score": round(h["score"], 3), "source": h["source"]} for h in hits}
        await self.transport.event({"type": "answer", "turn_id": turn_id, "generation": generation, "text": answer, "sources": list(sources.values()), "action": route.action if route else "RAG"})
        await self._state("speaking", turn_id, generation)
        tts_started = perf_counter()
        first = True
        stream = ready_audio.stream(answer, ready_audio.identity) if ready_audio is not None else self.tts.synthesize_stream(answer)
        async for pcm in stream:
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

    async def _acknowledge_bounded(self, pcm: bytes, turn_id: int, generation: int, metrics: TurnMetrics, ended_at: float):
        metrics.durations_ms["end_of_turn_to_first_acknowledgement_audio"] = round((perf_counter() - ended_at) * 1000, 2)
        metrics.durations_ms["ack_tts_first_audio"] = 0.0
        await self._play_ack(pcm, turn_id, generation)

    async def _play_ack(self, pcm: bytes, turn_id: int, generation: int):
        try:
            await asyncio.wait_for(self._acknowledge(pcm, turn_id, generation), timeout=4.0)
        except asyncio.TimeoutError:
            event("acknowledgement_timeout", self.call_id, turn_id)

    async def _acknowledge(self, pcm: bytes, turn_id: int, generation: int):
        """Plays the pre-rendered static ACK phrase from cached PCM -- no TTS
        round-trip, so there is no synthesis step to time or to contend with
        the real answer for."""
        try:
            await self.transport.event({"type": "acknowledgement", "turn_id": turn_id, "generation": generation, "text": self.ACKNOWLEDGEMENT})
            for offset in range(0, len(pcm), 3200):
                if generation != self.generation:
                    return
                await self.transport.audio(pcm[offset:offset + 3200], generation)
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
        if self._ack_render_task and not self._ack_render_task.done():
            self._ack_render_task.cancel()
            await asyncio.gather(self._ack_render_task, return_exceptions=True)
        if self._early_ack and not self._early_ack.done():
            self._early_ack.cancel()
            await asyncio.gather(self._early_ack, return_exceptions=True)
        if self._prefetch_debounce and not self._prefetch_debounce.done():
            self._prefetch_debounce.cancel()
            await asyncio.gather(self._prefetch_debounce, return_exceptions=True)
        if self._speculative_turn is not None:
            await self._speculative_turn.cancel()
            self._speculative_turn = None
        if self._pending_disposals:
            await asyncio.gather(*self._pending_disposals, return_exceptions=True)
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.asr.close()
        if hasattr(self.tts, "close"):
            await self.tts.close()
        if self.ack_tts is not self.tts and hasattr(self.ack_tts, "close"):
            await self.ack_tts.close()


class SpeculativeTurn:
    """Owns one speculative route+answer(+verify)+audio computation for a
    not-yet-final ASR partial. Every child task and the private audio
    buffer are torn down together by `cancel()`; nothing it produces is
    played until `CallSession._final`/`_respond` re-validate text, history
    revision and corpus version against the real final transcript.
    """

    def __init__(self, session: CallSession, text: str, history_revision: int, corpus_version):
        self.text, self.history_revision, self.corpus_version = text, history_revision, corpus_version
        self.prepared = session.orchestrator.prepare(text, session.history.copy()) if hasattr(session.orchestrator, "prepare") else None
        self.candidate_audio = CandidateAudio(session.tts, session.call_id, -1, -1)
        self.candidate_audio.on_start = self._on_audio_start
        self.candidate_audio.on_first_audio = self._on_first_audio
        self.metrics = TurnMetrics(session.call_id, -1)
        self.route = self.answer = self.hits = self.verify_task = self.failure = None
        self.outcome: str | None = None   # "verified" | "fallback" | "route" | "draft" | "rejected" | "failed"
        self._children: list[asyncio.Task] = []          # route_task / answer_task, tracked explicitly
        self.result_ready = asyncio.Event()               # (answer, hits, route) resolved -- NOT proof of audio or verification
        self.verified = asyncio.Event()                   # set ONLY on checker PASS / verify_task resolving to no-correction
        self.audio_ready = asyncio.Event()                 # set the instant the first PCM chunk actually lands, via a callback
        self.done = asyncio.Event()                        # set exactly once, success or failure; carries no success claim of its own
        self.timestamps: dict[str, float] = {}
        self._audio_ready_for_attempt = None
        self.task = asyncio.create_task(self._run(session))

    def _on_audio_start(self):
        # Mirrors what `CandidateAudio.start()` itself just reset (queue,
        # first_audio_ms, key) for this attempt -- without this, a RETRY's
        # fresh attempt would inherit a prior (discarded) attempt's
        # already-set `audio_ready` Event and stale timestamp. Bound to
        # `attempt_id`, not `key`: `transfer()` legitimately rewrites `key`
        # on a successful handoff, possibly before the first chunk has even
        # arrived, and that must not make this comparison start failing.
        self.audio_ready = asyncio.Event()
        self.timestamps.pop("audio_ready", None)
        self._audio_ready_for_attempt = self.candidate_audio.attempt_id

    def _on_first_audio(self):
        if self.candidate_audio.attempt_id != self._audio_ready_for_attempt:
            return   # late callback from a buffer already superseded by a later attempt
        self.timestamps["audio_ready"] = perf_counter()
        self.audio_ready.set()

    async def _run(self, session: CallSession):
        try:
            result = await session._run_pipeline(
                self.text, self.prepared, self.metrics, _noop_state,
                candidate_audio=self.candidate_audio, register_task=self._children.append)
            if result is None:
                self.outcome = "failed"
                return
            self.answer, self.hits, self.route, self.verify_task, status = result
            self.outcome = status
            self.timestamps["result_ready"] = perf_counter()
            self.result_ready.set()
            if status == "verified":
                self.timestamps["verified"] = perf_counter()
                self.verified.set()
            elif self.verify_task is not None:          # speak-first: status == "draft", verify runs after
                correction = await self.verify_task
                if correction is None:
                    self.outcome = "verified"
                    self.timestamps["verified"] = perf_counter()
                    self.verified.set()
                else:
                    self.outcome = "rejected"
                    # `_respond` still re-derives the correction from
                    # `.verify_task`'s own result itself when it reuses this
                    # turn, rather than trusting `.outcome` for that -- so
                    # there is no second source of truth to disagree.
        except Exception as exc:
            self.failure = exc
            self.outcome = "failed"
        finally:
            self.done.set()   # unblocks any waiter; callers check `.outcome`/`.failure` explicitly -- `done` alone is never success

    async def cancel(self):
        for task in self._children:
            if not task.done():
                task.cancel()
        if self._children:
            await asyncio.gather(*self._children, return_exceptions=True)
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        if self.verify_task and not self.verify_task.done():
            self.verify_task.cancel()
        if self.verify_task:
            await asyncio.gather(self.verify_task, return_exceptions=True)
        if self.prepared:
            await self.prepared.close()
        await self.candidate_audio.discard()
