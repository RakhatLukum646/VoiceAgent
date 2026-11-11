import asyncio
import struct
from types import SimpleNamespace
import pytest
from app.call.session import CallSession
from app.agents.orchestrator import Orchestrator
from app.tts.speculative import CandidateAudio


class PartialAwareASR:
    async def start(self, partial, final): self.partial, self.final = partial, final
    async def send_audio(self, pcm): pass
    async def finalize(self): pass
    async def close(self): pass


def _tone(ms, amplitude):
    samples = int(16000 * ms / 1000)
    return struct.pack("<%dh" % samples, *([amplitude] * samples))


class Transport:
    def __init__(self): self.events, self.pcm = [], []
    async def event(self, value): self.events.append(value)
    async def audio(self, value, generation): self.pcm.append((value, generation))
    async def clear_audio(self, generation): pass


async def _wait_for_audio_end(transport, timeout_iters=300):
    for _ in range(timeout_iters):
        if any(e.get("type") == "audio_end" for e in transport.events):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("audio_end not observed in time")


class Prepared:
    def __init__(self, hits, owner=None, question=None):
        self.hits, self.timings, self.closed, self.owner, self.question = hits, {}, False, owner, question
    async def search(self, q=None): return self.hits
    async def verifier_search(self, q=None): return self.hits
    async def close(self):
        self.closed = True
        if self.owner is not None:
            self.owner.closed.append(self.question)


class CountingTTS:
    def __init__(self):
        self.calls = []
    async def synthesize_stream(self, text):
        self.calls.append(text)
        yield b"X"


HIT = {"article": "a", "text": "official", "source": "s", "section": "title", "score": 1.0}


def make_scoped(generator, checker, hits=(HIT,), owner=None):
    rag = Orchestrator(Prepared(list(hits)), generator, checker)
    class Scoped:
        def __init__(self):
            self.prepare_calls = []
        def prepare(self, question, history):
            self.prepare_calls.append(question)
            return Prepared(list(hits), owner, question)
        async def answer(self, *args, **kwargs):
            return await rag.answer(*args, **kwargs)
        async def answer_speak_first(self, *args, **kwargs):
            return await rag.answer_speak_first(*args, **kwargs)
    return Scoped()


class PassChecker:
    async def check(self, *args, **kwargs):
        return {"action": "PASS"}


def make_generator(answer_text="verified answer", claims_article="a"):
    class Generator:
        async def generate(self, *args):
            return {"answer": answer_text, "cited_articles": [claims_article],
                     "claims": [{"text": "fact", "article": claims_article, "quote": "official"}],
                     "insufficient_context": False}, None
    return Generator()


@pytest.mark.asyncio
async def test_stable_partial_triggers_full_pipeline_before_final():
    checked = asyncio.Event()
    class Checker:
        async def check(self, *args, **kwargs):
            checked.set()
            return {"action": "PASS"}
    scoped = make_scoped(make_generator(), Checker())
    transport, tts = Transport(), CountingTTS()
    session = CallSession(transport, PartialAwareASR(), scoped, tts, speculative_prefetch=True)
    await session.start()
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    # Still speaking -- no VAD end yet -- but the debounce should fire.
    await asyncio.wait_for(checked.wait(), 1.0)
    assert scoped.prepare_calls == [text]
    assert session._speculative_turn is not None
    await session.feed_audio(_tone(700, 0))
    await session._final(text)
    await _wait_for_audio_end(transport)
    # Reused: no second prepare() call for the same stable text.
    assert scoped.prepare_calls == [text]
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_changed_partial_cancels_in_flight_turn_without_leaking_children():
    partial_text = "Как мне оформить Kaspi"
    new_text = "Как мне оформить Kaspi Gold для ребенка"
    completed_for = []
    release = asyncio.Event()
    class Generator:
        async def generate(self, question, *rest):
            if question == partial_text:
                # Only the superseded (old) turn's generate call blocks here --
                # if cancellation works, it must never resume past this await,
                # regardless of when/whether `release` is later set.
                await release.wait()
            completed_for.append(question)
            return {"answer": "x", "cited_articles": ["a"], "claims": [{"text": "fact", "article": "a", "quote": "official"}], "insufficient_context": False}, None
    class Checker:
        async def check(self, *args, **kwargs):
            return {"action": "PASS"}
    scoped = make_scoped(Generator(), Checker())
    transport = Transport()
    session = CallSession(transport, PartialAwareASR(), scoped, CountingTTS(), speculative_prefetch=True)
    await session.start()
    await session.feed_audio(_tone(200, 1000))
    await session._partial(partial_text)
    await asyncio.sleep(CallSession.PREFETCH_DEBOUNCE_SECONDS + 0.1)
    turn = session._speculative_turn
    assert turn is not None and not turn.task.done()
    await session._partial(new_text)
    for _ in range(300):
        if turn.task.done():
            break
        await asyncio.sleep(0.01)
    # The old turn's task must actually be cancelled (not just superseded).
    assert turn.task.done()
    release.set()
    await asyncio.sleep(0.05)
    assert partial_text not in completed_for, "generate() for the superseded text must never run to completion after cancellation"
    assert new_text in completed_for, "the new (current) turn must still be free to run normally"
    await session.close()


@pytest.mark.asyncio
async def test_history_revision_mismatch_treated_as_miss():
    scoped = make_scoped(make_generator(), PassChecker())
    transport = Transport()
    session = CallSession(transport, PartialAwareASR(), scoped, CountingTTS(), speculative_prefetch=True)
    await session.start()
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    await asyncio.sleep(CallSession.PREFETCH_DEBOUNCE_SECONDS + 0.1)
    assert session._speculative_turn is not None
    # Simulate another turn completing and appending to history in between.
    session.history.append({"role": "user", "content": "unrelated"})
    session.history_revision += 1
    await session.feed_audio(_tone(700, 0))
    await session._final(text)
    await _wait_for_audio_end(transport)
    # A fresh prepare() must have run for the live fallback path.
    assert scoped.prepare_calls.count(text) == 2, "stale (history-mismatched) turn must not be reused; live path re-prepares"
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_corpus_version_mismatch_treated_as_miss():
    version_holder = {"value": "v1"}
    class FakeRetriever:
        def manifest(self):
            return {}, SimpleNamespace(version=version_holder["value"])
    generator, checker = make_generator(), PassChecker()
    rag = Orchestrator(Prepared([HIT]), generator, checker)
    rag.retriever = FakeRetriever()
    class Scoped:
        def __init__(self):
            self.prepare_calls = []
            self.retriever = FakeRetriever()
        def prepare(self, question, history):
            self.prepare_calls.append(question)
            return Prepared([HIT])
        async def answer(self, *args, **kwargs):
            return await rag.answer(*args, **kwargs)
    scoped = Scoped()
    transport = Transport()
    session = CallSession(transport, PartialAwareASR(), scoped, CountingTTS(), speculative_prefetch=True)
    await session.start()
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    await asyncio.sleep(CallSession.PREFETCH_DEBOUNCE_SECONDS + 0.1)
    assert session._speculative_turn is not None
    assert session._speculative_turn.corpus_version == "v1"
    version_holder["value"] = "v2"   # corpus reindexed between prefetch and final
    await session.feed_audio(_tone(700, 0))
    await session._final(text)
    await _wait_for_audio_end(transport)
    assert scoped.prepare_calls.count(text) == 2, "a corpus-version mismatch must force a fresh prepare(), not reuse the stale turn"
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_speak_first_draft_audio_prefetched_and_reused():
    scoped = make_scoped(make_generator("draft text"), PassChecker())
    transport, tts = Transport(), CountingTTS()
    session = CallSession(transport, PartialAwareASR(), scoped, tts, speak_first=True, speculative_prefetch=True)
    await session.start()
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    for _ in range(200):
        if "draft text" in tts.calls:
            break
        await asyncio.sleep(0.01)
    assert session._speculative_turn.candidate_audio.approved, "speak-first must approve the draft's audio speculatively, before VAD end"
    assert tts.calls.count("draft text") == 1
    await session.feed_audio(_tone(700, 0))
    await session._final(text)
    await _wait_for_audio_end(transport)
    assert tts.calls.count("draft text") == 1, "the prefetched draft audio must be reused, not synthesized again"
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_error_before_any_result_falls_back_promptly_not_after_full_timeout():
    class BrokenGenerator:
        async def generate(self, *args):
            raise RuntimeError("boom")
    scoped = make_scoped(BrokenGenerator(), PassChecker())
    transport = Transport()
    session = CallSession(transport, PartialAwareASR(), scoped, CountingTTS(), speculative_prefetch=True)
    await session.start()
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    await asyncio.sleep(CallSession.PREFETCH_DEBOUNCE_SECONDS + 0.1)
    turn = session._speculative_turn
    assert turn is not None
    await asyncio.wait_for(turn.done.wait(), 2.0)
    assert turn.failure is not None and not turn.result_ready.is_set()
    await session.feed_audio(_tone(700, 0))
    import time
    started = time.monotonic()
    await session._final(text)
    # The live fallback path hits the same BrokenGenerator, so this turn
    # ends in an error either way -- what's under test is that the wait on
    # the failed speculative turn itself resolves promptly, not the 40s bound.
    for _ in range(300):
        if any(e.get("type") in ("audio_end", "error") for e in transport.events):
            break
        await asyncio.sleep(0.01)
    assert any(e.get("type") == "error" for e in transport.events)
    assert time.monotonic() - started < 5, "a failed speculative turn must not make _respond wait anywhere near the 40s bound"
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_fallback_without_pass_is_still_spoken_without_waiting_on_verified():
    scoped = make_scoped(make_generator(), PassChecker(), hits=())  # no hits -> FALLBACK path, no checker call needed
    transport = Transport()
    session = CallSession(transport, PartialAwareASR(), scoped, CountingTTS(), speculative_prefetch=True)
    await session.start()
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    await asyncio.sleep(CallSession.PREFETCH_DEBOUNCE_SECONDS + 0.1)
    turn = session._speculative_turn
    assert turn is not None
    await asyncio.wait_for(turn.result_ready.wait(), 2.0)
    assert turn.outcome == "fallback"
    assert not turn.verified.is_set()
    await session.feed_audio(_tone(700, 0))
    await session._final(text)
    await _wait_for_audio_end(transport)
    answers = [e["text"] for e in transport.events if e.get("type") == "answer"]
    from app.agents.prompts import FALLBACK
    assert answers == [FALLBACK]
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_barge_in_after_ownership_transfer_still_cancels_everything():
    result_ready_seen = asyncio.Event()
    class SlowChecker:
        async def check(self, *args, **kwargs):
            result_ready_seen.set()
            await asyncio.sleep(10)
            return {"action": "PASS"}
    scoped = make_scoped(make_generator(), SlowChecker())
    transport = Transport()
    session = CallSession(transport, PartialAwareASR(), scoped, CountingTTS(), speculative_prefetch=True)
    await session.start()
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    await asyncio.sleep(CallSession.PREFETCH_DEBOUNCE_SECONDS + 0.1)
    turn = session._speculative_turn
    assert turn is not None
    await session.feed_audio(_tone(700, 0))
    await session._final(text)
    # `_respond` is now racing result_ready/done on the handed-off turn
    # (checker is stuck for 10s) -- barge in with new speech before it resolves.
    await asyncio.sleep(0.05)
    pending_task = session.task
    await session.interrupt()
    await asyncio.gather(pending_task, return_exceptions=True)
    for _ in range(200):
        if turn.task.done():
            break
        await asyncio.sleep(0.01)
    assert turn.task.done(), "the handed-off turn's own pipeline task must still be torn down after a barge-in"
    await session.close()


@pytest.mark.asyncio
async def test_retry_after_first_pcm_chunk_resets_audio_ready_for_new_attempt():
    attempts = {"n": 0}
    class FlakyGenerator:
        async def generate(self, *args):
            attempts["n"] += 1
            text = "bad draft" if attempts["n"] == 1 else "good answer"
            return {"answer": text, "cited_articles": ["a"], "claims": [{"text": "fact", "article": "a", "quote": "official"}], "insufficient_context": False}, None
    class RetryOnceChecker:
        async def check(self, *args, **kwargs):
            return {"action": "PASS"} if attempts["n"] > 1 else {"action": "RETRY"}
    scoped = make_scoped(FlakyGenerator(), RetryOnceChecker())
    transport, tts = Transport(), CountingTTS()
    session = CallSession(transport, PartialAwareASR(), scoped, tts, speculative_prefetch=True)
    await session.start()
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    for _ in range(300):
        turn = session._speculative_turn
        if turn is not None and turn.result_ready.is_set():
            break
        await asyncio.sleep(0.01)
    assert turn.outcome == "verified"
    assert turn.candidate_audio.approved
    # Only the final, approved attempt's text may ever reach the real output.
    await session.feed_audio(_tone(700, 0))
    await session._final(text)
    await _wait_for_audio_end(transport)
    answers = [e["text"] for e in transport.events if e.get("type") == "answer"]
    assert answers == ["good answer"]
    assert "bad draft" not in answers
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


def test_candidate_audio_transfer_requires_approved_matching_untransferred_buffer():
    async def run():
        async def fake_tts(text):
            yield b"\x00\x00"
        class TTS:
            async def synthesize_stream(self, text):
                async for chunk in fake_tts(text):
                    yield chunk
        audio = CandidateAudio(TTS(), "call", -1, -1)
        with pytest.raises(ValueError):
            audio.transfer(("call", -1, -1), ("call", 1, 0), "x")  # not started/approved yet
        key = await audio.start("the text", 0)
        with pytest.raises(ValueError):
            audio.transfer(("call", -1, -1), ("call", 1, 0), "the text")  # not approved yet
        audio.approve(key, "the text")
        with pytest.raises(ValueError):
            audio.transfer(("call", "WRONG", -1), ("call", 1, 0), "the text")  # owner mismatch
        with pytest.raises(ValueError):
            audio.transfer(("call", -1, -1), ("call", 1, 0), "different text")  # text mismatch
        audio.transfer(("call", -1, -1), ("call", 1, 0), "the text")
        assert audio.identity == ("call", 1, 0)
        with pytest.raises(ValueError):
            audio.transfer(("call", 1, 0), ("call", 2, 0), "the text")  # one-shot: already transferred
        await audio.discard()
    asyncio.run(run())
