import asyncio
import pytest
from app.agents.generator import AnswerGenerator
from app.agents.orchestrator import Orchestrator
from app.agents.scope import ScopedOrchestrator, TRANSFER, REDIRECT, CLARIFY, RAG_FALLBACK, ROUTED_GENERATOR
from app.agents.prompts import FALLBACK
from app.observability.latency import TurnMetrics
from app.call.session import CallSession


HIT = {"article": "1", "section": "I", "text": "official text", "source": "s", "score": 0.9}


class FakeChat:
    """Non-streaming only: `json_completion` returns a fixed response."""
    def __init__(self, response):
        self.response = response
        self.calls = []
    async def json_completion(self, system, payload, stream=False):
        self.calls.append(system)
        return self.response, None


class StreamingFakeChat:
    """Mimics json_completion_streaming's early_fields contract: yields
    ("draft", field, value) for each field in `field_values` (in order),
    then ("done", final, ttft)."""
    def __init__(self, field_values: dict, final: dict):
        self.field_values, self.final = field_values, final
        self.streaming_calls = []
    async def json_completion(self, system, payload, stream=False):
        return self.final, None
    async def json_completion_streaming(self, system, payload, early_field=None, early_fields=None):
        self.streaming_calls.append(system)
        for field, value in self.field_values.items():
            yield "draft", field, value
        yield "done", self.final, 42.0


class PassChecker:
    async def check(self, *args, **kwargs):
        return {"action": "PASS"}


class Retriever:
    def __init__(self, hits=(HIT,)):
        self.hits = list(hits)
    async def search(self, q=None):
        return self.hits
    async def verifier_search(self, q=None):
        return self.hits


@pytest.mark.asyncio
async def test_generate_routed_validates_action_and_claims():
    good = {"action": "RAG", "answer": "x", "cited_articles": ["1"], "claims": [{"text": "t", "article": "1", "quote": "q"}], "insufficient_context": False}
    gen = AnswerGenerator(FakeChat(good))
    result, ttft = await gen.generate_routed("q", [], [HIT])
    assert result["action"] == "RAG" and result["answer"] == "x"


@pytest.mark.asyncio
async def test_generate_routed_rejects_invalid_action():
    bad = {"action": "SOMETHING_ELSE", "answer": "x", "cited_articles": [], "claims": [], "insufficient_context": False}
    gen = AnswerGenerator(FakeChat(bad))
    with pytest.raises(ValueError):
        await gen.generate_routed("q", [], [HIT])


@pytest.mark.asyncio
async def test_generate_routed_skips_claim_validation_for_non_rag():
    # TRANSFER/REDIRECT carry no answer -- must not be rejected for empty claims.
    redirect = {"action": "REDIRECT", "answer": "", "cited_articles": [], "claims": [], "insufficient_context": False}
    gen = AnswerGenerator(FakeChat(redirect))
    result, _ = await gen.generate_routed("q", [], [HIT])
    assert result["action"] == "REDIRECT"


@pytest.mark.asyncio
async def test_generate_routed_streaming_yields_action_before_draft_for_rag():
    final = {"action": "RAG", "answer": "the answer", "cited_articles": ["1"], "claims": [{"text": "t", "article": "1", "quote": "q"}], "insufficient_context": False}
    chat = StreamingFakeChat({"action": "RAG", "answer": "the answer"}, final)
    gen = AnswerGenerator(chat)
    events = []
    async for kind, *rest in gen.generate_routed_streaming("q", [], [HIT]):
        events.append((kind, rest[0] if rest else None))
        if kind == "done":
            break
    kinds = [k for k, _ in events]
    assert kinds[0] == "action" and events[0][1] == "RAG"
    assert ("draft", "the answer") in events
    assert kinds[-1] == "done"


@pytest.mark.asyncio
async def test_generate_routed_streaming_yields_empty_draft_immediately_for_non_rag():
    final = {"action": "TRANSFER", "answer": "", "cited_articles": [], "claims": [], "insufficient_context": False}
    # The model is instructed to leave `answer` empty for non-RAG, but even if
    # it streams a non-empty value anyway, it must never reach the caller as
    # a second draft once action is known non-RAG.
    chat = StreamingFakeChat({"action": "TRANSFER", "answer": "stray text"}, final)
    gen = AnswerGenerator(chat)
    events = []
    async for kind, *rest in gen.generate_routed_streaming("q", [], [HIT]):
        events.append((kind, rest[0] if rest else None))
        if kind == "done":
            break
    assert events[0] == ("action", "TRANSFER")
    assert events[1] == ("draft", "")
    assert ("draft", "stray text") not in events


@pytest.mark.asyncio
async def test_orchestrator_classify_and_answer_rag_pass():
    good = {"action": "RAG", "answer": "verified answer", "cited_articles": ["1"], "claims": [{"text": "t", "article": "1", "quote": "official"}], "insufficient_context": False}
    orch = Orchestrator(Retriever(), AnswerGenerator(FakeChat(good)), PassChecker())
    metrics = TurnMetrics("c", 1)
    async def state(v): pass
    action, answer, hits = await orch.classify_and_answer("q", [], metrics, state)
    assert action == "RAG" and answer == "verified answer" and hits


@pytest.mark.asyncio
async def test_orchestrator_classify_and_answer_short_circuits_non_rag():
    redirect = {"action": "REDIRECT", "answer": "", "cited_articles": [], "claims": [], "insufficient_context": False}
    class NeverCalledChecker:
        async def check(self, *a, **k):
            raise AssertionError("checker must not run for a non-RAG action")
    orch = Orchestrator(Retriever(), AnswerGenerator(FakeChat(redirect)), NeverCalledChecker())
    metrics = TurnMetrics("c", 1)
    async def state(v): pass
    action, answer, hits = await orch.classify_and_answer("q", [], metrics, state)
    assert action == "REDIRECT" and answer is None and hits == []


@pytest.mark.asyncio
async def test_scoped_orchestrator_route_and_answer_builds_route_for_transfer():
    transfer = {"action": "TRANSFER", "answer": "", "cited_articles": [], "claims": [], "insufficient_context": False}
    orch = Orchestrator(Retriever(), AnswerGenerator(FakeChat(transfer)), PassChecker())
    scoped = ScopedOrchestrator(orch, FakeChat({}))
    metrics = TurnMetrics("c", 1)
    async def state(v): pass
    answer, hits, route, verify_task, status = await scoped.route_and_answer("q", [], metrics, state)
    assert status == "route" and route.action == "TRANSFER" and answer == TRANSFER
    assert verify_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("action, expected", [("REDIRECT", REDIRECT), ("CLARIFY", CLARIFY)])
async def test_scoped_orchestrator_distinguishes_off_topic_from_clarify(action, expected):
    routed = {"action": action, "answer": "", "cited_articles": [], "claims": [], "insufficient_context": False}
    orch = Orchestrator(Retriever(), AnswerGenerator(FakeChat(routed)), PassChecker())
    scoped = ScopedOrchestrator(orch, FakeChat({}))
    async def state(v): pass
    answer, hits, route, verify_task, status = await scoped.route_and_answer("q", [], TurnMetrics("c", 1), state)
    assert status == "route" and route.action == action and answer == expected and verify_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("action, expected", [("CLARIFY", CLARIFY), ("REDIRECT", REDIRECT), ("BOGUS", REDIRECT)])
async def test_separate_router_maps_clarify_and_rejects_unknown(action, expected):
    route = await ScopedOrchestrator(None, FakeChat({"action": action})).route("q", [])
    assert route.text == expected
    assert route.action == (action if action != "BOGUS" else "REDIRECT")


def test_off_topic_and_clarify_phrases_differ():
    assert REDIRECT != CLARIFY and "Kaspi Gold" in REDIRECT and "Kaspi Gold" in CLARIFY
    assert "CLARIFY" in ROUTED_GENERATOR


@pytest.mark.asyncio
async def test_scoped_orchestrator_route_and_answer_verified_status():
    good = {"action": "RAG", "answer": "verified answer", "cited_articles": ["1"], "claims": [{"text": "t", "article": "1", "quote": "official text"}], "insufficient_context": False}
    orch = Orchestrator(Retriever(), AnswerGenerator(FakeChat(good)), PassChecker())
    scoped = ScopedOrchestrator(orch, FakeChat({}))
    metrics = TurnMetrics("c", 1)
    async def state(v): pass
    answer, hits, route, verify_task, status = await scoped.route_and_answer("q", [], metrics, state)
    assert status == "verified" and answer == "verified answer" and route is None


@pytest.mark.asyncio
async def test_scoped_orchestrator_route_and_answer_fallback_status_on_no_hits():
    # A realistic generator facing empty passages reports insufficient context.
    insufficient = {"action": "RAG", "answer": "", "cited_articles": [], "claims": [], "insufficient_context": True}
    orch = Orchestrator(Retriever(hits=()), AnswerGenerator(FakeChat(insufficient)), PassChecker())
    scoped = ScopedOrchestrator(orch, FakeChat({}))
    metrics = TurnMetrics("c", 1)
    async def state(v): pass
    answer, hits, route, verify_task, status = await scoped.route_and_answer("q", [], metrics, state)
    assert status == "fallback" and answer == FALLBACK


@pytest.mark.asyncio
async def test_scoped_orchestrator_route_and_answer_speak_first_draft():
    final = {"action": "RAG", "answer": "the draft", "cited_articles": ["1"], "claims": [{"text": "t", "article": "1", "quote": "official text"}], "insufficient_context": False}
    chat = StreamingFakeChat({"action": "RAG", "answer": "the draft"}, final)
    orch = Orchestrator(Retriever(), AnswerGenerator(chat), PassChecker())
    scoped = ScopedOrchestrator(orch, FakeChat({}))
    metrics = TurnMetrics("c", 1)
    async def state(v): pass
    answer, hits, route, verify_task, status = await scoped.route_and_answer_speak_first("q", [], metrics, state)
    assert status == "draft" and answer == "the draft" and route is None
    assert verify_task is not None
    correction = await verify_task
    assert correction is None  # checker PASS -> no correction needed


@pytest.mark.asyncio
async def test_scoped_orchestrator_route_and_answer_speak_first_transfer_has_no_verify_task():
    final = {"action": "TRANSFER", "answer": "", "cited_articles": [], "claims": [], "insufficient_context": False}
    chat = StreamingFakeChat({"action": "TRANSFER"}, final)
    class UnusedChecker:
        async def check(self, *a, **k):
            raise AssertionError("checker must not run for TRANSFER")
    orch = Orchestrator(Retriever(), AnswerGenerator(chat), UnusedChecker())
    scoped = ScopedOrchestrator(orch, FakeChat({}))
    metrics = TurnMetrics("c", 1)
    async def state(v): pass
    answer, hits, route, verify_task, status = await scoped.route_and_answer_speak_first("q", [], metrics, state)
    assert status == "route" and route.action == "TRANSFER" and answer == TRANSFER
    assert verify_task is None


def test_early_field_reads_booleans_and_strings_from_partial_json():
    from app.agents.generator import _early_field, _early_field_pattern
    buffer = '{"action": "RAG", "insufficient_context": true, "answer": "да, \\"можно\\"'
    assert _early_field(buffer, _early_field_pattern("insufficient_context")) is True
    assert _early_field(buffer, _early_field_pattern("action")) == "RAG"
    assert _early_field(buffer, _early_field_pattern("answer")) is None   # still streaming
    assert _early_field(buffer + '", ', _early_field_pattern("answer")) == 'да, "можно"'


def test_insufficient_context_is_generated_before_answer():
    from app.agents.generator import GENERATOR_SCHEMA, ROUTED_GENERATOR_SCHEMA
    assert list(GENERATOR_SCHEMA["properties"])[:2] == ["insufficient_context", "answer"]
    assert list(ROUTED_GENERATOR_SCHEMA["properties"])[:3] == ["action", "insufficient_context", "answer"]


@pytest.mark.asyncio
async def test_speak_first_never_drafts_an_answer_flagged_insufficient_up_front():
    """The model disowned the answer before writing it: speak FALLBACK now,
    with no draft audio and no later correction contradicting it."""
    final = {"action": "RAG", "insufficient_context": True, "answer": "частичный ответ", "cited_articles": [], "claims": []}
    chat = StreamingFakeChat({"action": "RAG", "insufficient_context": True, "answer": "частичный ответ"}, final)
    class UnusedChecker:
        async def check(self, *a, **k):
            raise AssertionError("checker must not run for an answer that won't be spoken")
    orch = Orchestrator(Retriever(), AnswerGenerator(chat), UnusedChecker())
    scoped = ScopedOrchestrator(orch, FakeChat({}))
    async def state(v): pass
    answer, hits, route, verify_task, status = await scoped.route_and_answer_speak_first("q", [], TurnMetrics("c", 1), state)
    assert answer == FALLBACK and verify_task is None and hits == []


@pytest.mark.asyncio
async def test_plain_speak_first_never_drafts_an_answer_flagged_insufficient_up_front():
    final = {"insufficient_context": True, "answer": "частичный ответ", "cited_articles": [], "claims": []}
    chat = StreamingFakeChat({"insufficient_context": True, "answer": "частичный ответ"}, final)
    orch = Orchestrator(Retriever(), AnswerGenerator(chat), PassChecker())
    async def state(v): pass
    answer, hits, verify_task = await orch.answer_speak_first("q", [], TurnMetrics("c", 1), state)
    assert answer == FALLBACK and verify_task is None


class _ASR:
    async def start(self, partial, final): self.partial, self.final = partial, final
    async def close(self): pass


class _Transport:
    def __init__(self): self.events, self.pcm = [], []
    async def event(self, value): self.events.append(value)
    async def audio(self, value, generation): self.pcm.append(value)


class _CountingTTS:
    def __init__(self): self.calls = []
    async def synthesize_stream(self, text):
        self.calls.append(text)
        yield b"X"


async def _wait_for_audio_end(transport):
    for _ in range(300):
        if any(e.get("type") == "audio_end" for e in transport.events):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("audio_end not observed in time")


@pytest.mark.asyncio
async def test_callsession_uses_merged_route_and_answer_with_real_scoped_orchestrator():
    """End-to-end: with the real ScopedOrchestrator (which now exposes
    `route_and_answer_speak_first`), `_run_pipeline` must pick the merged
    one-call path -- confirmed by the generator chat being invoked exactly
    once for a RAG turn, not once for routing and again for drafting."""
    good = {"action": "RAG", "answer": "verified answer", "cited_articles": ["1"], "claims": [{"text": "t", "article": "1", "quote": "official text"}], "insufficient_context": False}
    chat = StreamingFakeChat({"action": "RAG", "answer": "verified answer"}, good)
    orch = Orchestrator(Retriever(), AnswerGenerator(chat), PassChecker())
    scoped = ScopedOrchestrator(orch, FakeChat({}))
    transport, tts = _Transport(), _CountingTTS()
    session = CallSession(transport, _ASR(), scoped, tts, speak_first=True)
    await session.start()
    await session.asr.final("Как мне оформить Kaspi Gold")
    await _wait_for_audio_end(transport)
    assert chat.streaming_calls == [ROUTED_GENERATOR], "exactly one combined route+draft call, not a separate router call plus a generator call"
    answers = [e["text"] for e in transport.events if e.get("type") == "answer"]
    assert answers == ["verified answer"]
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()
