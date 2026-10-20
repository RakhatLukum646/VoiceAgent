import asyncio
from array import array
import pytest
from app.agents.checker import AnswerChecker
from app.agents.orchestrator import Orchestrator
from app.agents.prompts import FALLBACK
from app.call.session import CallSession
from app.call.utterances import is_contentful, is_filler, is_stop
from app.audio.vad import RMSVAD
from app.observability.latency import TurnMetrics
from app.rag.chunking import Passage, chunk_articles
from app.rag.retriever import Retriever
from app.rag.vector_store import SQLiteVectorStore


class Embeddings:
    async def embed(self, texts):
        return [[1.0, 0.0] if "прав" in t else [0.0, 1.0] for t in texts]


@pytest.mark.asyncio
async def test_semantic_retrieval_metadata(tmp_path):
    store = SQLiteVectorStore(str(tmp_path / "idx.db"))
    store.replace([(Passage("Constitution", "12", "Раздел II", "права человека", "official"), [1.0, 0.0]), (Passage("Constitution", "1", "Раздел I", "государство", "official"), [0.0, 1.0])], "model")
    hits = await Retriever(Embeddings(), store, "model").search("права")
    assert hits[0]["article"] == "12" and hits[0]["score"] == 1
    assert store.count_for_model("wrong") == 0


@pytest.mark.asyncio
async def test_hybrid_lexical_and_exact_article_paths(tmp_path):
    store = SQLiteVectorStore(str(tmp_path / "hybrid.db"))
    store.replace([
        (Passage("C", "23", "II", "обычный текст", "s"), [1.0, 0.0]),
        (Passage("C", "24", "II", "уникальный термин передвижения", "s"), [0.0, 1.0]),
        (Passage("C", "25", "II", "соседняя статья", "s"), [0.0, 1.0]),
    ], "model")
    lexical = store.search([1.0, 0.0], "model", 1, "уникальный термин")
    assert lexical[0]["article"] == "24" and lexical[0]["lexical_match"] and lexical[0]["score"] == 0
    exact = store.search([1.0, 0.0], "model", 1, "Что говорит статья 25?")
    assert exact[0]["article"] == "25" and exact[0]["exact_article_match"]


@pytest.mark.asyncio
async def test_verifier_search_is_question_only_and_includes_neighbors(tmp_path):
    store = SQLiteVectorStore(str(tmp_path / "verify.db"))
    store.replace([(Passage("C", str(n), "II", f"Статья {n} текст", "s"), [1.0, 0.0]) for n in (23, 24, 25)], "model")
    class RecordingEmbeddings:
        def __init__(self): self.texts = []
        async def embed(self, texts):
            self.texts.extend(texts)
            return [[1.0, 0.0] for _ in texts]
    embedding = RecordingEmbeddings()
    retriever = Retriever(embedding, store, "model", top_k=1)
    hits = await retriever.verifier_search("Что говорит статья 24?")
    assert embedding.texts == ["Что говорит статья 24?"]
    assert {"23", "24", "25"}.issubset({h["article"] for h in hits})


def test_chunk_keeps_header_and_section():
    text = "Раздел II. Человек\nСтатья 12\nВведение.\n1. Первое предложение.\n2. Второе предложение."
    chunks = chunk_articles(text, "official", max_chars=30)
    assert chunks[0].article == "12"
    assert chunks[0].section.startswith("Раздел II")
    assert "Статья 12" in chunks[0].text


class FakeChat:
    def __init__(self, response): self.response = response
    async def json_completion(self, *args, **kwargs): return self.response, None


@pytest.mark.asyncio
async def test_checker_rejects_inconsistent_pass_and_invalid_citation():
    checker = AnswerChecker(FakeChat({"grounded": True, "relevant": True, "article_references_correct": True, "safe_to_send": True, "complete": True, "missing_conditions": [], "unsupported_claims": [], "confidence": 0.9, "action": "PASS"}))
    result = await checker.check("?", [{"article": "1", "section": "I", "text": "text", "source": "s", "score": 1}], {"answer": "text", "cited_articles": ["999"]})
    assert result["action"] == "RETRY"
    assert (await checker.check("?", [], {}))["action"] == "RETRY"
    malformed = await AnswerChecker(FakeChat({"action": "PASS"})).check("?", [{"article": "1", "section": "I", "text": "text", "source": "s", "score": 1}], {"answer": "text", "cited_articles": ["1"], "claims": [{"text": "text", "article": "1", "quote": "text"}]})
    assert malformed["action"] == "RETRY" and "complete" in malformed["schema_errors"]
    class InvalidJSON:
        async def json_completion(self, *args, **kwargs): raise ValueError("Malformed provider JSON")
    parse_failure = await AnswerChecker(InvalidJSON()).check("?", [{"article": "1", "section": "I", "text": "text", "source": "s", "score": 1}], {"answer": "text", "cited_articles": ["1"], "claims": [{"text": "text", "article": "1", "quote": "text"}]})
    assert parse_failure["action"] == "RETRY" and parse_failure["schema_errors"] == ["response"]


@pytest.mark.asyncio
async def test_checker_only_allows_validated_quoted_cross_reference():
    quote = "Курултай не может быть распущен в случае, установленном пунктом 3 статьи 51 Конституции."
    article62 = {"article": "62", "section": "V", "text": quote, "source": "s", "score": 1}
    article51 = {"article": "51", "section": "V", "text": "Статья 51. Досрочное прекращение полномочий.", "source": "s", "score": 1}
    class Lookup:
        async def lookup_articles(self, articles):
            return [hit for hit in (article62, article51) if hit["article"] in articles]
        async def search(self, text): return []
    verdict = {"grounded": True, "relevant": True, "article_references_correct": True, "safe_to_send": True, "complete": True, "missing_conditions": [], "unsupported_claims": [], "confidence": .9, "action": "PASS"}
    checker = AnswerChecker(FakeChat(verdict), Lookup())
    candidate = {"answer": "Курултай не может быть распущен в случае, установленном пунктом 3 статьи 51 Конституции (статья 62).", "cited_articles": ["62"], "claims": [{"text": "Ограничение роспуска", "article": "62", "quote": quote}]}
    passed = await checker.check("Когда?", [article62], candidate)
    assert passed["action"] == "PASS" and passed["verified_cross_references"] == ["51"]
    fake = {**candidate, "answer": candidate["answer"] + " Также статья 999."}
    assert (await checker.check("Когда?", [article62], fake))["rejection_reason"] == "uncited_article"
    no_target = AnswerChecker(FakeChat(verdict), type("LookupWithout51", (), {"lookup_articles": lambda self, articles: asyncio.sleep(0, result=[article62] if "62" in articles else []), "search": lambda self, text: asyncio.sleep(0, result=[])})())
    assert (await no_target.check("Когда?", [article62], candidate))["rejection_reason"] == "uncited_article"


@pytest.mark.asyncio
async def test_retry_once_then_fail_closed():
    class Search:
        async def search(self, q): return [{"article": "1", "section": "I", "text": "text", "source": "s", "score": 0.9}]
    class Generator:
        calls = 0
        async def generate(self, *args):
            self.calls += 1
            return {"answer": "unsupported", "cited_articles": ["1"], "insufficient_context": False}, 2
    class Checker:
        async def check(self, *args): return {"action": "RETRY", "unsupported_claims": ["unsupported"]}
    generator = Generator()
    async def state(s): pass
    answer, sources = await Orchestrator(Search(), generator, Checker()).answer("?", [], TurnMetrics("c", 1), state)
    assert answer == FALLBACK and generator.calls == 2 and sources == []


class Transport:
    def __init__(self): self.events=[]; self.audio_frames=[]
    async def event(self, payload): self.events.append(payload)
    async def audio(self, pcm, generation): self.audio_frames.append((pcm, generation))


class ASR:
    async def start(self, partial, final): self.partial=partial; self.final=final
    async def send_audio(self, pcm): pass
    async def finalize(self): pass
    async def close(self): pass


class SlowTTS:
    async def synthesize_stream(self, text):
        yield b"\x00\x00"
        await asyncio.sleep(10)


@pytest.mark.asyncio
async def test_barge_in_cancels_and_excludes_unplayed_history():
    class Answer:
        async def answer(self, q, history, metrics, state): return "verified", []
    transport=Transport()
    session=CallSession(transport, ASR(), Answer(), SlowTTS())
    await session.start()
    await session.asr.final("first")
    await asyncio.sleep(0.01)
    old=session.generation
    assert transport.audio_frames
    await session.interrupt()
    assert session.generation>old and not session.history
    assert transport.events[-1]["type"] == "interrupt"
    await session.close()


def test_monotonic_latency():
    m=TurnMetrics("call", 2)
    m.mark("a"); m.mark("b")
    assert m.duration("d", "a", "b") >= 0
    assert m.as_event()["turn_id"] == 2


def test_vad_requires_sustained_voice_and_preserves_hysteresis():
    vad = RMSVAD()
    voice = array("h", [700] * 1600).tobytes()
    silence = array("h", [0] * 1600).tobytes()
    assert vad.feed(voice) == (False, False)
    assert vad.feed(silence) == (False, False)
    assert vad.feed(voice) == (False, False)
    assert vad.feed(voice) == (True, False)
    assert vad.speaking


def test_filler_filter_preserves_real_short_questions_and_stop():
    assert is_filler("Вот вот этот вот, что")
    assert not is_contentful("вот вот этот вот, ч")
    assert not is_contentful("угу")
    assert is_contentful("какая статья") and is_contentful("почему")
    assert not is_filler("да")
    assert is_stop("стоп")


@pytest.mark.asyncio
async def test_checker_rejects_fabricated_quote_and_missing_condition():
    passage = {"article": "24", "section": "II", "text": "Каждый человек имеет право свободно передвигаться по территории Республики Казахстан, за исключением случаев, установленных законом.", "source": "official", "score": 0.8}
    candidate = {"answer": "Передвижение свободно.", "cited_articles": ["24"], "claims": [{"text": "Передвижение свободно", "article": "24", "quote": "Каждый человек имеет право свободно передвигаться"}]}
    passed = {"grounded": True, "relevant": True, "article_references_correct": True, "safe_to_send": True, "complete": False, "missing_conditions": ["кроме случаев, установленных законом"], "unsupported_claims": [], "confidence": 0.9, "action": "PASS"}
    checker = AnswerChecker(FakeChat(passed))
    assert (await checker.check("Можно ли свободно передвигаться?", [passage], candidate))["action"] == "RETRY"
    candidate["claims"][0]["quote"] = "несуществующая цитата"
    assert (await checker.check("?", [passage], candidate))["action"] == "RETRY"
    candidate["claims"][0]["quote"] = "Каждый человек имеет право свободно передвигаться"
    candidate["answer"] = "Статья 99 также гарантирует это."
    assert (await checker.check("?", [passage], candidate))["action"] == "RETRY"


@pytest.mark.asyncio
async def test_independent_evidence_is_merged_for_retry():
    first = {"article": "1", "section": "I", "text": "Первый фрагмент", "source": "s", "score": 0.9}
    second = {"article": "2", "section": "I", "text": "Недостающее условие", "source": "s", "score": 0.7}
    class Search:
        calls = 0
        async def search(self, q):
            self.calls += 1
            return [first]
        async def verifier_search(self, q): return [second]
    class Generator:
        def __init__(self): self.seen = []
        async def generate(self, q, history, hits, feedback):
            self.seen.append(([h["article"] for h in hits], feedback))
            return {"answer": "ok", "cited_articles": ["1"], "claims": [{"text": "ok", "article": "1", "quote": "Первый фрагмент"}], "insufficient_context": False}, None
    class Checker:
        calls = 0
        async def check(self, q, hits, candidate):
            self.calls += 1
            assert hits == [second]
            return {"action": "RETRY", "missing_conditions": ["condition"], "extra_evidence": [second]}
    generator = Generator()
    async def state(s): pass
    answer, _ = await Orchestrator(Search(), generator, Checker()).answer("?", [], TurnMetrics("c", 1), state)
    assert answer == FALLBACK
    assert generator.seen[0][0] == ["1"]
    assert generator.seen[1][0] == ["1", "2"]
    assert generator.seen[1][1]["missing_conditions"] == ["condition"]


@pytest.mark.asyncio
async def test_retry_repairs_answer_then_passes_with_independent_evidence():
    first = {"article": "24", "section": "II", "text": "Свободное передвижение", "source": "s", "score": 0.9}
    condition = {"article": "24", "section": "II", "text": "кроме случаев, установленных законом", "source": "s", "score": 0.1}
    class Search:
        async def search(self, q): return [first]
        async def verifier_search(self, q): return [condition]
    class Generator:
        def __init__(self): self.calls = []
        async def generate(self, q, history, hits, feedback):
            self.calls.append(([h["text"] for h in hits], feedback))
            answer = "Свободное передвижение" if feedback is None else "Свободное передвижение, кроме случаев, установленных законом"
            return {"answer": answer, "cited_articles": ["24"], "claims": [{"text": answer, "article": "24", "quote": answer}], "insufficient_context": False}, None
    class Checker:
        def __init__(self): self.calls = 0
        async def check(self, q, hits, candidate):
            self.calls += 1
            if self.calls == 1:
                return {"action": "RETRY", "missing_conditions": ["кроме случаев"], "extra_evidence": [condition]}
            assert "кроме случаев" in candidate["answer"]
            return {"action": "PASS", "extra_evidence": [condition]}
    generator = Generator()
    async def state(s): pass
    answer, hits = await Orchestrator(Search(), generator, Checker()).answer("?", [], TurnMetrics("c", 1), state)
    assert "кроме случаев" in answer
    assert condition in hits
    assert len(generator.calls) == 2 and condition["text"] in generator.calls[1][0]


@pytest.mark.asyncio
async def test_insufficient_first_attempt_retries_with_independent_article_and_returns_only_citation():
    unrelated = {"article": "1", "section": "I", "text": "Государство", "source": "s", "score": 0.9}
    relevant = {"article": "62", "section": "IV", "text": "Президент вправе распустить Курултай после консультаций", "source": "s", "score": 0.2}
    class Search:
        async def search(self, q): return [unrelated]
        async def verifier_search(self, q): return [relevant]
    class Generator:
        calls = 0
        async def generate(self, q, history, hits, feedback):
            self.calls += 1
            if self.calls == 1:
                return {"answer": "Не знаю", "cited_articles": [], "claims": [], "insufficient_context": True}, None
            assert relevant in hits and feedback["action"] == "RETRY"
            return {"answer": "Президент может распустить Курултай после консультаций", "cited_articles": ["62"], "claims": [{"text": "Президент может распустить Курултай", "article": "62", "quote": relevant["text"]}], "insufficient_context": False}, None
    class Checker:
        async def check(self, q, hits, candidate): return {"action": "PASS", "extra_evidence": [relevant]}
    async def state(s): pass
    answer, sources = await Orchestrator(Search(), Generator(), Checker()).answer("Может ли президент распустить парламент?", [], TurnMetrics("c", 1), state)
    assert "Президент" in answer and [h["article"] for h in sources] == ["62"]


@pytest.mark.asyncio
async def test_unchecked_answer_never_reaches_tts():
    class Search:
        async def search(self, q): return [{"article": "1", "section": "I", "text": "Текст", "source": "s", "score": 0.9}]
    class Generator:
        async def generate(self, *args): return {"answer": "unchecked", "cited_articles": ["1"], "claims": [{"text": "unchecked", "article": "1", "quote": "fabricated"}], "insufficient_context": False}, None
    class TTS:
        def __init__(self): self.texts = []
        async def synthesize_stream(self, text):
            self.texts.append(text)
            yield b"\0\0"
    tts = TTS()
    session = CallSession(Transport(), ASR(), Orchestrator(Search(), Generator(), AnswerChecker(FakeChat({}))), tts)
    await session.start()
    await session.asr.final("question")
    for _ in range(100):
        if any(e["type"] == "audio_end" for e in session.transport.events): break
        await asyncio.sleep(0.01)
    assert session.transport.events[-1]["type"] == "audio_end"
    assert tts.texts == [CallSession.ACKNOWLEDGEMENT, FALLBACK]
    assert all(text != "unchecked" for text in tts.texts)
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_fast_answer_does_not_wait_for_stalled_ack():
    class Answer:
        async def answer(self, q, history, metrics, state): return "verified", []
    class TTS:
        async def synthesize_stream(self, text):
            if text == CallSession.ACKNOWLEDGEMENT:
                await asyncio.sleep(30)
            yield b"\0\0"
    transport = Transport()
    session = CallSession(transport, ASR(), Answer(), TTS())
    await session.start()
    await session.asr.final("question")
    for _ in range(150):
        if any(e["type"] == "audio_end" for e in transport.events): break
        await asyncio.sleep(0.01)
    assert any(e["type"] == "audio_end" for e in transport.events)
    assert not any(e["type"] == "acknowledgement" for e in transport.events)
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_resumed_speech_during_delayed_final_keeps_one_asr_begin():
    voice = array("h", [1000] * 320).tobytes()
    silence = array("h", [0] * 320).tobytes()
    class WaitingASR(ASR):
        def __init__(self):
            self.begins = 0
            self.waiting = asyncio.Event()
            self.release = asyncio.Event()
        async def begin_utterance(self):
            self.begins += 1
            if self.begins == 2:
                self.waiting.set()
                await self.release.wait()
    class Answer:
        async def answer(self, q, history, metrics, state): return q, []
    class TTS:
        async def synthesize_stream(self, text): yield b"\0\0"
    asr, transport = WaitingASR(), Transport()
    session = CallSession(transport, asr, Answer(), TTS())
    await session.start()
    for _ in range(8): await session.feed_audio(voice)
    for _ in range(28): await session.feed_audio(silence)
    for _ in range(7): await session.feed_audio(voice)
    resumed = asyncio.create_task(session.feed_audio(voice))
    await asyncio.wait_for(asr.waiting.wait(), 1)
    await asr.final("первая часть")
    assert session.vad.speaking and session.turn_id == 1
    asr.release.set()
    await resumed
    await session.feed_audio(voice)
    assert asr.begins == 2
    for _ in range(28): await session.feed_audio(silence)
    await asr.final("вторая часть")
    assert session.turn_id == 2
    assert [e["text"] for e in transport.events if e["type"] == "final"] == ["первая часть", "вторая часть"]
    await session.close()


@pytest.mark.asyncio
async def test_false_vad_starts_and_empty_finals_do_not_starve_answer():
    voice = array("h", [700] * 1600).tobytes()
    silence = array("h", [0] * 1600).tobytes()
    class WaitingASR(ASR):
        begins = 0
        async def begin_utterance(self): self.begins += 1
    class Answer:
        async def answer(self, q, history, metrics, state):
            await asyncio.sleep(0.05)
            return "verified", []
    class TTS:
        async def synthesize_stream(self, text): yield b"\0\0"
    asr, transport = WaitingASR(), Transport()
    session = CallSession(transport, asr, Answer(), TTS())
    await session.start()
    for _ in range(2): await session.feed_audio(voice)
    for _ in range(6): await session.feed_audio(silence)
    await asr.final("valid question")
    generation = session.generation
    assert session.turn_id == 1
    for _ in range(3):
        for _ in range(2): await session.feed_audio(voice)
        for _ in range(6): await session.feed_audio(silence)
        await asr.final("")
        assert session.generation == generation and session.turn_id == 1
    for _ in range(100):
        if any(e["type"] == "audio_end" for e in transport.events): break
        await asyncio.sleep(0.01)
    assert any(e["type"] == "audio_end" for e in transport.events)
    assert not any(e["type"] == "error" for e in transport.events)
    session.acknowledge_playback(session.generation)
    await session.task
    assert session.history[-1]["content"] == "verified"
    await session.close()


@pytest.mark.asyncio
async def test_pending_asr_audio_is_bounded_without_killing_answer_or_next_turn():
    voice = array("h", [700] * 1600).tobytes()
    silence = array("h", [0] * 1600).tobytes()
    class DelayedASR(ASR):
        def __init__(self):
            self.begins = 0
            self.release = asyncio.Event()
            self.waiting = asyncio.Event()
            self.sent = []
        async def begin_utterance(self):
            self.begins += 1
            if self.begins == 2:
                self.waiting.set()
                await self.release.wait()
        async def send_audio(self, pcm): self.sent.append(pcm)
    class Answer:
        async def answer(self, q, history, metrics, state): return q, []
    class TTS:
        async def synthesize_stream(self, text): yield b"\0\0"
    asr, transport = DelayedASR(), Transport()
    session = CallSession(transport, asr, Answer(), TTS())
    session._asr_pending_limit = 6400
    await session.start()
    for _ in range(2): await session.feed_audio(voice)
    for _ in range(6): await session.feed_audio(silence)
    await asyncio.wait_for(session._asr_begin_task, 1)
    await asr.final("first question")
    for _ in range(2): await session.feed_audio(voice)
    await asyncio.wait_for(asr.waiting.wait(), 1)
    for _ in range(10): await session.feed_audio(voice)
    assert len(session._asr_pending_audio) <= 6400
    for _ in range(6): await session.feed_audio(silence)
    assert session.turn_id == 1
    asr.release.set()
    await asyncio.wait_for(session._asr_begin_task, 1)
    await asr.final("")
    for _ in range(100):
        if any(e["type"] == "audio_end" for e in transport.events): break
        await asyncio.sleep(0.01)
    assert any(e["type"] == "audio_end" for e in transport.events)
    assert not any(e["type"] == "error" for e in transport.events)
    session.acknowledge_playback(session.generation)
    await session.task
    assert session.history[-1]["content"] == "first question"
    for _ in range(2): await session.feed_audio(voice)
    for _ in range(6): await session.feed_audio(silence)
    await asyncio.wait_for(session._asr_begin_task, 1)
    await asr.final("second question")
    assert asr.begins == 3 and session.turn_id == 2
    await session.close()


@pytest.mark.asyncio
async def test_filler_during_pending_answer_does_not_cancel_or_add_ack():
    started, release = asyncio.Event(), asyncio.Event()
    class Answer:
        async def answer(self, q, history, metrics, state):
            started.set()
            await release.wait()
            return "verified", []
    class TTS:
        def __init__(self): self.texts = []
        async def synthesize_stream(self, text):
            self.texts.append(text)
            yield b"\0\0"
    tts, transport = TTS(), Transport()
    session = CallSession(transport, ASR(), Answer(), tts)
    await session.start()
    question = "Может ли президент распустить парламент?"
    await session.asr.final(question)
    await asyncio.wait_for(started.wait(), 1)
    pending = session.task
    session._barge_pending = True
    await session.asr.partial("вот вот этот вот, ч")
    await session.asr.final("Вот вот этот вот, что")
    assert session.task is pending and not pending.done() and session.turn_id == 1
    assert len([e for e in transport.events if e["type"] == "final"]) == 1
    release.set()
    for _ in range(100):
        if any(e["type"] == "audio_end" for e in transport.events): break
        await asyncio.sleep(0.01)
    session.acknowledge_playback(session.generation)
    await pending
    assert session.history[0]["content"] == question
    assert tts.texts == [CallSession.acknowledgement_for(question), "verified"]
    assert CallSession.acknowledgement_for(question).startswith("Проверяю ваш вопрос:")
    await session.close()


@pytest.mark.asyncio
async def test_meaningful_partial_barges_in_and_stop_cancels():
    started = asyncio.Event()
    class Answer:
        async def answer(self, q, history, metrics, state):
            started.set()
            await asyncio.sleep(30)
    transport = Transport()
    session = CallSession(transport, ASR(), Answer(), SlowTTS())
    await session.start()
    await session.asr.final("Может ли президент распустить парламент?")
    await asyncio.wait_for(started.wait(), 1)
    first_generation = session.generation
    session._barge_pending = True
    await session.asr.partial("А какие ограничения?")
    assert session.generation > first_generation and session.task is None
    await session.asr.final("Какие ограничения?")
    await asyncio.wait_for(started.wait(), 1)
    await session.asr.final("стоп")
    assert session.task is None and session.turn_id == 2
    await session.close()


@pytest.mark.asyncio
async def test_history_only_after_playback_ack():
    class Answer:
        async def answer(self, q, history, metrics, state): return "verified", []
    class TTS:
        async def synthesize_stream(self, text): yield b"\x00\x00"
    transport = Transport()
    session = CallSession(transport, ASR(), Answer(), TTS())
    await session.start()
    await session.asr.final("question")
    for _ in range(100):
        if any(e["type"] == "audio_end" for e in transport.events): break
        await asyncio.sleep(0.01)
    assert not session.history
    assert any(e["type"] == "audio_end" for e in transport.events)
    session.acknowledge_playback(session.generation)
    await session.task
    assert session.history[-1]["content"] == "verified"
    assert any(e["type"] == "metrics" for e in transport.events)
    await session.close()
