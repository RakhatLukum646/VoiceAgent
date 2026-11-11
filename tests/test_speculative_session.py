import asyncio
import struct
import pytest
from app.call.session import CallSession
from app.agents.orchestrator import Orchestrator
from app.agents.generator import AnswerGenerator
from app.agents.prompts import FALLBACK


class ASR:
    async def start(self,partial,final):self.partial,self.final=partial,final
    async def close(self):pass


class PartialAwareASR:
    async def start(self, partial, final): self.partial, self.final = partial, final
    async def send_audio(self, pcm): pass
    async def finalize(self): pass
    async def close(self): pass


def _tone(ms, amplitude):
    samples = int(16000 * ms / 1000)
    return struct.pack("<%dh" % samples, *([amplitude] * samples))


class FakePrepared:
    def __init__(self, question, owner):
        self.question, self.owner, self.timings, self.closed = question, owner, {}, False
    async def close(self):
        self.closed = True
        self.owner.closed.append(self.question)
    async def search(self, q=None): return []
    async def verifier_search(self, q=None): return []


class CountingTTS:
    def __init__(self):
        self.calls = []
    async def synthesize_stream(self, text):
        self.calls.append(text)
        yield b"X"


class PrepareCountingOrchestrator:
    """No `.route` attribute: `_respond` skips routing and calls `.answer`
    directly, isolating the prefetch-reuse wiring from routing concerns."""
    def __init__(self):
        self.prepare_calls, self.closed = [], []
    def prepare(self, question, history):
        self.prepare_calls.append(question)
        return FakePrepared(question, self)
    async def answer(self, question, history, metrics, state, prepared=None, candidate_audio=None):
        return "the answer", []


class Transport:
    def __init__(self):self.events=[];self.pcm=[]
    async def event(self,value):self.events.append(value)
    async def audio(self,value,generation):self.pcm.append(value)


class SlowAudioTransport:
    """Each `.audio()` call takes real (if small) time, so a multi-chunk ACK
    takes a measurable duration to fully play out if nothing interrupts it."""
    def __init__(self):
        self.events, self.pcm = [], []
        self.clear_audio_calls = []
    async def event(self, value): self.events.append(value)
    async def audio(self, value, generation):
        await asyncio.sleep(0.02)
        self.pcm.append((value, generation))
    async def clear_audio(self, generation):
        self.clear_audio_calls.append(generation)


class FakeTTS:
    async def synthesize_stream(self, text):
        yield b'X'


async def _wait_for_audio_end(transport):
    for _ in range(200):
        if any(e.get('type') == 'audio_end' for e in transport.events):
            return
        await asyncio.sleep(0.01)
    raise AssertionError('audio_end not observed in time')


@pytest.mark.asyncio
async def test_full_response_queue_pending_checker_no_secret_pcm_then_barge_in_discards():
    blocked=asyncio.Event();cancelled=asyncio.Event()
    hit={'article':'a','text':'official','source':'s','section':'title','score':1.}
    class Prepared:
        timings={}
        async def search(self,q):return [hit]
        async def verifier_search(self,q):return [hit]
        async def close(self):pass
    class Generator:
        async def generate(self,*args):return {'answer':'private candidate','cited_articles':['a'],'claims':[{'text':'fact','article':'a','quote':'official'}],'insufficient_context':False},None
    class Checker:
        async def check(self,*args,**kwargs):blocked.set();await asyncio.sleep(60)
    class TTS:
        async def synthesize_stream(self,text):
            if text!='private candidate':yield b'ACK';return
            try:
                for _ in range(300):yield b'SECRET'*530
            finally:cancelled.set()
    rag=Orchestrator(Prepared(),Generator(),Checker())
    class Scoped:
        def prepare(self,*args):return Prepared()
        async def answer(self,*args,**kwargs):return await rag.answer(*args,**kwargs)
    transport=Transport();session=CallSession(transport,ASR(),Scoped(),TTS())
    await session.start();await session.asr.final('test question')
    await asyncio.wait_for(blocked.wait(),1);await asyncio.sleep(.02)
    assert all(b'SECRET' not in item for item in transport.pcm)
    await asyncio.wait_for(session.interrupt(),1)
    assert cancelled.is_set() and session.history==[]
    assert all(b'SECRET' not in item for item in transport.pcm)
    await session.close()


@pytest.mark.asyncio
async def test_speak_first_corrects_rejected_candidate_and_history_keeps_final_text():
    hit = {'article': 'a', 'text': 'official', 'source': 's', 'section': 'title', 'score': 1.}
    class Prepared:
        timings = {}
        async def search(self, q): return [hit]
        async def verifier_search(self, q): return [hit]
        async def close(self): pass
    class Generator:
        async def generate(self, *args): return {'answer': 'unverified draft', 'cited_articles': ['a'], 'claims': [{'text': 'fact', 'article': 'a', 'quote': 'official'}], 'insufficient_context': False}, None
    class Checker:
        async def check(self, *args, **kwargs): return {'action': 'RETRY'}
    rag = Orchestrator(Prepared(), Generator(), Checker())
    class Scoped:
        def prepare(self, *args): return Prepared()
        async def answer_speak_first(self, *args, **kwargs): return await rag.answer_speak_first(*args, **kwargs)
    transport = Transport()
    session = CallSession(transport, ASR(), Scoped(), FakeTTS(), speak_first=True)
    await session.start()
    await session.asr.final('test question')
    await _wait_for_audio_end(transport)
    answers = [e['text'] for e in transport.events if e.get('type') == 'answer']
    assert answers[0] == 'unverified draft'
    assert answers[-1] == CallSession.CORRECTION_BRIDGE + FALLBACK
    session.acknowledge_playback(session.generation)
    await session.task
    assert session.history and session.history[-1]['content'] == FALLBACK
    await session.close()


@pytest.mark.asyncio
async def test_speak_first_no_correction_when_checker_passes():
    hit = {'article': 'a', 'text': 'official', 'source': 's', 'section': 'title', 'score': 1.}
    class Prepared:
        timings = {}
        async def search(self, q): return [hit]
        async def verifier_search(self, q): return [hit]
        async def close(self): pass
    class Generator:
        async def generate(self, *args): return {'answer': 'verified draft', 'cited_articles': ['a'], 'claims': [{'text': 'fact', 'article': 'a', 'quote': 'official'}], 'insufficient_context': False}, None
    class Checker:
        async def check(self, *args, **kwargs): return {'action': 'PASS'}
    rag = Orchestrator(Prepared(), Generator(), Checker())
    class Scoped:
        def prepare(self, *args): return Prepared()
        async def answer_speak_first(self, *args, **kwargs): return await rag.answer_speak_first(*args, **kwargs)
    transport = Transport()
    session = CallSession(transport, ASR(), Scoped(), FakeTTS(), speak_first=True)
    await session.start()
    await session.asr.final('test question')
    await _wait_for_audio_end(transport)
    answers = [e['text'] for e in transport.events if e.get('type') == 'answer']
    assert answers == ['verified draft']
    session.acknowledge_playback(session.generation)
    await session.task
    assert session.history and session.history[-1]['content'] == 'verified draft'
    await session.close()


@pytest.mark.asyncio
async def test_speak_first_draft_audio_synthesized_once_not_twice():
    hit = {'article': 'a', 'text': 'official', 'source': 's', 'section': 'title', 'score': 1.}
    class Prepared:
        timings = {}
        async def search(self, q): return [hit]
        async def verifier_search(self, q): return [hit]
        async def close(self): pass
    class Generator:
        async def generate(self, *args): return {'answer': 'verified draft', 'cited_articles': ['a'], 'claims': [{'text': 'fact', 'article': 'a', 'quote': 'official'}], 'insufficient_context': False}, None
    class Checker:
        async def check(self, *args, **kwargs): return {'action': 'PASS'}
    rag = Orchestrator(Prepared(), Generator(), Checker())
    class Scoped:
        def prepare(self, *args): return Prepared()
        async def answer_speak_first(self, *args, **kwargs): return await rag.answer_speak_first(*args, **kwargs)
    transport = Transport()
    tts = CountingTTS()
    session = CallSession(transport, ASR(), Scoped(), tts, speak_first=True)
    await session.start()
    await session.asr.final('test question')
    await _wait_for_audio_end(transport)
    session.acknowledge_playback(session.generation)
    await session.task
    assert tts.calls.count('verified draft') == 1, "the draft must be synthesized once (speculatively, during checking), not again in _speak_audio"
    await session.close()


@pytest.mark.asyncio
async def test_non_speak_first_pass_audio_synthesized_once_not_twice():
    hit = {'article': 'a', 'text': 'official', 'source': 's', 'section': 'title', 'score': 1.}
    class Prepared:
        timings = {}
        async def search(self, q): return [hit]
        async def verifier_search(self, q): return [hit]
        async def close(self): pass
    class Generator:
        async def generate(self, *args): return {'answer': 'verified answer', 'cited_articles': ['a'], 'claims': [{'text': 'fact', 'article': 'a', 'quote': 'official'}], 'insufficient_context': False}, None
    class Checker:
        async def check(self, *args, **kwargs): return {'action': 'PASS'}
    rag = Orchestrator(Prepared(), Generator(), Checker())
    class Scoped:
        def prepare(self, *args): return Prepared()
        async def answer(self, *args, **kwargs): return await rag.answer(*args, **kwargs)
    transport = Transport()
    tts = CountingTTS()
    session = CallSession(transport, ASR(), Scoped(), tts, speak_first=False)
    await session.start()
    await session.asr.final('test question')
    await _wait_for_audio_end(transport)
    session.acknowledge_playback(session.generation)
    await session.task
    assert tts.calls.count('verified answer') == 1, "a PASS-ed candidate must be synthesized once (during checking), not again in _speak_audio"
    await session.close()


@pytest.mark.asyncio
async def test_vad_silence_prefetches_retrieval_and_final_reuses_it_on_matching_text():
    orchestrator = PrepareCountingOrchestrator()
    transport = Transport()
    session = CallSession(transport, PartialAwareASR(), orchestrator, FakeTTS())
    await session.start()
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))  # VAD start-of-speech
    await session._partial(text)                # ASR partial lands mid-utterance
    await session.feed_audio(_tone(700, 0))      # VAD end-of-speech (>550ms silence)
    assert orchestrator.prepare_calls == [text]
    await session._final(text)
    await _wait_for_audio_end(transport)
    assert orchestrator.prepare_calls == [text], "final must reuse the prefetched retrieval, not re-prepare"
    assert orchestrator.closed == [], "the reused prefetch must be closed by _respond's own cleanup, not discarded early"
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()
    assert text in orchestrator.closed


@pytest.mark.asyncio
async def test_vad_silence_prefetch_reused_when_final_differs_only_in_case_and_punctuation():
    orchestrator = PrepareCountingOrchestrator()
    transport = Transport()
    session = CallSession(transport, PartialAwareASR(), orchestrator, FakeTTS())
    await session.start()
    partial_text, final_text = "как мне оформить kaspi gold", "Как мне оформить Kaspi Gold?"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(partial_text)
    await session.feed_audio(_tone(700, 0))
    await session._final(final_text)
    await _wait_for_audio_end(transport)
    assert orchestrator.prepare_calls == [partial_text], "punctuation/case-only difference must reuse the prefetch"
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


def test_same_utterance_normalization():
    from app.call.utterances import same_utterance
    assert same_utterance("Я потерял карту, что делать?", "я потерял карту что делать")
    assert same_utterance("Ещё вопрос", "еще вопрос")
    assert same_utterance("Снять 4 миллиона", "снять 4 миллиона.")
    assert not same_utterance("Как мне оформить Kaspi", "Как мне оформить Kaspi Gold")
    assert not same_utterance("Снять 4 миллиона", "Снять 5 миллионов")


@pytest.mark.asyncio
async def test_vad_silence_prefetch_discarded_when_final_text_differs():
    orchestrator = PrepareCountingOrchestrator()
    transport = Transport()
    session = CallSession(transport, PartialAwareASR(), orchestrator, FakeTTS())
    await session.start()
    partial_text, final_text = "Как мне оформить Kaspi", "Как мне оформить Kaspi Gold для ребенка"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(partial_text)
    await session.feed_audio(_tone(700, 0))
    assert orchestrator.prepare_calls == [partial_text]
    await session._final(final_text)
    await _wait_for_audio_end(transport)
    assert orchestrator.prepare_calls == [partial_text, final_text], "mismatched text must trigger a fresh prepare()"
    for _ in range(100):
        if partial_text in orchestrator.closed:
            break
        await asyncio.sleep(0.01)
    assert partial_text in orchestrator.closed, "the stale prefetch must still be closed, not leaked"
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_ack_pcm_rendered_once_per_call_not_per_turn():
    orchestrator = PrepareCountingOrchestrator()
    transport = Transport()
    tts = CountingTTS()
    session = CallSession(transport, PartialAwareASR(), orchestrator, tts)
    await session.start()
    await asyncio.sleep(0.01)  # let the call-start render task actually run
    assert tts.calls.count(CallSession.ACKNOWLEDGEMENT) == 1, "the static ACK phrase is rendered once, at start(), not per turn"
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    await session.feed_audio(_tone(700, 0))
    await session._final(text)
    await _wait_for_audio_end(transport)
    assert tts.calls.count(CallSession.ACKNOWLEDGEMENT) == 1, "a real turn must play the cached PCM, not synthesize the ACK again"
    ack_events = [e for e in transport.events if e.get("type") == "acknowledgement"]
    assert ack_events and ack_events[0]["text"] == CallSession.ACKNOWLEDGEMENT
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_clear_audio_not_called_when_ack_already_finished_naturally():
    """`clear_audio` must only fire when the ACK was actually still playing
    and got cut off -- not on every turn. Calling it on a LiveKit transport
    when there's nothing queued risks clearing the AudioSource right before
    the real answer's own frames are pushed into it."""
    class SlowAnswerOrchestrator:
        def __init__(self):
            self.prepare_calls = []
        def prepare(self, question, history):
            self.prepare_calls.append(question)
            return FakePrepared(question, self)
        async def answer(self, question, history, metrics, state, prepared=None, candidate_audio=None):
            await asyncio.sleep(0.05)  # long enough for the (tiny) ACK to finish first
            return "the answer", []
    orchestrator = SlowAnswerOrchestrator()
    orchestrator.closed = []
    transport = SlowAudioTransport()
    session = CallSession(transport, PartialAwareASR(), orchestrator, FakeTTS())
    await session.start()
    await asyncio.sleep(0.02)  # let the (instant) ACK render + play out fully
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    await session.feed_audio(_tone(700, 0))
    await session._final(text)
    # `interrupt()` (called once per turn regardless of ACK state, to bump
    # generation) already issues its own `clear_audio` -- that one is a
    # baseline, unrelated to the ACK. What's under test is that `cut_ack()`
    # does not issue a *second* one when the ACK had already finished.
    baseline = len(transport.clear_audio_calls)
    await _wait_for_audio_end(transport)
    assert len(transport.clear_audio_calls) == baseline, "the ACK had already finished -- cut_ack() must not issue its own clear_audio on top of interrupt()'s"
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_ack_tts_uses_dedicated_connection_when_given():
    orchestrator = PrepareCountingOrchestrator()
    transport = Transport()
    tts, ack_tts = CountingTTS(), CountingTTS()
    session = CallSession(transport, PartialAwareASR(), orchestrator, tts, ack_tts=ack_tts)
    await session.start()
    await asyncio.sleep(0.01)
    assert ack_tts.calls.count(CallSession.ACKNOWLEDGEMENT) == 1
    assert CallSession.ACKNOWLEDGEMENT not in tts.calls, "the real-answer connection must never be asked to synthesize the ACK"
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    await session.feed_audio(_tone(700, 0))
    await session._final(text)
    await _wait_for_audio_end(transport)
    assert "the answer" in tts.calls
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_ack_starts_at_vad_end_before_asr_final_and_is_played_once():
    transport = Transport()
    session = CallSession(transport, PartialAwareASR(), PrepareCountingOrchestrator(), CountingTTS())
    await session.start()
    await asyncio.sleep(0.01)  # ACK render
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    await session.feed_audio(_tone(700, 0))
    await asyncio.sleep(0)
    types = [e["type"] for e in transport.events]
    assert "acknowledgement" in types and "final" not in types, "ACK must not wait for ASR finalization"
    generation = session.generation
    await session._final(text)
    assert session.generation == generation, "nothing was playing: the early ACK must not be interrupted"
    await _wait_for_audio_end(transport)
    assert [e["type"] for e in transport.events].count("acknowledgement") == 1
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("final", ["", "стоп"])
async def test_early_ack_is_cut_when_final_is_not_a_question(final):
    class LongAckTTS:
        async def synthesize_stream(self, text):
            yield b"\x00\x00" * 16000
    transport = SlowAudioTransport()
    session = CallSession(transport, PartialAwareASR(), PrepareCountingOrchestrator(), CountingTTS(), ack_tts=LongAckTTS())
    await session.start()
    await asyncio.sleep(0.02)
    await session.feed_audio(_tone(200, 1000))
    await session._partial("Как мне оформить Kaspi Gold")
    await session.feed_audio(_tone(700, 0))
    ack = session._early_ack
    assert ack is not None
    await session._final(final)
    assert ack.done() and session._early_ack is None
    assert transport.clear_audio_calls, "queued ACK audio must be flushed"
    await session.close()


@pytest.mark.asyncio
async def test_answer_gets_priority_over_still_playing_ack():
    from time import perf_counter
    class AckTTS:
        async def synthesize_stream(self, text):
            yield b"\x00\x00" * 16000   # 32000 bytes -> 10 chunks of 3200B each when played
    class ContentTTS:
        async def synthesize_stream(self, text):
            yield b"\x01\x01"
    orchestrator = PrepareCountingOrchestrator()
    transport = SlowAudioTransport()
    session = CallSession(transport, PartialAwareASR(), orchestrator, ContentTTS(), ack_tts=AckTTS())
    await session.start()
    await asyncio.sleep(0.02)  # let the (fast) ACK render finish
    text = "Как мне оформить Kaspi Gold"
    await session.feed_audio(_tone(200, 1000))
    await session._partial(text)
    await session.feed_audio(_tone(700, 0))
    started = perf_counter()
    # The ACK already started at VAD end above, and with nothing else playing
    # `_final` keeps its generation instead of calling interrupt().
    assert session._early_ack is not None
    baseline = len(transport.clear_audio_calls)
    await session._final(text)
    answer_pcm = b"\x01\x01"
    for _ in range(200):
        if any(pcm == answer_pcm for pcm, _ in transport.pcm):
            break
        await asyncio.sleep(0.005)
    elapsed = perf_counter() - started
    assert any(pcm == answer_pcm for pcm, _ in transport.pcm), "the real answer must actually reach the transport"
    # Playing all 10 ACK chunks at 0.02s each would take >=0.2s; the answer is
    # ready essentially immediately (orchestrator.answer resolves with no
    # delay), so it must not be made to wait behind the ACK finishing.
    assert elapsed < 0.15, f"answer audio took {elapsed:.3f}s -- looks like it waited on the ACK"
    # cut_ack() must flush since the ACK was genuinely still mid-flight when
    # it was cancelled here.
    assert len(transport.clear_audio_calls) > baseline, "the ACK's cancellation must flush anything still queued, not just stop the producer"
    await _wait_for_audio_end(transport)
    session.acknowledge_playback(session.generation)
    await session.task
    await session.close()


@pytest.mark.asyncio
async def test_empty_insufficient_context_is_safe_fallback_not_invalid_json():
    class Chat:
        async def json_completion(self,*args,**kwargs):return {'answer':'','cited_articles':[],'claims':[],'insufficient_context':True},None
    answer,_=await AnswerGenerator(Chat()).generate('unknown',[],[])
    assert answer['answer']==FALLBACK and answer['insufficient_context']
