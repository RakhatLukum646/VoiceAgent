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
async def test_empty_insufficient_context_is_safe_fallback_not_invalid_json():
    class Chat:
        async def json_completion(self,*args,**kwargs):return {'answer':'','cited_articles':[],'claims':[],'insufficient_context':True},None
    answer,_=await AnswerGenerator(Chat()).generate('unknown',[],[])
    assert answer['answer']==FALLBACK and answer['insufficient_context']
