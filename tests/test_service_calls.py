import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest
from app.agents.scope import ScopedOrchestrator, Route, REDIRECT, TRANSFER
from app.agents.prompts import FALLBACK
from app.call.session import CallSession
from app.call.livekit_transport import LiveKitTransport, LiveKitCall, room_token
from scripts.ingest_service import extract
import jwt


class ASR:
    async def start(self, partial, final): self.partial, self.final = partial, final
    async def close(self): self.closed = True


class Transport:
    def __init__(self): self.events, self.order = [], []
    async def event(self, item): self.events.append(item); self.order.append(item['type'])
    async def audio(self, pcm, generation): self.order.append('audio')
    async def drain(self, generation): self.order.append('drain')
    async def hangup(self, reason): self.order.append('hangup'); self.reason = reason


async def wait_event(transport, kind):
    async def poll():
        while not any(item['type'] == kind for item in transport.events):
            await asyncio.sleep(.005)
    await asyncio.wait_for(poll(), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize('action', ['TRANSFER', 'REDIRECT', 'RAG'])
async def test_scenarios_only_transfer_hangs_up_after_drain_and_playback(action):
    class Router:
        async def route(self, q, history): return Route(action, TRANSFER if action == 'TRANSFER' else REDIRECT)
        async def answer(self, *args): return FALLBACK, []
    class TTS:
        async def synthesize_stream(self, text): yield b'\0\0'
    transport = Transport()
    session = CallSession(transport, ASR(), Router(), TTS())
    await session.start()
    await session.asr.final('question')
    await wait_event(transport, 'audio_end')
    assert 'hangup' not in transport.order
    assert transport.order.index('drain') < transport.order.index('audio_end')
    if action == 'TRANSFER':
        old, generation = session.task, session.generation
        await session.asr.partial('новый вопрос')
        await session.asr.final('не хочу перевод, хочу Gold')
        await session.interrupt()
        assert session.task is old and session.turn_id == 1 and session.generation == generation
    session.acknowledge_playback(session.generation)
    await session.task
    assert ('hangup' in transport.order) == (action == 'TRANSFER')
    if action == 'TRANSFER':
        assert transport.order[-1] == 'hangup'
        assert transport.reason == 'transfer'
    else:
        assert session.state == 'listening' and not session.terminal
    await session.close()


@pytest.mark.asyncio
async def test_transfer_tts_failure_is_visible_and_does_not_claim_complete_playback():
    class Router:
        async def route(self, *args): return Route('TRANSFER', TRANSFER)
    class TTS:
        async def synthesize_stream(self, text):
            raise RuntimeError('provider secret must not reach caller')
            yield b''
    transport = Transport()
    session = CallSession(transport, ASR(), Router(), TTS())
    await session.start(); await session.asr.final('credit')
    await session.task
    assert transport.reason == 'transfer_audio_failed'
    assert 'drain' not in transport.order
    assert any(item['type'] == 'error' for item in transport.events)
    assert 'provider secret' not in str(transport.events)
    await session.close()


@pytest.mark.asyncio
async def test_manual_close_cancels_terminal_speech():
    class Router:
        async def route(self, *args): return Route('TRANSFER', TRANSFER)
    begun = asyncio.Event()
    class TTS:
        async def synthesize_stream(self, text):
            begun.set(); await asyncio.sleep(30); yield b''
    session = CallSession(Transport(), ASR(), Router(), TTS())
    await session.start(); await session.asr.final('credit'); await begun.wait()
    await asyncio.wait_for(session.close(), 1)
    assert session.closed and session.task.cancelled() and session.asr.closed


@pytest.mark.asyncio
@pytest.mark.parametrize('result', [{}, {'action':'DELETE'}, None, {'action':1}])
async def test_invalid_classifier_never_transfers(result):
    class Chat:
        async def json_completion(self, *args): return result, None
    assert (await ScopedOrchestrator(None, Chat()).route('question', [])).action == 'REDIRECT'


def test_snapshot_contains_only_answer_body():
    title, body, updated = extract(Path('data/kaspi_gold_source.html').read_text(encoding='utf-8'))
    assert 'Kartomat' in body and 'мобильный телефон' in body
    assert 'Кредит Наличными' not in body and '9999' not in body
    assert '31.01.2021' in updated and title == 'Где я могу получить Kaspi Gold?'


def test_browser_token_scoped_short_lived_without_admin_grants():
    config = SimpleNamespace(livekit_api_key='key', livekit_api_secret='x'*40)
    token = room_token(config, 'one-room', 'one-caller', browser=True)
    payload = jwt.decode(token, config.livekit_api_secret, algorithms=['HS256'])
    assert payload['video']['room'] == 'one-room'
    assert payload['video']['canPublishSources'] == ['microphone']
    assert not payload['video'].get('roomAdmin')
    assert payload['exp'] - payload['nbf'] <= 300


@pytest.mark.asyncio
async def test_cancelled_capture_cannot_refill_cleared_queue_and_old_generation_is_dropped():
    capturing = asyncio.Event()
    class Source:
        def __init__(self): self.queue=[]
        async def capture_frame(self, frame):
            capturing.set(); await asyncio.sleep(30); self.queue.append(frame)
        def clear_queue(self): self.queue.clear()
    class Participant:
        async def publish_data(self, *args, **kwargs): pass
    source = Source()
    transport = LiveKitTransport(SimpleNamespace(local_participant=Participant()), source, None, 'room', 'caller', asyncio.Event())
    producer = asyncio.create_task(transport.audio(b'\0'*640, 0))
    await capturing.wait()
    await transport.event({'type':'interrupt', 'generation':1})
    producer.cancel(); await asyncio.gather(producer, return_exceptions=True)
    await transport.clear_audio(1)
    await transport.audio(b'\0'*640, 0)
    assert not source.queue


@pytest.mark.asyncio
async def test_finish_before_caller_joins_does_not_wait_connect_timeout():
    call = object.__new__(LiveKitCall)
    call.joined, call.finished = asyncio.Event(), asyncio.Event()
    call.finished.set()
    call.config = SimpleNamespace(call_connect_seconds=45, max_call_seconds=600)
    called = []
    async def close(): called.append(True)
    call.close = close
    await asyncio.wait_for(call.watch(), .2)
    assert called == [True]


@pytest.mark.asyncio
async def test_single_page_verifier_loads_complete_corpus_without_extra_embedding():
    from app.rag.retriever import Retriever
    class Store:
        def count_for_model(self, model): return 1
        def all_passages(self, model): return [{'article':'q27', 'text':'Full official body'}]
    class Embeddings:
        async def embed(self, texts): raise AssertionError('Redundant verifier embedding')
    hits = await Retriever(Embeddings(), Store(), 'model').verifier_search('question')
    assert hits[0]['text'] == 'Full official body'


@pytest.mark.asyncio
async def test_rtc_packetizer_retains_partial_frames_and_drains_aligned_pcm():
    frames=[]
    class Source:
        async def capture_frame(self, frame):frames.append(bytes(frame.data))
        async def wait_for_playout(self):pass
        def clear_queue(self):pass
    class Participant:
        async def publish_data(self,*args,**kwargs):pass
    transport=LiveKitTransport(SimpleNamespace(local_participant=Participant()),Source(),None,'room','caller',asyncio.Event())
    await transport.audio(b'a'*101,0)
    assert frames==[]
    await transport.audio(b'b'*541,0)
    assert len(frames)==1 and len(frames[0])==640
    await transport.drain(0)
    assert all(len(frame)==640 for frame in frames)
