import asyncio
import json
import httpx
import pytest
from app.rag.embeddings import GoogleEmbeddings
from app.agents.generator import GoogleChat
from app.agents.prompts import CHECKER, GENERATOR
from app.asr.google_live import GoogleLiveASR
from app.config import Settings
from app.providers import create_embeddings, create_chat, create_asr


@pytest.mark.asyncio
async def test_google_embedding_batch_contract():
    def respond(request):
        assert request.url.path.endswith("/models/gemini-embedding-001:batchEmbedContents")
        assert request.headers["x-goog-api-key"] == "secret"
        body = json.loads(request.content)
        assert [r["taskType"] for r in body["requests"]] == ["RETRIEVAL_DOCUMENT"] * 2
        return httpx.Response(200, json={"embeddings": [{"values": [1.0, 0.2]}, {"values": [0.3, 1.0]}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = GoogleEmbeddings("secret", "https://generativelanguage.googleapis.com/v1beta", "gemini-embedding-001", client)
        assert len(await provider.embed(["one", "two"], task_type="RETRIEVAL_DOCUMENT")) == 2


@pytest.mark.asyncio
async def test_google_json_stream_and_checker_unary():
    def respond(request):
        assert request.headers["x-goog-api-key"] == "secret"
        body = json.loads(request.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        if "streamGenerateContent" in request.url.path:
            return httpx.Response(200, text='data: {"candidates":[{"content":{"parts":[{"text":"{\\"answer\\":"}]}}]}\n\ndata: {"candidates":[{"content":{"parts":[{"text":"\\"yes\\"}"}]}}]}\n\n')
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{\"action\":\"PASS\"}"}]}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        chat = GoogleChat("secret", "https://generativelanguage.googleapis.com/v1beta", "gemini-3.5-flash", client)
        candidate, ttft = await chat.json_completion("system", {"q": "?"}, stream=True)
        assert candidate == {"answer": "yes"} and ttft is not None
        verdict, _ = await chat.json_completion("checker", {"a": "yes"})
        assert verdict["action"] == "PASS"


class FakeLiveSocket:
    def __init__(self): self.sent=[]; self.messages=asyncio.Queue()
    async def send(self, value): self.sent.append(json.loads(value))
    async def recv(self): return json.dumps({"setupComplete": {}})
    async def close(self): pass
    def __aiter__(self): return self
    async def __anext__(self):
        return await self.messages.get()


@pytest.mark.asyncio
async def test_live_asr_setup_pcm_partial_final(monkeypatch):
    socket = FakeLiveSocket()
    async def connect(url, **kwargs):
        assert url.startswith("wss://generativelanguage.googleapis.com/")
        assert "key=secret" in url
        return socket
    monkeypatch.setattr("app.asr.google_live.websockets.connect", connect)
    partials, finals = [], []
    async def partial(value): partials.append(value)
    async def final(value): finals.append(value)
    asr = GoogleLiveASR("secret", language="ru")
    await asr.start(partial, final)
    assert socket.sent[0]["setup"]["inputAudioTranscription"]["languageCodes"] == ["ru-RU"]
    await asr.begin_utterance()
    await asr.send_audio(b"\0" * 3200)
    assert socket.sent[-1]["realtimeInput"]["audio"]["mimeType"] == "audio/pcm;rate=16000"
    await socket.messages.put(json.dumps({"serverContent": {"interimInputTranscription": {"text": "прив"}, "inputTranscription": {"text": "привет"}}}))
    await asyncio.sleep(0.01)
    await asr.finalize()
    await asyncio.sleep(0.35)
    assert partials == ["прив"] and finals == ["привет"]
    assert socket.sent[-1] == {"realtimeInput": {"audioStreamEnd": True}}
    await asr.close()


@pytest.mark.asyncio
async def test_live_asr_waits_for_trailing_final_before_next_turn(monkeypatch):
    socket = FakeLiveSocket()
    async def connect(url, **kwargs): return socket
    monkeypatch.setattr("app.asr.google_live.websockets.connect", connect)
    finals = []
    async def callback(value): finals.append(value)
    asr = GoogleLiveASR("secret")
    await asr.start(callback, callback)
    await asr.begin_utterance()
    await asr.send_audio(b"\0" * 3200)
    await asr.finalize()
    next_turn = asyncio.create_task(asr.begin_utterance())
    await asyncio.sleep(0.05)
    assert not next_turn.done()
    await socket.messages.put(json.dumps({"serverContent": {"inputTranscription": {"text": "первый"}}}))
    await asyncio.sleep(0.05)
    await socket.messages.put(json.dumps({"serverContent": {"inputTranscription": {"text": "второй"}}}))
    await asyncio.wait_for(next_turn, timeout=1)
    assert finals == ["первый второй"]
    assert asr.listening
    await asr.close()


@pytest.mark.asyncio
async def test_live_asr_reuses_stream_for_two_utterances_without_duplicate_begin(monkeypatch):
    socket = FakeLiveSocket()
    async def connect(url, **kwargs): return socket
    monkeypatch.setattr("app.asr.google_live.websockets.connect", connect)
    finals = []
    async def on_final(value): finals.append(value)
    asr = GoogleLiveASR("secret")
    await asr.start(lambda value: asyncio.sleep(0), on_final)
    for n in (1, 2):
        await asr.begin_utterance()
        sequence = asr._turn_seq
        await asr.begin_utterance()
        assert asr._turn_seq == sequence
        await asr.send_audio(b"\0" * 3200)
        await asr.finalize()
        await socket.messages.put(json.dumps({"serverContent": {"inputTranscription": {"text": f"вопрос {n}"}}}))
        await asyncio.wait_for(asr._turn_complete.wait(), 1)
    assert finals == ["вопрос 1", "вопрос 2"]
    assert len([message for message in socket.sent if message.get("realtimeInput", {}).get("audioStreamEnd")]) == 2
    await asr.close()


@pytest.mark.asyncio
async def test_live_asr_stale_final_timer_cannot_complete_new_turn(monkeypatch):
    socket = FakeLiveSocket()
    async def connect(url, **kwargs): return socket
    monkeypatch.setattr("app.asr.google_live.websockets.connect", connect)
    finals = []
    async def on_final(value): finals.append(value)
    asr = GoogleLiveASR("secret")
    await asr.start(lambda value: asyncio.sleep(0), on_final)
    await asr.begin_utterance()
    old_sequence = asr._turn_seq
    await asr.finalize()
    asr._turn_complete.set()
    await asr.begin_utterance()
    await asr._emit_final(old_sequence)
    assert not asr._turn_complete.is_set() and not finals
    await asr.close()


@pytest.mark.asyncio
async def test_old_final_callback_cannot_complete_new_turn(monkeypatch):
    socket = FakeLiveSocket()
    async def connect(url, **kwargs): return socket
    monkeypatch.setattr("app.asr.google_live.websockets.connect", connect)
    entered, release = asyncio.Event(), asyncio.Event()
    async def on_final(value):
        entered.set()
        await release.wait()
    asr = GoogleLiveASR("secret")
    await asr.start(lambda value: asyncio.sleep(0), on_final)
    await asr.begin_utterance()
    await asr.send_audio(b"\0" * 3200)
    await asr.finalize()
    await socket.messages.put(json.dumps({"serverContent": {"inputTranscription": {"text": "первый"}}}))
    await asyncio.wait_for(entered.wait(), 1)
    await asr.begin_utterance()
    assert not asr._turn_complete.is_set()
    release.set()
    await asyncio.sleep(0.02)
    assert not asr._turn_complete.is_set()
    await asr.close()


@pytest.mark.asyncio
async def test_live_asr_reconnects_after_missing_final(monkeypatch):
    sockets = [FakeLiveSocket(), FakeLiveSocket()]
    async def connect(url, **kwargs): return sockets.pop(0)
    monkeypatch.setattr("app.asr.google_live.websockets.connect", connect)
    finals = []
    async def on_final(value): finals.append(value)
    asr = GoogleLiveASR("secret")
    await asr.start(lambda value: asyncio.sleep(0), on_final)
    first_socket = asr.ws
    await asr.begin_utterance()
    await asr.finalize()
    await asyncio.wait_for(asr._turn_complete.wait(), 6)
    assert asr._needs_reconnect and finals == [""]
    await asr.begin_utterance()
    assert asr.ws is not first_socket and asr.listening
    await asr.close()


def test_provider_aware_readiness_and_index_identity(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "google")
    monkeypatch.setenv("ASR_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    config = Settings()
    assert config.embedding_model == "google:gemini-embedding-001"
    assert isinstance(create_embeddings(config), GoogleEmbeddings)
    assert isinstance(create_chat(config), GoogleChat)
    assert isinstance(create_asr(config), GoogleLiveASR)
    assert "GOOGLE_API_KEY" not in config.missing_credentials()


@pytest.mark.asyncio
async def test_google_checker_requests_required_json_schema():
    def respond(request):
        schema = json.loads(request.content)["generationConfig"]["responseJsonSchema"]
        assert set(schema["required"]) == set(schema["properties"])
        assert schema["properties"]["complete"]["type"] == "boolean"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        await GoogleChat("secret", "https://example.test", "model", client).json_completion(CHECKER, {})


@pytest.mark.asyncio
async def test_google_minimal_thinking_only_for_generator():
    seen = []
    def respond(request):
        seen.append(json.loads(request.content)["generationConfig"])
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        chat = GoogleChat("secret", "https://example.test", "gemini-3.5-flash", client)
        await chat.json_completion(GENERATOR, {})
        await chat.json_completion(CHECKER, {})
    assert seen[0]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert "thinkingConfig" not in seen[1]
