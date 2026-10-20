import asyncio
import base64
import json
import pytest
from app.asr.deepgram import DeepgramASR
from app.tts.deepdub import DeepdubTTSProvider


class FakeSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self._recv_queue = asyncio.Queue()
        for message in messages:
            self._recv_queue.put_nowait(message)
        self.sent = []
    async def recv(self): return await self._recv_queue.get()
    def push(self, message): self._recv_queue.put_nowait(message)
    async def send(self, value): self.sent.append(json.loads(value) if isinstance(value, str) and value.startswith("{") else value)
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    def __aiter__(self): return self
    async def __anext__(self):
        if not self.messages: raise StopAsyncIteration
        return self.messages.pop(0)


@pytest.mark.asyncio
async def test_deepgram_aggregates_segment_finals_and_finalize():
    asr = DeepgramASR("key")
    asr.ws = FakeSocket([json.dumps({"type":"Results","is_final":True,"channel":{"alternatives":[{"transcript":"первый"}]}}), json.dumps({"type":"Results","is_final":True,"from_finalize":True,"channel":{"alternatives":[{"transcript":"второй"}]}})])
    finals = []
    async def partial(value): pass
    async def final(value): finals.append(value)
    await asr._read(partial, final)
    assert finals == ["первый второй"]


@pytest.mark.asyncio
async def test_deepdub_stream_contract_and_error(monkeypatch):
    sock = FakeSocket([json.dumps({"action":"status","message":"connected"}), json.dumps({"data":base64.b64encode(b"\x01\x02").decode(),"isFinal":False}), json.dumps({"isFinal":True})])
    monkeypatch.setattr("app.tts.deepdub.websockets.connect", lambda *args, **kwargs: sock)
    provider = DeepdubTTSProvider("key", "voice", "wss://wss.deepdub.ai/ws")
    chunks = [chunk async for chunk in provider.synthesize_stream("hello")]
    assert chunks == [b"\x01\x02"]
    assert [m["action"] for m in sock.sent] == ["stream-config", "stream-text", "end-stream"]
    error_sock = FakeSocket([json.dumps({"action":"status","message":"connected"}), json.dumps({"action":"error","message":"bad voice"})])
    monkeypatch.setattr("app.tts.deepdub.websockets.connect", lambda *args, **kwargs: error_sock)
    other = DeepdubTTSProvider("key", "voice", "wss://wss.deepdub.ai/ws")
    with pytest.raises(RuntimeError, match="bad voice"):
        _ = [chunk async for chunk in other.synthesize_stream("hello")]


@pytest.mark.asyncio
async def test_deepdub_reuses_connection_across_turns(monkeypatch):
    sock = FakeSocket([
        json.dumps({"action": "status", "message": "connected"}),
        json.dumps({"data": base64.b64encode(b"a").decode(), "isFinal": True}),
        json.dumps({"data": base64.b64encode(b"b").decode(), "isFinal": True}),
    ])
    connects = []
    monkeypatch.setattr("app.tts.deepdub.websockets.connect", lambda *a, **k: (connects.append(1), sock)[1])
    provider = DeepdubTTSProvider("key", "voice", "wss://wss.deepdub.ai/ws")
    first = [chunk async for chunk in provider.synthesize_stream("one")]
    second = [chunk async for chunk in provider.synthesize_stream("two")]
    assert first == [b"a"] and second == [b"b"]
    assert len(connects) == 1
    assert [m["action"] for m in sock.sent] == ["stream-config", "stream-text", "end-stream", "stream-text", "end-stream"]
    await provider.close()


@pytest.mark.asyncio
async def test_deepdub_cancel_forces_reconnect_for_next_turn(monkeypatch):
    # A clean isCancelled drain does not prove the server's turn state is safe
    # to reuse; the provider must reconnect (fresh stream-config) rather than
    # trust the post-cancel socket for the next turn.
    sock1 = FakeSocket([json.dumps({"action": "status", "message": "connected"})])
    sock2 = FakeSocket([json.dumps({"action": "status", "message": "connected"})])
    sockets = [sock1, sock2]
    monkeypatch.setattr("app.tts.deepdub.websockets.connect", lambda *a, **k: sockets.pop(0))
    provider = DeepdubTTSProvider("key", "voice", "wss://wss.deepdub.ai/ws")

    sock1.push(json.dumps({"data": base64.b64encode(b"x").decode(), "isFinal": False}))
    agen = provider.synthesize_stream("long answer")
    assert await agen.__anext__() == b"x"
    task = asyncio.ensure_future(agen.__anext__())
    await asyncio.sleep(0)
    task.cancel()
    sock1.push(json.dumps({"isCancelled": True}))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert {"action": "cancel"} in sock1.sent
    assert provider._ws is None

    sock2.push(json.dumps({"data": base64.b64encode(b"y").decode(), "isFinal": True}))
    resumed = [chunk async for chunk in provider.synthesize_stream("next turn")]
    assert resumed == [b"y"]
    assert [m["action"] for m in sock2.sent] == ["stream-config", "stream-text", "end-stream"]


@pytest.mark.asyncio
async def test_deepgram_idle_keepalive():
    asr = DeepgramASR("key")
    sock = FakeSocket([])
    asr.ws = sock
    asr.last_audio -= 20
    task = asyncio.create_task(asr._keepalive())
    await asyncio.sleep(3.1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert {"type": "KeepAlive"} in sock.sent
