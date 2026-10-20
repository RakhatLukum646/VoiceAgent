import json
import httpx
import pytest
from scripts.check_deepdub import list_voices, probe_websocket


@pytest.mark.asyncio
async def test_voice_listing_selects_only_safe_fields():
    def respond(request):
        assert request.url.path == "/api/v1/voice"
        assert request.url.params["limit"] == "100"
        assert request.headers["x-api-key"] == "secret"
        return httpx.Response(200, json={"voicePrompts": [{"id": "vp1", "name": "Narrator", "locale": "en-US", "text": "private sample", "speakerId": "secret"}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        assert await list_voices("secret", client) == [{"id": "vp1", "name": "Narrator", "locale": "en-US"}]


@pytest.mark.asyncio
async def test_empty_voice_listing_is_valid():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"voicePrompts": []}))) as client:
        assert await list_voices("secret", client) == []


@pytest.mark.asyncio
async def test_ws_probe_authenticated(monkeypatch):
    class FakeWS:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def recv(self): return json.dumps({"action": "status", "message": "connected"})
    def connect(url, **kwargs):
        assert kwargs["additional_headers"] == {"x-api-key": "secret"}
        return FakeWS()
    monkeypatch.setattr("scripts.check_deepdub.websockets.connect", connect)
    assert await probe_websocket("secret", "wss://wss.deepdub.ai/ws")
