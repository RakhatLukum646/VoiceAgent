"""Safe, TTS-only Deepdub account diagnostic. Never prints API credentials."""
import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
import wave

import httpx
import websockets

from app.config import Settings
from app.tts.deepdub import DeepdubTTSProvider


VOICES_URL = "https://restapi.deepdub.ai/api/v1/voice"
DEFAULT_WS_URL = "wss://wss.deepdub.ai/ws"


async def list_voices(api_key: str, client: httpx.AsyncClient | None = None) -> list[dict]:
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=15)
    try:
        response = await client.get(VOICES_URL, params={"limit": 100}, headers={"x-api-key": api_key})
        if response.status_code != 200:
            raise RuntimeError(f"Voice list request failed (HTTP {response.status_code})")
        payload = response.json()
        voices = payload.get("voicePrompts")
        if not isinstance(voices, list):
            raise RuntimeError("Voice list response missing voicePrompts")
        return [{"id": voice.get("id"), "name": voice.get("name"), "locale": voice.get("locale")} for voice in voices if isinstance(voice, dict)]
    finally:
        if own_client:
            await client.aclose()


async def probe_websocket(api_key: str, base_url: str) -> bool:
    try:
        async with websockets.connect(base_url, additional_headers={"x-api-key": api_key}, open_timeout=10) as ws:
            response = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            return response.get("action") == "status" and response.get("message") == "connected"
    except (OSError, TimeoutError, websockets.exceptions.WebSocketException, ValueError):
        return False


async def synthesize_sample(config: Settings, text: str, output: Path, base_url: str) -> dict:
    if not config.deepdub_voice_id:
        raise RuntimeError("DEEPDUB_VOICE_ID is required for synthesis; choose an account voice matching the locale")
    provider = DeepdubTTSProvider(config.deepdub_api_key, config.deepdub_voice_id, base_url, config.deepdub_locale)
    started = perf_counter()
    first_audio_ms = None
    audio = bytearray()
    async for chunk in provider.synthesize_stream(text):
        if first_audio_ms is None:
            first_audio_ms = round((perf_counter() - started) * 1000, 2)
        audio.extend(chunk)
        if len(audio) > 16_000 * 2 * 60:
            raise RuntimeError("Diagnostic audio exceeded 60 seconds")
    if not audio or len(audio) % 2:
        raise RuntimeError("Deepdub returned no valid PCM16 audio")
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(audio)
    return {"first_audio_ms": first_audio_ms, "total_ms": round((perf_counter() - started) * 1000, 2), "audio_seconds": round(len(audio) / 32000, 2), "output": str(output)}


async def main() -> int:
    parser = argparse.ArgumentParser(description="Check Deepdub credentials and account voices; optional TTS-only WAV synthesis")
    parser.add_argument("--text", help="Optional text to synthesize (requires DEEPDUB_VOICE_ID)")
    parser.add_argument("--output", type=Path, default=Path("data/deepdub_test.wav"))
    args = parser.parse_args()
    config = Settings()
    if not config.deepdub_api_key:
        print("DEEPDUB_API_KEY is missing from the environment/.env")
        return 2
    ws_url = config.deepdub_base_url or DEFAULT_WS_URL
    try:
        voices = await list_voices(config.deepdub_api_key)
        print(json.dumps({"voice_count": len(voices), "voices": voices}, ensure_ascii=False, indent=2))
    except Exception as exc:
        # Restrict output to diagnostic type/status; never print response bodies or credentials.
        print(f"Voice list unavailable: {type(exc).__name__}")
        voices = []
    connected = await probe_websocket(config.deepdub_api_key, ws_url)
    print(json.dumps({"websocket_authenticated": connected}))
    if not connected:
        return 1
    if args.text:
        try:
            result = await synthesize_sample(config, args.text, args.output, ws_url)
            print(json.dumps({"tts_sample": result}, ensure_ascii=False))
        except Exception as exc:
            print(f"Synthesis failed: {type(exc).__name__}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
