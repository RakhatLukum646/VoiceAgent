"""Small live Google provider probes; prints status/shape only, never credentials."""
import argparse
import asyncio
import audioop
import json
import wave
from pathlib import Path
from time import perf_counter
import httpx
from app.config import Settings
from app.rag.embeddings import GoogleEmbeddings
from app.agents.generator import GoogleChat
from app.asr.google_live import GoogleLiveASR


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-asr", action="store_true", help="Probe Gemini Live transcription WebSocket setup")
    parser.add_argument("--audio", type=Path, help="Optional mono PCM16 WAV to stream through Live ASR in real time")
    parser.add_argument("--repeat-audio", action="store_true", help="Stream the bundled speech fixture twice on one Live ASR session")
    args = parser.parse_args()
    if args.repeat_audio and not args.audio:
        args.audio = Path("artifacts/asr-smoke-source.wav")
    config = Settings()
    if not config.google_api_key:
        print("GOOGLE_API_KEY is missing")
        return 2
    status = {}
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(f"{config.google_base_url.rstrip('/')}/models", headers={"x-goog-api-key": config.google_api_key})
        status["models_http"] = response.status_code
        if response.status_code == 200:
            names = {m.get("name") for m in response.json().get("models", [])}
            status["generation_model_listed"] = f"models/{config.google_llm_model}" in names
            status["embedding_model_listed"] = f"models/{config.google_embedding_model}" in names
        embeddings = GoogleEmbeddings(config.google_api_key, config.google_base_url, config.google_embedding_model, client)
        try:
            vector = (await embeddings.embed(["Constitution of Kazakhstan"]))[0]
            status["embedding_dimensions"] = len(vector)
        except Exception as exc:
            status["embedding_error_type"] = type(exc).__name__
        chat = GoogleChat(config.google_api_key, config.google_base_url, config.google_llm_model, client)
        try:
            value, _ = await chat.json_completion("Return JSON only", {"question": "Return {\"ok\": true}"})
            status["generation_json_received"] = isinstance(value, dict)
        except Exception as exc:
            status["generation_error_type"] = type(exc).__name__
    if args.live_asr or args.audio:
        asr = GoogleLiveASR(config.google_api_key, config.google_asr_model, config.asr_language)
        try:
            result = {"first_partial_ms": None, "final_ms": None, "final_text": None}
            began = perf_counter()
            completed = asyncio.Event()
            async def partial(value):
                if result["first_partial_ms"] is None:
                    result["first_partial_ms"] = round((perf_counter() - began) * 1000, 2)
            async def final(value):
                result["final_ms"] = round((perf_counter() - began) * 1000, 2)
                result["final_text"] = value
                completed.set()
            await asr.start(partial, final)
            status["live_asr_setup"] = True
            if args.audio:
                with wave.open(str(args.audio), "rb") as wav:
                    if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                        raise ValueError("Audio must be mono PCM16 WAV")
                    rate = wav.getframerate()
                    pcm = wav.readframes(wav.getnframes())
                if rate != 16000:
                    pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 16000, None)
                probes = []
                for _ in range(2 if args.repeat_audio else 1):
                    result = {"first_partial_ms": None, "final_ms": None, "final_text": None}
                    completed.clear()
                    await asr.begin_utterance()
                    began = perf_counter()
                    for i in range(0, len(pcm), 3200):
                        await asr.send_audio(pcm[i:i + 3200])
                        await asyncio.sleep(0.1)
                    await asr.finalize()
                    try:
                        await asyncio.wait_for(completed.wait(), timeout=10)
                    except TimeoutError:
                        result["error"] = "No final transcript within 10 seconds"
                    probes.append(result)
                status["audio_probe"] = probes if args.repeat_audio else probes[0]
        except Exception as exc:
            status["live_asr_setup"] = False
            status["live_asr_error_type"] = type(exc).__name__
        finally:
            await asr.close()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    probe = status.get("audio_probe", [])
    probes = probe if isinstance(probe, list) else [probe]
    return 0 if status.get("embedding_dimensions") and status.get("generation_json_received") and (not (args.live_asr or args.audio) or status.get("live_asr_setup")) and (not args.audio or len(probes) == (2 if args.repeat_audio else 1) and all(item.get("final_text") for item in probes)) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
