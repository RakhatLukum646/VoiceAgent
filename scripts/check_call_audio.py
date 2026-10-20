"""Exercise two complete microphone-style turns over the real call WebSocket."""

import argparse
import asyncio
import audioop
import json
import wave
from array import array
from pathlib import Path

import websockets
from app.agents.prompts import FALLBACK


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("Fixture must be mono PCM16 WAV")
        pcm = source.readframes(source.getnframes())
        rate = source.getframerate()
    if rate != 16000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 16000, None)
    return pcm


async def run(url: str, audio: Path, turns: int, report_path: Path, noise_bursts: bool = False):
    pcm = read_pcm(audio)
    noise_frame = array("h", [700] * 1600).tobytes()
    report = {"turns_requested": turns, "finals": 0, "answers": 0, "abstentions": 0, "acknowledgements": 0, "audio_frames": 0, "audio_ends": 0, "playback_acks": 0, "completed_turns": 0, "history_turns": 0, "noise_frames_sent": 0, "error_types": []}
    completed = asyncio.Event()
    reader_task = None
    finished = False
    try:
        async with websockets.connect(url, max_size=4 * 1024 * 1024, open_timeout=10) as ws:
            async def read_events():
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            report["audio_frames"] += 1
                            continue
                        item = json.loads(raw)
                        kind = item.get("type")
                        if kind == "final":
                            report["finals"] += 1
                        elif kind == "answer":
                            report["answers"] += 1
                            if item.get("text") == FALLBACK:
                                report["abstentions"] += 1
                        elif kind == "acknowledgement":
                            report["acknowledgements"] += 1
                        elif kind == "audio_end":
                            report["audio_ends"] += 1
                            await ws.send(json.dumps({"type": "playback_done", "generation": item["generation"]}))
                            report["playback_acks"] += 1
                        elif kind == "metrics":
                            report["completed_turns"] += 1
                            report["history_turns"] = item.get("history_turns", 0)
                            report.setdefault("durations_ms", []).append(item.get("metrics_ms", {}))
                            completed.set()
                        elif kind == "error":
                            report["error_types"].append("call_error")
                            completed.set()
                finally:
                    if not finished:
                        report["error_types"].append("websocket_closed")
                        completed.set()
            reader_task = asyncio.create_task(read_events())
            for number in range(1, turns + 1):
                completed.clear()
                for offset in range(0, len(pcm), 3200):
                    await ws.send(pcm[offset:offset + 3200])
                    await asyncio.sleep(0.1)
                for _ in range(8):
                    await ws.send(b"\0" * 3200)
                    await asyncio.sleep(0.1)
                async def keep_mic_open():
                    tick = 0
                    while not completed.is_set():
                        noisy = noise_bursts and tick % 20 in (0, 1)
                        await ws.send(noise_frame if noisy else b"\0" * 3200)
                        if noisy:
                            report["noise_frames_sent"] += 1
                        tick += 1
                        await asyncio.sleep(0.1)
                silence_task = asyncio.create_task(keep_mic_open())
                try:
                    await asyncio.wait_for(completed.wait(), timeout=60)
                finally:
                    silence_task.cancel()
                    await asyncio.gather(silence_task, return_exceptions=True)
                if report["error_types"] or report["completed_turns"] != number:
                    raise RuntimeError(f"Call turn {number} failed")
            finished = True
            await ws.send(json.dumps({"type": "end"}))
    except Exception as exc:
        report["failure_type"] = type(exc).__name__
    finally:
        if reader_task and not reader_task.done():
            reader_task.cancel()
        if reader_task:
            await asyncio.gather(reader_task, return_exceptions=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("failure_type") or report["error_types"] or report["finals"] != turns or report["answers"] != turns or report["abstentions"] or report["audio_ends"] != turns or report["playback_acks"] != turns or report["completed_turns"] != turns or report["history_turns"] != turns or report["audio_frames"] == 0:
        raise RuntimeError("Call audio smoke failed; inspect report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8010/ws/call")
    parser.add_argument("--audio", type=Path, default=Path("artifacts/asr-smoke-source.wav"))
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--report", type=Path, default=Path("artifacts/call-audio-smoke.json"))
    parser.add_argument("--noise-bursts", action="store_true", help="Inject short and sustained above-threshold noise while answers are pending")
    options = parser.parse_args()
    asyncio.run(run(options.url, options.audio, options.turns, options.report, options.noise_bursts))
