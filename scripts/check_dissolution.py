"""Live dissolution answer plus filler interruption probe; no microphone required."""

import asyncio
import json
from pathlib import Path

from app.agents.checker import AnswerChecker
from app.agents.generator import AnswerGenerator
from app.agents.orchestrator import Orchestrator
from app.agents.prompts import FALLBACK
from app.call.session import CallSession
from app.config import Settings
from app.providers import create_chat, create_embeddings
from app.rag.retriever import Retriever
from app.rag.vector_store import SQLiteVectorStore
from app.tts.deepdub import DeepdubTTSProvider


QUESTION = "Может ли президент распустить парламент?"
FILLER = "Вот вот этот вот, что"


class FakeASR:
    async def start(self, partial, final):
        self.partial, self.final = partial, final

    async def close(self):
        pass


class Transport:
    def __init__(self):
        self.events = []
        self.ack = asyncio.Event()
        self.frames = 0
        self.session = None

    async def event(self, payload):
        self.events.append(payload)
        if payload["type"] == "acknowledgement":
            self.ack.set()
        if payload["type"] == "audio_end":
            self.session.acknowledge_playback(payload["generation"])

    async def audio(self, pcm, generation):
        self.frames += 1


class TracingOrchestrator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.trace = {}

    async def answer(self, question, history, metrics, state):
        return await self.pipeline.answer(question, history, metrics, state, self.trace)


async def run():
    config = Settings()
    if config.missing_credentials():
        raise RuntimeError("Configure providers before live dissolution probe")
    store = SQLiteVectorStore(config.db_path)
    if not store.count_for_model(config.embedding_model):
        raise RuntimeError("Configured corpus is missing")
    embeddings = create_embeddings(config)
    generator_chat, checker_chat = create_chat(config), create_chat(config)
    report = {}
    session = None
    try:
        retriever = Retriever(embeddings, store, config.embedding_model, config.top_k)
        pipeline = TracingOrchestrator(Orchestrator(retriever, AnswerGenerator(generator_chat), AnswerChecker(checker_chat, retriever), config.min_similarity))
        transport = Transport()
        session = CallSession(transport, FakeASR(), pipeline, DeepdubTTSProvider(config.deepdub_api_key, config.deepdub_voice_id, config.deepdub_base_url, config.deepdub_locale))
        transport.session = session
        await session.start()
        await session.asr.final(QUESTION)
        pending = session.task
        await asyncio.wait_for(transport.ack.wait(), 15)
        generation = session.generation
        session._barge_pending = True  # Simulate a VAD start before ASR partials.
        await session.asr.partial("Вот вот этот вот, ч")
        await session.asr.final(FILLER)
        filler_preserved = session.task is pending and session.generation == generation and session.turn_id == 1
        await asyncio.wait_for(pending, 90)
        answers = [item for item in transport.events if item["type"] == "answer"]
        acks = [item for item in transport.events if item["type"] == "acknowledgement"]
        finals = [item for item in transport.events if item["type"] == "final"]
        metrics = [item for item in transport.events if item["type"] == "metrics"]
        report = {"filler_preserved_task": filler_preserved, "acknowledgements": len(acks), "ack_confirmed_question": bool(acks and QUESTION in acks[0]["text"]), "finals": len(finals), "answers": len(answers), "abstained": bool(answers and answers[0]["text"] == FALLBACK), "source_articles": [source["article"] for source in answers[0]["sources"]] if answers else [], "checker_actions": [attempt.get("checker", {}).get("action") for attempt in pipeline.trace.get("attempts", [])], "committed_question": bool(session.history and session.history[0]["content"] == QUESTION), "frames": transport.frames, "metrics": len(metrics), "errors": len([item for item in transport.events if item["type"] == "error"]), "trace": pipeline.trace}
    except Exception as exc:
        report["failure_type"] = type(exc).__name__
    finally:
        if session:
            await session.close()
        await embeddings.client.aclose()
        await generator_chat.client.aclose()
        await checker_chat.client.aclose()
        path = Path("artifacts/dissolution-smoke.json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("failure_type") or not report.get("filler_preserved_task") or report.get("acknowledgements") != 1 or not report.get("ack_confirmed_question") or report.get("finals") != 1 or report.get("answers") != 1 or report.get("abstained") or "62" not in report.get("source_articles", []) or report.get("checker_actions", [])[-1:] != ["PASS"] or not report.get("committed_question") or not report.get("frames") or report.get("metrics") != 1 or report.get("errors"):
        raise RuntimeError("Dissolution smoke failed; inspect artifacts/dissolution-smoke.json")


if __name__ == "__main__":
    asyncio.run(run())
