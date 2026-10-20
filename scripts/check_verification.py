"""Optional live verification and voice smoke test; no credentials in output."""

import argparse
import asyncio
import json
from pathlib import Path

from app.agents.checker import AnswerChecker
from app.agents.generator import AnswerGenerator
from app.agents.orchestrator import Orchestrator
from app.agents.prompts import FALLBACK
from app.call.session import CallSession
from app.config import Settings
from app.observability.latency import TurnMetrics
from app.providers import create_chat, create_embeddings
from app.rag.retriever import Retriever
from app.rag.vector_store import SQLiteVectorStore
from app.tts.deepdub import DeepdubTTSProvider


class SilentASR:
    async def start(self, partial, final):
        self.final = final

    async def close(self):
        pass


class CaptureTransport:
    def __init__(self):
        self.events = []
        self.frames = 0
        self.session = None

    async def event(self, payload):
        self.events.append(payload)
        if payload["type"] == "audio_end":
            self.session.acknowledge_playback(payload["generation"])

    async def audio(self, pcm, generation):
        self.frames += 1


async def run(voice: bool):
    config = Settings()
    if not config.google_api_key and config.ai_provider == "google":
        raise RuntimeError("Configure GOOGLE_API_KEY to run live verification")
    store = SQLiteVectorStore(config.db_path)
    if not store.count_for_model(config.embedding_model):
        raise RuntimeError("No index for configured embedding model")
    embedding = create_embeddings(config)
    generator_chat, checker_chat = create_chat(config), create_chat(config)
    retriever = Retriever(embedding, store, config.embedding_model, config.top_k)
    checker = AnswerChecker(checker_chat, retriever)
    orchestrator = Orchestrator(retriever, AnswerGenerator(generator_chat), checker, config.min_similarity)
    report = {"index_count": store.count_for_model(config.embedding_model)}
    failures = []
    error = None
    try:
        async def state(_):
            pass

        trace = {}
        report["article_24"] = {"trace": trace}
        answer, _ = await orchestrator.answer("Что говорит статья 24 о свободе передвижения и возможных ограничениях?", [], TurnMetrics("smoke", 1), state, trace)
        report["article_24"].update({"abstained": answer == FALLBACK, "attempts": len(trace["attempts"]), "actions": [x.get("checker", {}).get("action") for x in trace["attempts"]]})
        if answer == FALLBACK:
            failures.append("Answerable Article 24 question abstained")
        if answer != FALLBACK and (not trace["attempts"] or trace["attempts"][-1].get("checker", {}).get("action") != "PASS"):
            failures.append("Article 24 answer escaped PASS gate")
        fabricated = {"answer": "Свобода передвижения не ограничена.", "cited_articles": ["24"], "claims": [{"text": "Свобода передвижения не ограничена", "article": "24", "quote": "Этой выдуманной фразы нет в Конституции"}]}
        verdict = await checker.check("Что говорит статья 24?", await retriever.verifier_search("Что говорит статья 24?"), fabricated)
        report["fabricated_quote_action"] = verdict["action"]
        if verdict["action"] == "PASS":
            failures.append("Fabricated quote passed")
        omitted = {"answer": "Каждый может свободно передвигаться по Казахстану без ограничений.", "cited_articles": ["24"], "claims": [{"text": "Свобода передвижения без ограничений", "article": "24", "quote": "право свободного передвижения по ее территории"}]}
        omitted_verdict = await checker.check("Есть ли ограничения свободы передвижения по статье 24?", await retriever.verifier_search("Есть ли ограничения свободы передвижения по статье 24?"), omitted)
        report["omitted_condition"] = {"action": omitted_verdict["action"], "missing_conditions": omitted_verdict["missing_conditions"], "unsupported_claims": omitted_verdict["unsupported_claims"]}
        if omitted_verdict["action"] == "PASS":
            failures.append("Omitted legal condition passed")
        out, _ = await orchestrator.answer("Какой курс евро будет завтра?", [], TurnMetrics("smoke", 2), state)
        report["out_of_scope_abstained"] = out == FALLBACK
        if out != FALLBACK:
            failures.append("Out-of-scope question did not abstain")
        if voice:
            if not config.deepdub_api_key or not config.deepdub_voice_id:
                raise RuntimeError("Configure Deepdub key and voice for --voice")
            transport = CaptureTransport()
            tts = DeepdubTTSProvider(config.deepdub_api_key, config.deepdub_voice_id, config.deepdub_base_url, config.deepdub_locale)
            session = CallSession(transport, SilentASR(), orchestrator, tts)
            transport.session = session
            try:
                await session.start()
                await session.asr.final("Что говорит статья 24 о свободе передвижения?")
                await asyncio.wait_for(session.task, timeout=90)
                report["voice"] = {"frames": transport.frames, "ack_before_answer": next((i for i, e in enumerate(transport.events) if e["type"] == "acknowledgement"), 10**9) < next((i for i, e in enumerate(transport.events) if e["type"] == "answer"), -1), "committed_history": bool(session.history), "metrics_ms": next((e["metrics_ms"] for e in transport.events if e["type"] == "metrics"), {})}
                if transport.frames == 0 or not report["voice"]["ack_before_answer"] or not session.history or not {"end_of_turn_to_first_acknowledgement_audio", "end_of_turn_to_first_substantive_audio"}.issubset(report["voice"]["metrics_ms"]) or any(e["type"] == "error" for e in transport.events):
                    failures.append("Voice acknowledgement or checked answer failed")
            finally:
                await session.close()
    except Exception as exc:
        error = exc
        report["error"] = {"type": type(exc).__name__}
    finally:
        await embedding.client.aclose()
        await generator_chat.client.aclose()
        await checker_chat.client.aclose()
    target = Path("artifacts/verification-smoke.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if error:
        raise RuntimeError("Verification smoke errored; inspect artifacts/verification-smoke.json") from error
    if failures:
        raise RuntimeError("Verification smoke failed: " + "; ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run real provider verification")
    parser.add_argument("--voice", action="store_true", help="Include Deepdub audio playback simulation")
    args = parser.parse_args()
    if args.live:
        asyncio.run(run(args.voice))
    else:
        parser.print_help()
