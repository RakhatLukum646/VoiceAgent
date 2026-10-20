"""Live evaluation; requires real credentials and indexed corpus. No synthetic scores."""
import asyncio
import json
from pathlib import Path
from time import perf_counter
from app.config import Settings
from app.providers import create_embeddings, create_chat
from app.rag.vector_store import SQLiteVectorStore
from app.rag.retriever import Retriever
from app.agents.generator import AnswerGenerator
from app.agents.checker import AnswerChecker
from app.agents.orchestrator import Orchestrator
from app.agents.prompts import FALLBACK
from app.observability.latency import TurnMetrics


async def main():
    config = Settings()
    if config.ai_provider == "google" and not config.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY required for live evaluation")
    if config.ai_provider == "openai" and not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY required for live evaluation")
    store = SQLiteVectorStore(config.db_path)
    if not store.count_for_model(config.embedding_model):
        raise RuntimeError("Ingest Constitution before evaluation")
    embeddings = create_embeddings(config)
    generator_chat = create_chat(config)
    checker_chat = create_chat(config)
    retriever = Retriever(embeddings, store, config.embedding_model, config.top_k)
    pipeline = Orchestrator(retriever, AnswerGenerator(generator_chat), AnswerChecker(checker_chat, retriever), config.min_similarity)
    cases = json.loads(Path("data/eval_questions.json").read_text(encoding="utf-8"))
    results = []
    try:
        for i, case in enumerate(cases, 1):
            metrics = TurnMetrics("evaluation", i)
            trace = {}
            started = perf_counter()
            async def state(value): pass
            answer, hits = await pipeline.answer(case["question"], [], metrics, state, trace)
            final_attempt = trace["attempts"][-1] if trace["attempts"] else {}
            candidate = final_attempt.get("candidate", {})
            verdict = final_attempt.get("checker", {})
            citations = set(candidate.get("cited_articles", []))
            citation_correct = bool(citations & set(case["articles"])) and citations.issubset({h["article"] for h in hits}) if candidate and case["answerable"] else None
            result = {"question": case["question"], "category": case["category"], "expected_behavior": case["expected_behavior"], "expected_articles": case["articles"], "retrieved_articles": trace["retrieved_articles"], "retrieval_correct": bool(set(case["articles"]) & set(trace["retrieved_articles"])) if case["answerable"] else None, "citation_correct_vs_gold": citation_correct, "checker_reported_grounded": verdict.get("grounded"), "checker_action": verdict.get("action"), "checker_unsupported_claims": verdict.get("unsupported_claims"), "attempts": trace["attempts"], "abstained": answer == FALLBACK, "expected_abstention": case["expected_behavior"] == "abstain", "response_latency_ms": round((perf_counter()-started)*1000, 2), "stage_latency_ms": metrics.durations_ms, "answer": answer}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    finally:
        await embeddings.client.aclose()
        await generator_chat.client.aclose()
        await checker_chat.client.aclose()
    actions = [a["checker"]["action"] for r in results for a in r["attempts"] if "checker" in a]
    score = {"cases": len(results), "retrieval_recall_on_answerable": sum(bool(r["retrieval_correct"]) for r in results if r["retrieval_correct"] is not None) / sum(r["retrieval_correct"] is not None for r in results), "citation_correct_on_answerable_with_candidate": sum(bool(r["citation_correct_vs_gold"]) for r in results if r["citation_correct_vs_gold"] is not None) / max(1,sum(r["citation_correct_vs_gold"] is not None for r in results)), "checker_passes_all_attempts": actions.count("PASS"), "checker_rejects_all_attempts": sum(a in ("RETRY","INSUFFICIENT_CONTEXT") for a in actions), "checker_reported_grounded_rate": sum(r["checker_reported_grounded"] is True for r in results) / max(1,sum(r["checker_reported_grounded"] is not None for r in results)), "abstention_accuracy_on_expected_abstain": sum(r["abstained"] for r in results if r["expected_abstention"]) / sum(r["expected_abstention"] for r in results), "mean_response_ms": round(sum(r["response_latency_ms"] for r in results)/len(results), 2)}
    Path("data/eval_results.json").write_text(json.dumps({"summary": score, "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(score, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
