import asyncio
from time import perf_counter
from .prompts import FALLBACK
from app.observability.logging import event


class Orchestrator:
    def __init__(self, retriever, generator, checker, min_similarity: float = 0.28):
        self.retriever, self.generator, self.checker = retriever, generator, checker
        self.min_similarity = min_similarity

    async def answer(self, question: str, history: list[dict], metrics, state, trace: dict | None = None, prepared=None, candidate_audio=None) -> tuple[str, list[dict]]:
        retrieval = prepared or self.retriever
        verifier_task = asyncio.create_task(getattr(retrieval, "verifier_search", retrieval.search)(question))
        try:
            await state("retrieving")
            started = perf_counter()
            hits = await retrieval.search(question)
            hits = [hit for hit in hits if hit["score"] >= self.min_similarity or hit.get("hybrid_match") or hit.get("lexical_match") or hit.get("exact_article_match")]
            if trace is not None:
                trace["retrieved_articles"] = [h["article"] for h in hits]
                trace["attempts"] = []
            metrics.elapsed("retrieval", started)
            event("retriever", metrics.call_id, metrics.turn_id, latency_ms=metrics.durations_ms["retrieval"], top_k=len(hits), at_monotonic=perf_counter())
            if not hits:
                return FALLBACK, []
            feedback = None
            for attempt in range(2):
                await state("generating")
                started = perf_counter()
                candidate, ttft = await self.generator.generate(question, history, hits, feedback)
                if trace is not None:
                    trace["attempts"].append({"candidate": candidate})
                metrics.durations_ms["generator_total"] = round(metrics.durations_ms.get("generator_total", 0) + (perf_counter() - started) * 1000, 2)
                event("generator", metrics.call_id, metrics.turn_id, attempt=attempt + 1, latency_ms=round((perf_counter()-started)*1000,2), ttft_ms=ttft, at_monotonic=perf_counter())
                if ttft is not None and "generator_ttft" not in metrics.durations_ms:
                    metrics.durations_ms["generator_ttft"] = round(ttft, 2)
                if candidate["insufficient_context"]:
                    if attempt == 0:
                        independent = await verifier_task
                        seen = {(h.get("id"), h["article"], h["text"]) for h in hits}
                        added = False
                        for hit in independent:
                            key = (hit.get("id"), hit["article"], hit["text"])
                            if key not in seen:
                                hits.append(hit)
                                seen.add(key)
                                added = True
                        if added:
                            feedback = {"action": "RETRY", "reason": "Additional source passages are available. Reconsider the question using the expanded passages."}
                            continue
                    return FALLBACK, []
                audio_key = await candidate_audio.start(candidate["answer"], attempt) if candidate_audio else None
                await state("checking")
                started = perf_counter()
                independent = await verifier_task
                if prepared:
                    verdict = await self.checker.check(question, independent, candidate, retrieval=prepared, metrics=metrics)
                else:
                    verdict = await self.checker.check(question, independent, candidate)
                if trace is not None:
                    trace["attempts"][-1]["checker"] = {k: v for k, v in verdict.items() if k != "extra_evidence"}
                metrics.durations_ms["checker"] = round(metrics.durations_ms.get("checker", 0) + (perf_counter() - started) * 1000, 2)
                event("checker", metrics.call_id, metrics.turn_id, attempt=attempt + 1, latency_ms=round((perf_counter()-started)*1000,2), action=verdict["action"], rejection_reason=verdict.get("rejection_reason"), schema_errors=verdict.get("schema_errors", []), at_monotonic=perf_counter())
                if verdict["action"] == "PASS":
                    if candidate_audio:
                        candidate_audio.approve(audio_key, candidate["answer"])
                    source_articles = set(candidate["cited_articles"]) | set(verdict.get("verified_cross_references", []))
                    seen = {(h.get("id"), h["article"], h["text"]) for h in hits}
                    for hit in verdict.get("extra_evidence", []):
                        key = (hit.get("id"), hit["article"], hit["text"])
                        if hit["article"] in source_articles and key not in seen:
                            hits.append(hit)
                            seen.add(key)
                    return candidate["answer"], [h for h in hits if h["article"] in source_articles]
                if candidate_audio:
                    await candidate_audio.discard()
                if verdict["action"] == "INSUFFICIENT_CONTEXT":
                    return FALLBACK, []
                feedback = {k: v for k, v in verdict.items() if k != "extra_evidence"}
                if attempt == 0:
                    seen = {(h.get("id"), h["article"], h["text"]) for h in hits}
                    for hit in [*independent, *verdict.get("extra_evidence", [])]:
                        key = (hit.get("id"), hit["article"], hit["text"])
                        if key not in seen:
                            hits.append(hit)
                            seen.add(key)
            return FALLBACK, []
        finally:
            if not verifier_task.done():
                verifier_task.cancel()
            await asyncio.gather(verifier_task, return_exceptions=True)

    async def answer_speak_first(self, question: str, history: list[dict], metrics, state, trace: dict | None = None, prepared=None):
        """Speak-first variant: returns as soon as the `answer` text is
        readable in the generator's raw stream — typically before the model
        finishes writing `claims`/`cited_articles` after it, since those
        quote long verbatim evidence spans — without waiting for the
        independent checker either. The checker still runs, as `verify_task`;
        it resolves to None if the candidate is confirmed, or to a safe
        fallback string if the caller must interrupt and correct. No retry
        loop here — once speech has started on a draft, the only safe
        recovery is a correction, not a second generation.

        The draft text is read from the same model output the final
        validated result comes from, not guessed, so it never disagrees with
        it; it can still turn out insufficient_context or fail validation,
        which is handled as a correction exactly like a checker rejection.
        Sources returned alongside the draft are the full retrieved set, not
        yet narrowed to `cited_articles` (which streams in after `answer`) —
        a known precision trade-off in the UI's sources panel only; the
        checker verifies the real claims against real evidence regardless.
        """
        retrieval = prepared or self.retriever
        verifier_task = asyncio.create_task(getattr(retrieval, "verifier_search", retrieval.search)(question))
        await state("retrieving")
        started = perf_counter()
        hits = await retrieval.search(question)
        hits = [hit for hit in hits if hit["score"] >= self.min_similarity or hit.get("hybrid_match") or hit.get("lexical_match") or hit.get("exact_article_match")]
        metrics.elapsed("retrieval", started)
        event("retriever", metrics.call_id, metrics.turn_id, latency_ms=metrics.durations_ms["retrieval"], top_k=len(hits), at_monotonic=perf_counter())
        if not hits:
            verifier_task.cancel()
            await asyncio.gather(verifier_task, return_exceptions=True)
            return FALLBACK, [], None

        await state("generating")
        gen_started = perf_counter()
        loop = asyncio.get_event_loop()
        draft_future, candidate_future = loop.create_future(), loop.create_future()

        async def drive_generation():
            try:
                if not hasattr(self.generator, "generate_streaming"):
                    # Generator doesn't support early-draft extraction (e.g. a
                    # test double, or a non-streaming implementation): fall
                    # back to one call, same as the pre-streaming behavior.
                    candidate, ttft = await self.generator.generate(question, history, hits)
                    draft_future.set_result(candidate["answer"])
                    candidate_future.set_result((candidate, ttft))
                    return
                async for kind, *rest in self.generator.generate_streaming(question, history, hits):
                    if kind == "draft":
                        if not draft_future.done():
                            draft_future.set_result(rest[0])
                    else:
                        candidate, ttft = rest
                        if not draft_future.done():
                            draft_future.set_result(candidate["answer"])
                        if not candidate_future.done():
                            candidate_future.set_result((candidate, ttft))
            except Exception as exc:
                if not draft_future.done():
                    draft_future.set_exception(exc)
                if not candidate_future.done():
                    candidate_future.set_exception(exc)

        drive_task = asyncio.create_task(drive_generation())
        try:
            draft_text = await draft_future
        except Exception as exc:
            event("generator_error", metrics.call_id, metrics.turn_id, error_type=type(exc).__name__, at_monotonic=perf_counter())
            verifier_task.cancel()
            await asyncio.gather(verifier_task, drive_task, return_exceptions=True)
            return FALLBACK, [], None
        metrics.durations_ms["generator_draft_ms"] = round((perf_counter() - gen_started) * 1000, 2)
        if trace is not None:
            trace["draft"] = draft_text

        async def finish_and_verify():
            try:
                try:
                    candidate, ttft = await candidate_future
                except Exception as exc:
                    event("generator_error", metrics.call_id, metrics.turn_id, error_type=type(exc).__name__, at_monotonic=perf_counter())
                    return FALLBACK
                metrics.durations_ms["generator_total"] = round((perf_counter() - gen_started) * 1000, 2)
                event("generator", metrics.call_id, metrics.turn_id, attempt=1, latency_ms=metrics.durations_ms["generator_total"], ttft_ms=ttft, at_monotonic=perf_counter())
                if ttft is not None:
                    metrics.durations_ms["generator_ttft"] = round(ttft, 2)
                if trace is not None:
                    trace["candidate"] = candidate
                if candidate["insufficient_context"]:
                    return FALLBACK
                await state("checking")
                checker_started = perf_counter()
                independent = await verifier_task
                if prepared:
                    verdict = await self.checker.check(question, independent, candidate, retrieval=prepared, metrics=metrics)
                else:
                    verdict = await self.checker.check(question, independent, candidate)
                metrics.durations_ms["checker"] = round((perf_counter() - checker_started) * 1000, 2)
                event("checker", metrics.call_id, metrics.turn_id, attempt=1, latency_ms=metrics.durations_ms["checker"], action=verdict["action"], rejection_reason=verdict.get("rejection_reason"), schema_errors=verdict.get("schema_errors", []), at_monotonic=perf_counter())
                if trace is not None:
                    trace["checker"] = {k: v for k, v in verdict.items() if k != "extra_evidence"}
                return None if verdict["action"] == "PASS" else FALLBACK
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                event("checker_error", metrics.call_id, metrics.turn_id, error_type=type(exc).__name__, at_monotonic=perf_counter())
                return FALLBACK
            finally:
                # If the turn is interrupted while finish_and_verify is still
                # waiting on candidate_future, drive_generation (and the live
                # HTTP stream it holds) would otherwise keep running unobserved.
                if not drive_task.done():
                    drive_task.cancel()
                await asyncio.gather(drive_task, return_exceptions=True)

        verify_task = asyncio.create_task(finish_and_verify())
        return draft_text, hits, verify_task
