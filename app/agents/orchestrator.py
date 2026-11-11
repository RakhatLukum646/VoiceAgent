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

    async def classify_and_answer(self, question: str, history: list[dict], metrics, state, trace: dict | None = None, prepared=None, candidate_audio=None):
        """Classifies intent and (for RAG) drafts+verifies the answer,
        merging what used to be two separately-awaited model round-trips
        (intent classification, then drafting) into one for the first
        attempt -- the two never actually depended on each other's output,
        so there was never a reason to pay for two calls. A checker-rejected
        retry still falls back to the plain, non-routed `generate()` (intent
        is already settled by then; no need to reclassify).

        Returns `(action, answer, hits)`. For non-RAG, `answer` is `None`
        and `hits` is empty -- the caller supplies its own fixed TRANSFER/
        REDIRECT phrase, exactly as it already does for `.route()` results.
        """
        retrieval = prepared or self.retriever
        verifier_task = asyncio.create_task(getattr(retrieval, "verifier_search", retrieval.search)(question))
        try:
            await state("retrieving")
            started = perf_counter()
            hits = await retrieval.search(question)
            hits = [hit for hit in hits if hit["score"] >= self.min_similarity or hit.get("hybrid_match") or hit.get("lexical_match") or hit.get("exact_article_match")]
            metrics.elapsed("retrieval", started)
            event("retriever", metrics.call_id, metrics.turn_id, latency_ms=metrics.durations_ms["retrieval"], top_k=len(hits), at_monotonic=perf_counter())
            await state("generating")
            started = perf_counter()
            candidate, ttft = await self.generator.generate_routed(question, history, hits)
            action = candidate["action"]
            if trace is not None:
                trace["action"] = action
                trace["attempts"] = [{"candidate": candidate}]
            metrics.durations_ms["generator_total"] = round((perf_counter() - started) * 1000, 2)
            event("generator", metrics.call_id, metrics.turn_id, attempt=1, latency_ms=metrics.durations_ms["generator_total"], ttft_ms=ttft, action=action, at_monotonic=perf_counter())
            if ttft is not None:
                metrics.durations_ms["generator_ttft"] = round(ttft, 2)
            if action != "RAG":
                return action, None, []
            if not hits:
                return action, FALLBACK, []
            feedback = None
            for attempt in range(2):
                if attempt > 0:
                    await state("generating")
                    started = perf_counter()
                    candidate, ttft = await self.generator.generate(question, history, hits, feedback)
                    if trace is not None:
                        trace["attempts"].append({"candidate": candidate})
                    metrics.durations_ms["generator_total"] += round((perf_counter() - started) * 1000, 2)
                    event("generator", metrics.call_id, metrics.turn_id, attempt=attempt + 1, latency_ms=round((perf_counter()-started)*1000,2), ttft_ms=ttft, at_monotonic=perf_counter())
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
                    return action, FALLBACK, []
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
                    return action, candidate["answer"], [h for h in hits if h["article"] in source_articles]
                if candidate_audio:
                    await candidate_audio.discard()
                if verdict["action"] == "INSUFFICIENT_CONTEXT":
                    return action, FALLBACK, []
                feedback = {k: v for k, v in verdict.items() if k != "extra_evidence"}
                if attempt == 0:
                    seen = {(h.get("id"), h["article"], h["text"]) for h in hits}
                    for hit in [*independent, *verdict.get("extra_evidence", [])]:
                        key = (hit.get("id"), hit["article"], hit["text"])
                        if key not in seen:
                            hits.append(hit)
                            seen.add(key)
            return action, FALLBACK, []
        finally:
            if not verifier_task.done():
                verifier_task.cancel()
            await asyncio.gather(verifier_task, return_exceptions=True)

    async def classify_and_answer_speak_first(self, question: str, history: list[dict], metrics, state, trace: dict | None = None, prepared=None, candidate_audio=None):
        """Speak-first variant of `classify_and_answer`: returns as soon as
        `action` is known and, for RAG, `answer` is readable in the same
        combined stream -- before the rest of the JSON (claims/citations)
        finishes and without waiting on the independent checker. Mirrors
        `answer_speak_first`'s contract and safety properties exactly (the
        checker still runs as `verify_task`; nothing it finds is skipped),
        the only difference is classification and drafting share one call.
        """
        retrieval = prepared or self.retriever
        verifier_task = asyncio.create_task(getattr(retrieval, "verifier_search", retrieval.search)(question))
        await state("retrieving")
        started = perf_counter()
        hits = await retrieval.search(question)
        hits = [hit for hit in hits if hit["score"] >= self.min_similarity or hit.get("hybrid_match") or hit.get("lexical_match") or hit.get("exact_article_match")]
        metrics.elapsed("retrieval", started)
        event("retriever", metrics.call_id, metrics.turn_id, latency_ms=metrics.durations_ms["retrieval"], top_k=len(hits), at_monotonic=perf_counter())

        await state("generating")
        gen_started = perf_counter()
        loop = asyncio.get_event_loop()
        action_future, draft_future, candidate_future = loop.create_future(), loop.create_future(), loop.create_future()
        early_insufficient = False

        async def drive_generation():
            nonlocal early_insufficient
            try:
                if not hasattr(self.generator, "generate_routed_streaming"):
                    candidate, ttft = await self.generator.generate_routed(question, history, hits)
                    early_insufficient = candidate.get("insufficient_context") is True
                    action_future.set_result(candidate["action"])
                    draft_future.set_result(candidate.get("answer", ""))
                    candidate_future.set_result((candidate, ttft))
                    return
                async for kind, *rest in self.generator.generate_routed_streaming(question, history, hits):
                    if kind == "action":
                        if not action_future.done():
                            action_future.set_result(rest[0])
                    elif kind == "insufficient":
                        early_insufficient = rest[0] is True
                    elif kind == "draft":
                        if not draft_future.done():
                            draft_future.set_result(rest[0])
                    else:
                        candidate, ttft = rest
                        if not action_future.done():
                            action_future.set_result(candidate["action"])
                        if not draft_future.done():
                            draft_future.set_result(candidate.get("answer", ""))
                        if not candidate_future.done():
                            candidate_future.set_result((candidate, ttft))
            except Exception as exc:
                for fut in (action_future, draft_future, candidate_future):
                    if not fut.done():
                        fut.set_exception(exc)

        drive_task = asyncio.create_task(drive_generation())
        try:
            action = await action_future
        except Exception as exc:
            event("generator_error", metrics.call_id, metrics.turn_id, error_type=type(exc).__name__, at_monotonic=perf_counter())
            verifier_task.cancel()
            await asyncio.gather(verifier_task, drive_task, return_exceptions=True)
            return "REDIRECT", None, [], None

        if action != "RAG":
            # No claim was ever drafted (or it's about to be discarded) --
            # nothing to verify, nothing to speak from this call. The caller
            # speaks its own fixed phrase, same as a non-streaming `.route()`.
            verifier_task.cancel()
            await asyncio.gather(verifier_task, drive_task, return_exceptions=True)
            return action, None, [], None

        if not hits:
            verifier_task.cancel()
            await asyncio.gather(verifier_task, drive_task, return_exceptions=True)
            return action, FALLBACK, [], None

        try:
            draft_text = await draft_future
        except Exception as exc:
            event("generator_error", metrics.call_id, metrics.turn_id, error_type=type(exc).__name__, at_monotonic=perf_counter())
            verifier_task.cancel()
            await asyncio.gather(verifier_task, drive_task, return_exceptions=True)
            return action, FALLBACK, [], None
        metrics.durations_ms["generator_draft_ms"] = round((perf_counter() - gen_started) * 1000, 2)
        if early_insufficient:
            # See answer_speak_first: never speak a draft the model disowned.
            event("generator_insufficient_context", metrics.call_id, metrics.turn_id, at_monotonic=perf_counter())
            verifier_task.cancel()
            drive_task.cancel()
            await asyncio.gather(verifier_task, drive_task, return_exceptions=True)
            return action, FALLBACK, [], None
        if trace is not None:
            trace["action"] = action
            trace["draft"] = draft_text
        if candidate_audio is not None:
            audio_key = await candidate_audio.start(draft_text, 0)
            candidate_audio.approve(audio_key, draft_text)

        async def finish_and_verify():
            try:
                try:
                    candidate, ttft = await candidate_future
                except Exception as exc:
                    event("generator_error", metrics.call_id, metrics.turn_id, error_type=type(exc).__name__, at_monotonic=perf_counter())
                    return FALLBACK
                metrics.durations_ms["generator_total"] = round((perf_counter() - gen_started) * 1000, 2)
                event("generator", metrics.call_id, metrics.turn_id, attempt=1, latency_ms=metrics.durations_ms["generator_total"], ttft_ms=ttft, action=action, at_monotonic=perf_counter())
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
                if not drive_task.done():
                    drive_task.cancel()
                await asyncio.gather(drive_task, return_exceptions=True)

        verify_task = asyncio.create_task(finish_and_verify())
        return action, draft_text, hits, verify_task

    async def answer_speak_first(self, question: str, history: list[dict], metrics, state, trace: dict | None = None, prepared=None, candidate_audio=None):
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
        early_insufficient = False

        async def drive_generation():
            nonlocal early_insufficient
            try:
                if not hasattr(self.generator, "generate_streaming"):
                    # Generator doesn't support early-draft extraction (e.g. a
                    # test double, or a non-streaming implementation): fall
                    # back to one call, same as the pre-streaming behavior.
                    candidate, ttft = await self.generator.generate(question, history, hits)
                    early_insufficient = candidate["insufficient_context"] is True
                    draft_future.set_result(candidate["answer"])
                    candidate_future.set_result((candidate, ttft))
                    return
                async for kind, *rest in self.generator.generate_streaming(question, history, hits):
                    if kind == "insufficient":
                        early_insufficient = rest[0] is True
                    elif kind == "draft":
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
        if early_insufficient:
            # The model said up front it can't support an answer: speak the
            # fallback now rather than a draft it already disowned, followed
            # by a contradicting correction 10s later.
            event("generator_insufficient_context", metrics.call_id, metrics.turn_id, at_monotonic=perf_counter())
            verifier_task.cancel()
            drive_task.cancel()
            await asyncio.gather(verifier_task, drive_task, return_exceptions=True)
            return FALLBACK, [], None
        if trace is not None:
            trace["draft"] = draft_text
        if candidate_audio is not None:
            # Speak-first's premise is "speak the unverified draft now, correct
            # later" -- approving the draft's audio the instant it's extracted
            # (rather than waiting on `verify_task`) is consistent with that,
            # not a new risk. This lets synthesis run concurrently with the
            # background checker instead of starting only once the caller
            # reaches its own TTS call.
            audio_key = await candidate_audio.start(draft_text, 0)
            candidate_audio.approve(audio_key, draft_text)

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
