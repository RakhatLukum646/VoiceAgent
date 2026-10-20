import asyncio
import re
from time import perf_counter
from .prompts import CHECKER, context_text


class AnswerChecker:
    ACTIONS = {"PASS", "RETRY", "INSUFFICIENT_CONTEXT"}

    def __init__(self, chat, retriever=None):
        self.chat = chat
        self.retriever = retriever

    async def check(self, question: str, hits: list[dict], answer: dict, retrieval=None, metrics=None) -> dict:
        retriever = retrieval or self.retriever
        evidence_started = perf_counter()
        claims = answer.get("claims")
        cited = answer.get("cited_articles")
        if not isinstance(claims, list) or not claims or not isinstance(cited, list) or any(not isinstance(x, str) for x in cited):
            return self._reject("Missing evidence claims")
        if any(not isinstance(c, dict) or set(c) != {"text", "article", "quote"} or any(not isinstance(c[k], str) or not c[k].strip() for k in ("text", "article", "quote")) for c in claims):
            return self._reject("Invalid evidence claim")
        if set(cited) != {c["article"] for c in claims}:
            return self._reject("Article references do not match claims")
        spoken = answer.get("answer", "").casefold()
        spoken_articles = set(re.findall(r"(?:стать(?:я|и|ю|е|ей)|article)\s*(\d+)", spoken))
        evidence = list(hits)
        if retriever:
            searches = [retriever.lookup_articles(list(set(cited)))]
            if not getattr(retriever, "evidence_complete", False):
                texts = list(dict.fromkeys(c["text"] for c in claims))[:3]
                if hasattr(retriever, "batch_claim_search"):
                    groups = await retriever.batch_claim_search(texts)
                    for group in groups:
                        evidence.extend(group)
                else:
                    searches.extend(retriever.search(text) for text in texts)
            for found in await asyncio.gather(*searches):
                evidence.extend(found)
        seen = set()
        unique = []
        for hit in evidence:
            key = (hit.get("id"), hit["article"], hit["text"])
            if key not in seen:
                unique.append(hit)
                seen.add(key)
        evidence = unique
        invalid = [c["text"] for c in claims if not any(h["article"] == c["article"] and c["quote"] in h["text"] for h in evidence)]
        if invalid:
            return self._reject("Quote absent from cited Article", invalid, evidence)
        # A cited provision may itself refer to another Article by paragraph. That
        # cross-reference is supported by the exact quote; it is not a claim that
        # the referenced Article was independently cited as evidence.
        cross_refs = re.findall(r"пункт(?:ом|а|е)?\s+\d+\s+стать(?:я|и|ю|е|ей)\s*\d+", spoken)
        quoted_cross_refs = {ref for ref in cross_refs if any(ref in claim["quote"].casefold() for claim in claims)}
        referenced = {number for ref in quoted_cross_refs for number in re.findall(r"стать(?:я|и|ю|е|ей)\s*(\d+)", ref)} - set(cited)
        verified_references = set()
        if referenced and retriever:
            referenced_hits = await retriever.lookup_articles(sorted(referenced))
            verified_references = {hit["article"] for hit in referenced_hits} & referenced
            evidence.extend(referenced_hits)
        covered = set(cited) | verified_references
        if not spoken_articles.issubset(covered):
            return self._reject("Answer mentions an uncited Article", evidence=evidence)
        try:
            if metrics:
                metrics.elapsed("checker_evidence", evidence_started)
            model_started = perf_counter()
            result, _ = await self.chat.json_completion(CHECKER, {"question": question, "passages": context_text(evidence), "candidate": answer, "verified_cross_references": sorted(verified_references)})
            if metrics:
                metrics.elapsed("checker_model", model_started)
        except ValueError:
            return self._reject("Invalid checker JSON", evidence=evidence, schema_errors=["response"])
        if not isinstance(result, dict):
            return self._reject("Invalid checker JSON", evidence=evidence, schema_errors=["root"])
        errors = []
        for key in ("grounded", "relevant", "article_references_correct", "safe_to_send", "complete"):
            if type(result.get(key)) is not bool:
                errors.append(key)
        for key in ("unsupported_claims", "missing_conditions"):
            if not isinstance(result.get(key), list) or any(not isinstance(x, str) for x in result[key]):
                errors.append(key)
        if type(result.get("confidence")) not in (int, float) or not 0 <= result["confidence"] <= 1:
            errors.append("confidence")
        if result.get("action") not in self.ACTIONS:
            errors.append("action")
        if errors:
            return self._reject("Invalid checker JSON", evidence=evidence, schema_errors=errors)
        if result["action"] == "PASS" and (not all(result[k] for k in ("grounded", "relevant", "article_references_correct", "safe_to_send", "complete")) or result["unsupported_claims"] or result["missing_conditions"] or result["confidence"] < 0.6):
            result["action"] = "RETRY"
        result["extra_evidence"] = evidence
        result["verified_cross_references"] = sorted(verified_references)
        result["rejection_reason"] = "model_rejected" if result["action"] != "PASS" else None
        return result

    @staticmethod
    def _reject(reason: str, claims: list[str] | None = None, evidence: list[dict] | None = None, schema_errors: list[str] | None = None) -> dict:
        codes = {"Missing evidence claims": "missing_claims", "Invalid evidence claim": "invalid_claim", "Article references do not match claims": "citation_mismatch", "Answer mentions an uncited Article": "uncited_article", "Quote absent from cited Article": "quote_absent", "Invalid checker JSON": "invalid_checker_json"}
        return {"grounded": False, "relevant": True, "article_references_correct": False, "safe_to_send": False, "complete": False, "missing_conditions": [], "unsupported_claims": claims or [reason], "confidence": 0.0, "action": "RETRY", "extra_evidence": evidence or [], "schema_errors": schema_errors or [], "rejection_reason": codes.get(reason, "deterministic_reject")}
