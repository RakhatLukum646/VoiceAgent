import json
import copy
from time import perf_counter
import httpx
from .prompts import GENERATOR, CHECKER, FALLBACK, context_text


GENERATOR_SCHEMA = {
    "type": "object", "properties": {
        "answer": {"type": "string"},
        "cited_articles": {"type": "array", "items": {"type": "string"}},
        "claims": {"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}, "article": {"type": "string"}, "quote": {"type": "string"}}, "required": ["text", "article", "quote"]}},
        "insufficient_context": {"type": "boolean"},
    }, "required": ["answer", "cited_articles", "claims", "insufficient_context"]
}
CHECKER_SCHEMA = {
    "type": "object", "properties": {
        **{name: {"type": "boolean"} for name in ("grounded", "relevant", "article_references_correct", "safe_to_send", "complete")},
        "missing_conditions": {"type": "array", "items": {"type": "string"}},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "action": {"type": "string", "enum": ["PASS", "RETRY", "INSUFFICIENT_CONTEXT"]},
    }, "required": ["grounded", "relevant", "article_references_correct", "safe_to_send", "complete", "missing_conditions", "unsupported_claims", "confidence", "action"]
}


class OpenAIChat:
    def __init__(self, api_key: str, base_url: str, model: str, client: httpx.AsyncClient | None = None):
        self.api_key, self.base_url, self.model = api_key, base_url.rstrip("/"), model
        self.client = client or httpx.AsyncClient(timeout=40)

    async def json_completion(self, system: str, payload: dict, stream: bool = False) -> tuple[dict, float | None]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY required for generator/checker")
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], "response_format": {"type": "json_object"}, "stream": stream, "temperature": 0}
        started = perf_counter()
        if not stream:
            response = await self.client.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"]), None
        pieces: list[str] = []
        first: float | None = None
        async with self.client.stream("POST", f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"}, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                choices = json.loads(line[6:]).get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    if first is None:
                        first = (perf_counter() - started) * 1000
                    pieces.append(delta)
        return json.loads("".join(pieces)), first


class GoogleChat:
    """Gemini generateContent/streamGenerateContent: https://ai.google.dev/api/generate-content"""

    def __init__(self, api_key: str, base_url: str, model: str, client: httpx.AsyncClient | None = None, thinking_level: str | None = "minimal", explicit_thinking: bool = False):
        self.api_key, self.base_url, self.model = api_key, base_url.rstrip("/"), model
        self.client = client or httpx.AsyncClient(timeout=60)
        self.thinking_level = thinking_level if thinking_level in {"minimal", "low", "medium", "high"} else None
        self.explicit_thinking = explicit_thinking

    @staticmethod
    def _candidate_text(payload: dict) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            raise ValueError("Google returned no candidate")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)

    async def json_completion(self, system: str, payload: dict, stream: bool = False) -> tuple[dict, float | None]:
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY required for generator/checker")
        body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0}}
        if system == GENERATOR:
            schema = copy.deepcopy(GENERATOR_SCHEMA)
            if payload.get("citation_ids"):
                allowed = payload["citation_ids"]
                schema["properties"]["cited_articles"]["items"]["enum"] = allowed
                schema["properties"]["claims"]["items"]["properties"]["article"]["enum"] = allowed
            body["generationConfig"]["responseJsonSchema"] = schema
        elif system == CHECKER:
            body["generationConfig"]["responseJsonSchema"] = CHECKER_SCHEMA
        # Checker thinking is opt-in via explicit_thinking; every other role (generator,
        # router) applies the configured budget by default so it is never left at the
        # model's full default thinking, which previously made router classification slow.
        if (system != CHECKER or self.explicit_thinking) and self.model.startswith("gemini-3") and "flash" in self.model and self.thinking_level:
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        headers = {"x-goog-api-key": self.api_key}
        endpoint = f"{self.base_url}/models/{self.model}:{'streamGenerateContent' if stream else 'generateContent'}"
        if not stream:
            response = await self.client.post(endpoint, headers=headers, json=body)
            if response.status_code != 200:
                raise RuntimeError(f"Google generation failed (HTTP {response.status_code})")
            return json.loads(self._candidate_text(response.json())), None
        started = perf_counter()
        first = None
        pieces = []
        async with self.client.stream("POST", endpoint, params={"alt": "sse"}, headers=headers, json=body) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Google streaming generation failed (HTTP {response.status_code})")
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if not event.get("candidates"):
                    continue
                fragment = self._candidate_text(event)
                if fragment:
                    if first is None:
                        first = (perf_counter() - started) * 1000
                    pieces.append(fragment)
        if not pieces:
            raise ValueError("Google returned no text")
        return json.loads("".join(pieces)), first


class AnswerGenerator:
    def __init__(self, chat: OpenAIChat):
        self.chat = chat

    async def generate(self, question: str, history: list[dict], hits: list[dict], feedback: dict | None = None) -> tuple[dict, float | None]:
        result, ttft = await self.chat.json_completion(GENERATOR, {"question": question, "history": history[-6:], "passages": context_text(hits), "citation_ids": list(dict.fromkeys(h["article"] for h in hits)), "checker_feedback": feedback}, stream=True)
        if result.get("insufficient_context") is True and result.get("answer") == "":
            result["answer"] = FALLBACK
        if not isinstance(result.get("answer"), str) or not result["answer"].strip() or len(result["answer"]) > 1200 or not isinstance(result.get("cited_articles"), list) or type(result.get("insufficient_context")) is not bool or not isinstance(result.get("claims"), list):
            raise ValueError("Invalid generator JSON")
        if any(not isinstance(x, str) for x in result["cited_articles"]):
            raise ValueError("Invalid article reference")
        if any(not isinstance(claim, dict) or set(claim) != {"text", "article", "quote"} or any(not isinstance(claim[key], str) or not claim[key].strip() for key in ("text", "article", "quote")) for claim in result["claims"]):
            raise ValueError("Invalid claims")
        if not result["insufficient_context"] and (not result["claims"] or set(result["cited_articles"]) != {claim["article"] for claim in result["claims"]}):
            raise ValueError("Uncited or missing claims")
        return result, ttft
