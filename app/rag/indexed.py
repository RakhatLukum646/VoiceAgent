"""Qdrant hybrid index. Every online operation is pinned to a server-owned scope.

No corpus-wide vector/text scan on the call path. Dense vectors remain Gemini;
BM25 is the local qdrant/bm25 model built into the pinned Qdrant server.
"""
import asyncio
import hashlib
import json
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import httpx

BM25_OPTIONS = {"language": "russian", "tokenizer": "word", "lowercase": True}


def bm25(text):
    return {"model": "qdrant/bm25", "text": text, "options": BM25_OPTIONS}


@dataclass(frozen=True)
class Scope:
    collection: str
    tenant: str
    service: str
    version: str
    model: str
    language: str = "ru"

    def fields(self):
        return {"tenant": self.tenant, "service": self.service, "corpus_version": self.version, "model": self.model, "language": self.language}

    def filter(self):
        return {"must": [{"key": key, "match": {"value": value}} for key, value in self.fields().items()]}

    def accepts(self, payload):
        return all(payload.get(key) == value for key, value in self.fields().items())


def read_manifest(path, tenant, service, model):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("dataset_kind") != "official" or value.get("tenant") != tenant or value.get("service") != service or value.get("model") != model:
        raise ValueError("Active corpus does not match official service/model scope")
    if not value.get("collection", "").startswith("official_") or not value.get("version") or not value.get("passages"):
        raise ValueError("Invalid official corpus manifest")
    return value, Scope(value["collection"], tenant, service, value["version"], model)


class QdrantIndex:
    def __init__(self, url, client=None):
        self.client = client or httpx.AsyncClient(base_url=url.rstrip("/"), timeout=20)
        self._slots = asyncio.Semaphore(12)

    async def request(self, method, path, body=None):
        async with self._slots:
            response = await self.client.request(method, path, json=body)
            if response.status_code >= 400:
                raise RuntimeError(f"Retrieval index failed (HTTP {response.status_code})")
            return response.json().get("result")

    async def search(self, vector, text, scope, limit):
        candidates = min(60, max(20, limit * 3))
        result = await self.request("POST", f"/collections/{scope.collection}/points/query", {
            "prefetch": [
                {"query": vector, "using": "dense", "filter": scope.filter(), "limit": candidates, "params": {"hnsw_ef": 256}},
                {"query": bm25(text), "using": "bm25", "filter": scope.filter(), "limit": candidates},
            ], "query": {"fusion": "rrf"}, "filter": scope.filter(), "limit": limit, "with_payload": True,
        })
        hits = []
        for point in result["points"]:
            payload = point["payload"]
            if not scope.accepts(payload):
                raise RuntimeError("Retrieval scope mismatch")
            hits.append(payload | {"id": point["id"], "score": point["score"], "hybrid_match": True})
        return hits

    async def lookup(self, ids, scope):
        # IDs are UUID chunk IDs, not arbitrary document text. Scope checked even
        # for point retrieval, where the REST API does not accept query filters.
        valid = []
        for identity in list(dict.fromkeys(ids))[:30]:
            try:
                valid.append(str(uuid.UUID(identity)))
            except (ValueError, TypeError, AttributeError):
                continue
        if not valid:
            return []
        result = await self.request("POST", f"/collections/{scope.collection}/points", {"ids": valid, "with_payload": True, "with_vector": False})
        return [point["payload"] | {"id": point["id"], "score": 0.0} for point in result if scope.accepts(point["payload"])]

    async def count(self, scope):
        value = await self.request("POST", f"/collections/{scope.collection}/points/count", {"filter": scope.filter(), "exact": True})
        return value["count"]


class TurnRetrieval:
    evidence_complete = False

    def __init__(self, owner, question, history, scope):
        self.owner, self.scope = owner, scope
        self.question = question
        # Short elliptical follow-ups get bounded conversation context. The user
        # cannot change service/version/ACL filters with question or history text.
        prior = [h["content"] for h in history[-6:] if h["role"] == "user" and any(term in h["content"].casefold() for term in ("gold", "голд"))]
        explicit = any(term in question.casefold() for term in ("gold", "голд"))
        self.query = question if len(question.split()) >= 7 or explicit or not prior else prior[-1][:160] + "\nУточнение: " + question
        self.timings = {}
        self._vector_task = asyncio.create_task(self._vector())
        self._main = asyncio.create_task(self._search(5, "retrieval_index"))
        self._verifier = asyncio.create_task(self._search(10, "verifier_index"))
        self.closed = False

    async def _vector(self):
        start = perf_counter()
        vector = (await self.owner.embed([self.query]))[0]
        self.timings["query_embedding"] = round((perf_counter() - start) * 1000, 2)
        return vector

    async def _search(self, limit, key):
        vector = await self._vector_task
        start = perf_counter()
        result = await self.owner.index.search(vector, self.query, self.scope, limit)
        self.timings[key] = round((perf_counter() - start) * 1000, 2)
        return result

    async def search(self, question=None):
        if question is None or question == self.question:
            return await self._main
        return (await self.batch_claim_search([question]))[0]

    async def verifier_search(self, question=None):
        return await self._verifier

    async def batch_claim_search(self, texts):
        unique = list(dict.fromkeys(texts))[:3]
        vectors = await self.owner.embed(unique)
        return await asyncio.gather(*(self.owner.index.search(vector, text, self.scope, 4) for text, vector in zip(unique, vectors)))

    async def lookup_articles(self, articles, neighbors=False):
        return await self.owner.index.lookup(articles, self.scope)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        for task in (self._main, self._verifier, self._vector_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(self._main, self._verifier, self._vector_task, return_exceptions=True)


class IndexedRetriever:
    def __init__(self, embeddings, index, manifest_path, tenant, service, model):
        self.embeddings, self.index = embeddings, index
        self.path, self.tenant, self.service, self.model = manifest_path, tenant, service, model
        self._cache = OrderedDict()
        self._embed_slots = asyncio.Semaphore(6)

    def manifest(self):
        return read_manifest(self.path, self.tenant, self.service, self.model)

    def prepare(self, question, history):
        _, scope = self.manifest()
        return TurnRetrieval(self, question, history, scope)

    async def embed(self, texts):
        if not texts:
            return []
        normalized = [" ".join(text.split()) for text in texts]
        resolved = {text: self._cache[text] for text in normalized if text in self._cache}
        missing = list(dict.fromkeys(text for text in normalized if text not in resolved))
        if missing:
            async with self._embed_slots:
                values = await asyncio.wait_for(self.embeddings.embed(missing), 15)
            for text, vector in zip(missing, values):
                self._cache[text] = vector
                resolved[text] = vector
            while len(self._cache) > 256:
                self._cache.popitem(last=False)
        return [resolved[text] for text in normalized]

    async def search(self, question):
        prepared = self.prepare(question, [])
        try:
            return await prepared.search()
        finally:
            await prepared.close()
