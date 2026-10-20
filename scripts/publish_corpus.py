"""Offline immutable official publication; changed content/model embeddings only."""
import argparse
import asyncio
import hashlib
import json
import os
import uuid
from pathlib import Path
from app.config import Settings
from app.providers import create_embeddings
from app.rag.indexed import QdrantIndex, Scope, bm25


def chunk_id(source, text):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, source + "#parent-v1#" + hashlib.sha256(text.encode()).hexdigest()))


async def publish(documents_path, config=None, max_new_embeddings=32):
    config = config or Settings()
    raw = json.loads(Path(documents_path).read_text(encoding="utf-8"))
    if raw.get("dataset_kind") != "official" or raw.get("service") != config.service_scope:
        raise ValueError("Only verified official service corpus can be published")
    documents = raw["documents"]
    if not 1 <= len(documents) <= 100000:
        raise ValueError("Offline publication document limit exceeded")
    if any(not d["source"].startswith("https://guide.kaspi.kz/client/ru/gold/") or not d["text"].strip() or len(d["text"]) > 12000 for d in documents):
        raise ValueError("Invalid official source or extraction")
    canonical = json.dumps({"documents": [{k: d[k] for k in ("source", "title", "text", "updated")} for d in documents], "model": config.embedding_model, "sparse": "qdrant-bm25-russian-v1"}, ensure_ascii=False, sort_keys=True)
    version = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    collection = "official_" + config.service_scope.replace("-", "_") + "_" + version
    scope = Scope(collection, config.tenant_scope, config.service_scope, version, config.embedding_model)
    index = QdrantIndex(config.qdrant_url)
    cache_path = Path("data/document_embeddings.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    embedder = create_embeddings(config)
    try:
        ids, payloads, keys = [], [], []
        for document in documents:
            text = document["text"]
            identity = chunk_id(document["source"], text)
            ids.append(identity)
            keys.append(hashlib.sha256((config.embedding_model + "\n" + document["title"] + "\n" + text).encode()).hexdigest())
            payloads.append(scope.fields() | {"article": identity, "document": document["title"], "section": document["title"], "text": text,
                "source": document["source"], "version": document.get("updated", ""), "retrieved_at": document.get("retrieved_at", ""),
                "content_hash": hashlib.sha256(text.encode()).hexdigest(), "parent_id": document["source"], "dataset_kind": "official"})
        missing = [i for i, key in enumerate(keys) if key not in cache]
        if len(missing) > max_new_embeddings:
            raise ValueError("New document embedding budget exceeded; explicitly approve a larger offline ingestion budget")
        for offset in range(0, len(missing), 16):
            batch = missing[offset:offset + 16]
            texts = [payloads[i]["document"] + "\n" + payloads[i]["text"] for i in batch]
            vectors = await embedder.embed(texts, task_type="RETRIEVAL_DOCUMENT") if config.ai_provider == "google" else await embedder.embed(texts)
            for i, vector in zip(batch, vectors):
                cache[keys[i]] = vector
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        exists = await index.client.get(f"/collections/{collection}")
        if exists.status_code == 404:
            await index.request("PUT", f"/collections/{collection}", {"vectors": {"dense": {"size": len(cache[keys[0]]), "distance": "Cosine"}}, "sparse_vectors": {"bm25": {"modifier": "idf"}}, "hnsw_config": {"m": 32, "ef_construct": 200, "full_scan_threshold": 10}, "optimizers_config": {"indexing_threshold": 1000}})
            for name in scope.fields():
                await index.request("PUT", f"/collections/{collection}/index?wait=true", {"field_name": name, "field_schema": "keyword"})
        elif exists.status_code != 200:
            raise RuntimeError("Unable to inspect retrieval collection")
        # Idempotent staged upserts also recover a collection interrupted before
        # publication. Content hash fixes the content; no active pointer changes
        # until all records and the hybrid smoke are validated.
        for offset in range(0, len(ids), 100):
            points = [{"id": identity, "vector": {"dense": cache[key], "bm25": bm25(payload["text"])}, "payload": payload} for identity, key, payload in zip(ids[offset:offset+100], keys[offset:offset+100], payloads[offset:offset+100])]
            await index.request("PUT", f"/collections/{collection}/points?wait=true", {"points": points})
        if await index.count(scope) != len(ids):
            raise RuntimeError("Publication validation failed; active manifest unchanged")
        hits = await index.search(cache[keys[0]], payloads[0]["text"], scope, 3)
        if not hits or any(not scope.accepts(hit) for hit in hits):
            raise RuntimeError("Hybrid publication smoke failed")
        manifest = {"dataset_kind": "official", "collection": collection, "version": version, "tenant": scope.tenant, "service": scope.service, "model": scope.model,
                    "passages": len(ids), "documents": [{k: payload[k] for k in ("article", "source", "document", "content_hash", "retrieved_at", "version")} for payload in payloads]}
        target = Path(config.corpus_manifest)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
        Path("data/corpus_versions").mkdir(exist_ok=True)
        Path("data/corpus_versions", version + ".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"published": collection, "passages": len(ids), "new_embedding_documents": len(missing)}))
        return manifest
    finally:
        await embedder.client.aclose()
        await index.client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", default="data/official_corpus.json")
    parser.add_argument("--max-new-embeddings", type=int, default=32, help="Explicit paid-document embedding budget; unchanged content is reused")
    args = parser.parse_args()
    asyncio.run(publish(args.documents, max_new_embeddings=args.max_new_embeddings))
