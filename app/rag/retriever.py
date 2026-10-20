import asyncio


class Retriever:
    def __init__(self, embeddings, store, model: str, top_k: int = 5):
        self.embeddings, self.store, self.model, self.top_k = embeddings, store, model, top_k
        self.evidence_complete = store.count_for_model(model) == 1

    async def search(self, question: str) -> list[dict]:
        vector = (await self.embeddings.embed([question]))[0]
        return await asyncio.to_thread(self.store.search, vector, self.model, self.top_k, question)

    async def verifier_search(self, question: str) -> list[dict]:
        if self.evidence_complete:
            # The verifier independently loads the ENTIRE one-page corpus.
            # An additional vector request cannot discover any omitted passage.
            return await asyncio.to_thread(self.store.all_passages, self.model)
        vector = (await self.embeddings.embed([question]))[0]
        broad = await asyncio.to_thread(self.store.search, vector, self.model, self.top_k * 2, question)
        neighboring = await self.lookup_articles(list({h["article"] for h in broad}), neighbors=True)
        seen = set()
        evidence = []
        for hit in [*broad, *neighboring]:
            key = (hit.get("id"), hit["article"], hit["text"])
            if key not in seen:
                evidence.append(hit)
                seen.add(key)
        return evidence

    async def lookup_articles(self, articles: list[str], neighbors: bool = False) -> list[dict]:
        return await asyncio.to_thread(self.store.lookup_articles, articles, self.model, neighbors)
