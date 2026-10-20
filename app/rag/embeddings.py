import httpx
import math


class OpenAIEmbeddings:
    def __init__(self, api_key: str, base_url: str, model: str, client: httpx.AsyncClient | None = None):
        self.api_key, self.base_url, self.model = api_key, base_url.rstrip("/"), model
        self.client = client or httpx.AsyncClient(timeout=25)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for semantic embeddings")
        response = await self.client.post(f"{self.base_url}/embeddings", headers={"Authorization": f"Bearer {self.api_key}"}, json={"model": self.model, "input": texts})
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(len(texts))):
            raise ValueError("Embedding result count/index mismatch")
        vectors = [row["embedding"] for row in rows]
        if any(not v or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in v) or not any(x != 0 for x in v) for v in vectors) or len({len(v) for v in vectors}) != 1:
            raise ValueError("Invalid embedding vector")
        return vectors


class GoogleEmbeddings:
    """Gemini batchEmbedContents. Official contract: https://ai.google.dev/api/embeddings"""

    def __init__(self, api_key: str, base_url: str, model: str, client: httpx.AsyncClient | None = None):
        self.api_key, self.base_url, self.model = api_key, base_url.rstrip("/"), model
        self.client = client or httpx.AsyncClient(timeout=40)

    async def embed(self, texts: list[str], task_type: str = "RETRIEVAL_QUERY") -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for semantic embeddings")
        if not texts:
            return []
        if task_type not in ("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"):
            raise ValueError("Unsupported embedding task type")
        model_name = f"models/{self.model}"
        body = {"requests": [{"model": model_name, "content": {"parts": [{"text": text}]}, "taskType": task_type} for text in texts]}
        response = await self.client.post(f"{self.base_url}/{model_name}:batchEmbedContents", headers={"x-goog-api-key": self.api_key}, json=body)
        if response.status_code != 200:
            raise RuntimeError(f"Google embeddings failed (HTTP {response.status_code})")
        embeddings = response.json().get("embeddings", [])
        vectors = [row.get("values", []) for row in embeddings]
        if len(vectors) != len(texts) or any(not v or any(type(x) not in (int, float) or not math.isfinite(x) for x in v) or not any(x != 0 for x in v) for v in vectors) or len({len(v) for v in vectors}) != 1:
            raise ValueError("Invalid Google embedding response")
        return vectors
