"""Fetch official source, split by article, create real embeddings, atomically replace local index."""
import argparse
import asyncio
from datetime import date
from pathlib import Path
from dataclasses import replace
import httpx
from bs4 import BeautifulSoup
from app.config import Settings
from app.rag.chunking import chunk_articles
from app.providers import create_embeddings
from app.rag.vector_store import SQLiteVectorStore

DEFAULT_URL = "https://zan.gov.kz/api/documents/223244/rus/download/pdf"
DEFAULT_FILE = "data/constitution_clean.txt"


async def ingest(url: str, config: Settings, file: str | None = None):
    if file:
        raw = Path(file).read_text(encoding="utf-8")
    else:
        if url.lower().endswith("/pdf"):
            raise RuntimeError("PDF requires text extraction; use --file data/constitution_clean.txt or a verified official HTML URL")
        async with httpx.AsyncClient(timeout=45, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            raw = response.text
    if file and Path(file).suffix.lower() == ".txt":
        text = raw
    else:
        soup = BeautifulSoup(raw, "html.parser")
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()
        text = soup.get_text("\n", strip=True)
    text = text.replace("Ваш браузер не поддерживает аудио. / Your browser does not support the audio element.", "")
    chunks = chunk_articles(text, url)
    articles = {p.article for p in chunks}
    if articles != {str(i) for i in range(1, 97)} or "1 июля 2026 года" not in text:
        raise RuntimeError(f"Official 2026 text validation failed ({len(articles)} articles); index left untouched")
    Path("data").mkdir(exist_ok=True)
    Path("data/constitution_index_input.txt").write_text(text, encoding="utf-8")
    chunks = [replace(p, retrieved_at=date.today().isoformat()) for p in chunks]
    embeddings = create_embeddings(config)
    try:
        vectors = []
        for i in range(0, len(chunks), 32):
            batch = [p.text for p in chunks[i:i + 32]]
            vectors.extend(await embeddings.embed(batch, task_type="RETRIEVAL_DOCUMENT") if config.ai_provider == "google" else await embeddings.embed(batch))
    finally:
        await embeddings.client.aclose()
    store = SQLiteVectorStore(config.db_path)
    store.replace(list(zip(chunks, vectors)), config.embedding_model)
    print(f"Indexed {len(chunks)} chunks from {len({p.article for p in chunks})} articles; source={url}; retrieved={date.today()}; model={config.embedding_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--file", default=DEFAULT_FILE if Path(DEFAULT_FILE).exists() else None, help="Verified official .txt or .html source; defaults to bundled clean official 2026 text")
    args = parser.parse_args()
    asyncio.run(ingest(args.url, Settings(), args.file))
