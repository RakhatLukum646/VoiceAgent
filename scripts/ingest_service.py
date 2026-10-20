"""Index only the answer body of one official Kaspi Guide page; preserve provenance."""
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import httpx
from bs4 import BeautifulSoup
from app.config import Settings
from app.providers import create_embeddings
from app.rag.chunking import Passage
from app.rag.vector_store import SQLiteVectorStore

SOURCE = 'https://guide.kaspi.kz/client/ru/gold/receive/general/q27'


def extract(raw: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(raw, 'html.parser')
    title = soup.find('h1')
    block = title.find_parent('div', class_='ds-expand') if title else None
    content = block.select_one('.question-answer__content') if block else None
    updated = block.select_one('.question-answer__updated') if block else None
    if not title or not content or 'Kaspi Gold' not in title.get_text():
        raise ValueError('Official page structure changed; index left untouched')
    text = ' '.join(content.get_text(' ', strip=True).replace('\xa0', ' ').split())
    if not all(word in text for word in ('Kartomat', 'мобильный телефон', 'именную')):
        raise ValueError('Incomplete answer body; index left untouched')
    return title.get_text(' ', strip=True), text, updated.get_text(' ', strip=True) if updated else ''


async def ingest():
    config = Settings()
    if 'constitution' in config.db_path.lower():
        raise ValueError('Use a separate service index; original corpus must be preserved')
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(SOURCE)
        response.raise_for_status()
    title, text, updated = extract(response.text)
    fetched = datetime.now(timezone.utc).isoformat()
    snapshot = title + '\n\n' + text + '\n'
    Path('data/kaspi_gold_source.html').write_text(response.text, encoding='utf-8')
    Path('data/kaspi_gold_clean.txt').write_text(snapshot, encoding='utf-8')
    metadata = {'source': SOURCE, 'title': title, 'retrieved_at': fetched, 'source_updated': updated, 'sha256': hashlib.sha256(snapshot.encode()).hexdigest(), 'scope': 'Оформление и получение Kaspi Gold', 'selector': 'h1 ancestor div.ds-expand / .question-answer__content'}
    Path('data/kaspi_gold_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    passage = Passage('Kaspi Gold: оформление и получение', 'q27', title, text, SOURCE, version=updated, retrieved_at=fetched)
    embeddings = create_embeddings(config)
    try:
        vectors = await embeddings.embed([title + '\n' + text], task_type='RETRIEVAL_DOCUMENT') if config.ai_provider == 'google' else await embeddings.embed([text])
    finally:
        await embeddings.client.aclose()
    SQLiteVectorStore(config.db_path).replace([(passage, vectors[0])], config.embedding_model)
    print('Indexed 1 official service passage; source=' + SOURCE)


if __name__ == '__main__':
    asyncio.run(ingest())
