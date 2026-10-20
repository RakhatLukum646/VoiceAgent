"""Fetch a reviewed small official Gold receiving corpus; never site navigation."""
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup
import httpx

PATHS = ["receive/general/q27", "receive/branches/q43", "conditions/q12371", "receive/app/q13478", "receive/general/q12508", "receive/general/q14620", "receive/general/q13483", "receive/general/q17279",
         "receive/general/q17278", "receive/general/q45", "receive/general/q19417", "receive/general/q15458", "receive/general/q46", "receive/general/q15157", "receive/general/q1638", "receive/general/q81"]


async def fetch():
    directory = Path("data/official_snapshots")
    directory.mkdir(exist_ok=True)
    documents = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for path in PATHS:
            url = "https://guide.kaspi.kz/client/ru/gold/" + path
            response = await client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            h1 = soup.find("h1")
            block = h1.find_parent("div", class_="ds-expand") if h1 else None
            content = block.select_one(".question-answer__content") if block else None
            if not content:
                raise ValueError("Official structure changed; publication not attempted")
            full_text = " ".join(content.get_text(" ", strip=True).replace("\xa0", " ").split())
            text = full_text
            inclusion = "complete answer; attached conditions preserved"
            if path.endswith("q13478"):
                # This source's later usage-benefits list mentions other products
                # and image-only text. Keep only literal application/collection
                # paragraphs; exact offsets are persisted for provenance.
                paragraphs = [" ".join(p.get_text(" ", strip=True).replace("\xa0", " ").split()) for p in content.find_all("p")]
                picked = [p for p in paragraphs if any(term in p for term in ("оформ", "Оформ", "Kartomat", "получить карту", "забрать")) and not any(term in p for term in ("Kaspi Red", "кредит", "Кредит"))]
                text = "\n".join(picked)
                if not text or "Kartomat" not in text:
                    raise ValueError("App excerpt requires review")
                inclusion = "literal application/collection paragraphs only; omit usage benefits and image-only instructions"
            updated = block.select_one(".question-answer__updated")
            document = {"source": url, "title": h1.get_text(" ", strip=True), "text": text, "full_body": full_text,
                "updated": updated.get_text(" ", strip=True) if updated else "", "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "raw_sha256": hashlib.sha256(response.content).hexdigest(), "body_sha256": hashlib.sha256(full_text.encode()).hexdigest(),
                "extraction_version": "kaspi-answer-v2", "inclusion_rule": inclusion,
                "excerpt_offsets": [[full_text.index(part), full_text.index(part) + len(part)] for part in text.split("\n")]}
            key = path.rsplit("/", 1)[-1]
            (directory / (key + ".html")).write_text(response.text, encoding="utf-8")
            (directory / (key + ".json")).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
            documents.append(document)
            print(key + " " + str(len(text)) + " chars; " + document["title"])
    Path("data/official_corpus.json").write_text(json.dumps({"dataset_kind": "official", "service": "gold-receive", "documents": documents}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(fetch())
