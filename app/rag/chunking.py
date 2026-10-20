import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    document: str
    article: str
    section: str
    text: str
    source: str
    language: str = "ru"
    version: str = "2026"
    retrieved_at: str = ""


ARTICLE = re.compile(r"(?im)^\s*(?:Article|Статья|Бап)\s+(\d+[\-\d]*)\s*[.:]?\s*")
SECTION = re.compile(r"(?m)^\s*(\d+)\.\s+")
CHAPTER = re.compile(r"(?im)^\s*(?:Раздел|Section|Бөлім)\s+([IVXLCDM\d]+)[^.\n]*")


def chunk_articles(text: str, source: str, max_chars: int = 2000) -> list[Passage]:
    matches = list(ARTICLE.finditer(text))
    result: list[Passage] = []
    for i, match in enumerate(matches):
        body = text[match.start(): matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
        preceding = list(CHAPTER.finditer(text, 0, match.start()))
        chapter = preceding[-1].group(0).strip() if preceding else ""
        if len(body) <= max_chars:
            result.append(Passage("Constitution of the Republic of Kazakhstan", match.group(1), chapter, body, source))
            continue
        parts = list(SECTION.finditer(body))
        if not parts or parts[0].start() > 0:
            parts.insert(0, re.match(r"", body))
        for j, part in enumerate(parts):
            fragment = body[part.start():parts[j + 1].start() if j + 1 < len(parts) else len(body)].strip()
            if not fragment:
                continue
            # A numbered paragraph can exceed the limit; split only at sentence boundaries.
            sentences = re.split(r"(?<=[.!?])\s+", fragment)
            batch = ""
            for sentence in sentences:
                if len(batch) + len(sentence) > max_chars and batch:
                    result.append(Passage("Constitution of the Republic of Kazakhstan", match.group(1), chapter, batch.strip(), source))
                    batch = ""
                batch += sentence + " "
            if batch.strip():
                result.append(Passage("Constitution of the Republic of Kazakhstan", match.group(1), chapter, batch.strip(), source))
    return result
