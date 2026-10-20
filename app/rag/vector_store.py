import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from .chunking import Passage


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Embedding dimension mismatch")
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def _tokens(value: str) -> list[str]:
    return re.findall(r"[\w]+", value.casefold())


class SQLiteVectorStore:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS passages (id INTEGER PRIMARY KEY, document TEXT, article TEXT, section TEXT, text TEXT, source TEXT, model TEXT, embedding TEXT, language TEXT, version TEXT, retrieved_at TEXT)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(passages)")}
            for column in ("language", "version", "retrieved_at"):
                if column not in columns:
                    db.execute(f"ALTER TABLE passages ADD COLUMN {column} TEXT")

    def _connect(self):
        return sqlite3.connect(self.path)

    def replace(self, rows: list[tuple[Passage, list[float]]], model: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM passages")
            db.executemany("INSERT INTO passages (document,article,section,text,source,model,embedding,language,version,retrieved_at) VALUES (?,?,?,?,?,?,?,?,?,?)", [(p.document,p.article,p.section,p.text,p.source,model,json.dumps(v),p.language,p.version,p.retrieved_at) for p,v in rows])

    def count(self) -> int:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]

    def count_for_model(self, model: str) -> int:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM passages WHERE model=?", (model,)).fetchone()[0]

    def _rows(self, model: str) -> list[dict]:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT id,document,article,section,text,source,embedding,language,version,retrieved_at FROM passages WHERE model=?", (model,)).fetchall()
        return [dict(row) for row in rows]

    def lookup_articles(self, articles: list[str], model: str, neighbors: bool = False) -> list[dict]:
        wanted = set(articles)
        if neighbors:
            for article in articles:
                if article.isdecimal():
                    number = int(article)
                    wanted.update((str(number - 1), str(number + 1)))
        return [self._hit(row, 0.0) for row in self._rows(model) if row["article"] in wanted]

    def all_passages(self, model: str) -> list[dict]:
        return [self._hit(row, 0.0) for row in self._rows(model)]

    @staticmethod
    def _hit(row: dict, score: float) -> dict:
        return {key: row[key] for key in ("id", "document", "article", "section", "text", "source", "language", "version", "retrieved_at")} | {"score": score}

    def search(self, vector: list[float], model: str, top_k: int, query: str = "") -> list[dict]:
        if not vector or any(not math.isfinite(x) for x in vector):
            raise ValueError("Invalid query embedding")
        rows = self._rows(model)
        if not rows:
            return []
        cosine = {row["id"]: _cosine(vector, json.loads(row["embedding"])) for row in rows}
        semantic = sorted(rows, key=lambda row: cosine[row["id"]], reverse=True)
        if not query:
            return [self._hit(row, cosine[row["id"]]) for row in semantic[:top_k]]
        terms = _tokens(query)
        documents = [_tokens(row["text"]) for row in rows]
        avg_length = sum(map(len, documents)) / len(documents) or 1
        frequency = Counter(term for doc in documents for term in set(doc))
        bm25 = {}
        for row, doc in zip(rows, documents):
            counts = Counter(doc)
            score = 0.0
            for term in set(terms):
                if counts[term]:
                    idf = math.log(1 + (len(rows) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                    score += idf * counts[term] * 2.2 / (counts[term] + 1.2 * (0.25 + 0.75 * len(doc) / avg_length))
            bm25[row["id"]] = score
        lexical = sorted(rows, key=lambda row: bm25[row["id"]], reverse=True)
        exact_articles = re.findall(r"(?:стать(?:я|и|ю|е|ей)|article)\s*(\d+)", query.casefold())
        fused = Counter()
        for rank, row in enumerate(semantic):
            fused[row["id"]] += 1 / (60 + rank + 1)
        for rank, row in enumerate(lexical):
            if bm25[row["id"]] > 0:
                fused[row["id"]] += 1 / (60 + rank + 1)
        if exact_articles:
            for row in rows:
                if row["article"] in exact_articles:
                    fused[row["id"]] += 1.0
                elif row["article"].isdecimal() and any(abs(int(row["article"]) - int(article)) == 1 for article in exact_articles):
                    fused[row["id"]] += 0.03
        ranked = sorted(rows, key=lambda row: fused[row["id"]], reverse=True)
        lexical_ids = {row["id"] for row in lexical[:top_k] if bm25[row["id"]] > 0}
        return [self._hit(row, cosine[row["id"]]) | {"rank_score": fused[row["id"]], "lexical_match": row["id"] in lexical_ids, "exact_article_match": row["article"] in exact_articles} for row in ranked[:top_k]]
