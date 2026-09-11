"""TF-IDF retrieval over the local networking knowledge base with performance optimizations."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


class KnowledgeBase:
    """TF-IDF indexed knowledge base for networking topics."""

    def __init__(self, entries: List[Dict[str, Any]]):
        self.entries = entries
        self._texts: List[str] = []
        for e in entries:
            keywords = e.get("keywords", [])
            if isinstance(keywords, list):
                keywords_str = " ".join(keywords)
            else:
                keywords_str = str(keywords)
            
            topic = str(e.get("topic", ""))
            answer = str(e.get("answer", ""))
            self._texts.append(f"{topic} {keywords_str} {answer}")

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = (
            self._vectorizer.fit_transform(self._texts) if self._texts else None
        )

    @functools.lru_cache(maxsize=128)
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for top_k relevant entries using cosine similarity over TF-IDF vectors.
        LRU cache is used to optimize repeat queries.
        """
        if not self.entries or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        results: List[Dict[str, Any]] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            entry = dict(self.entries[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results

    def clear_cache(self) -> None:
        """Clear the query search cache."""
        self.search.cache_clear()


def load_knowledge_base(path: Optional[Path] = None) -> KnowledgeBase:
    """Load the knowledge base from a JSON file."""
    kb_path = path or KB_PATH
    with kb_path.open(encoding="utf-8") as f:
        entries = json.load(f)
    return KnowledgeBase(entries)
