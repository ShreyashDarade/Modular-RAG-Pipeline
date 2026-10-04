"""In-memory implementations of the ports, for testing orchestrators without services."""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.core.types import RawHit, SearchRequest, SearchResult


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        (math.sqrt(sum(x * x for x in a)) or 1) * (math.sqrt(sum(y * y for y in b)) or 1)
    )


class MemorySearcher:
    """Implements ``Searcher`` over dicts: word-overlap for the lexical list, cosine for vectors."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, dict]] = {}  # index -> chunk_id -> source
        self.search_calls: list[int] = []
        self.fetch_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def add(self, index: str, source: dict) -> None:
        self.docs.setdefault(index, {})[source["chunk_id"]] = source

    async def search(self, requests: Sequence[SearchRequest]) -> list[SearchResult]:
        self.search_calls.append(len(requests))
        results = []
        for request in requests:
            docs = list(self.docs.get(request.index, {}).values())
            if request.sources:
                docs = [d for d in docs if d["source"] in request.sources]
            result = SearchResult()
            if request.text is not None:
                words = set(request.text.lower().split())
                scored = [(len(words & set(d["content"].lower().split())), d) for d in docs]
                result.lexical = [
                    self._hit(request.index, d, float(s))
                    for s, d in sorted(scored, key=lambda t: -t[0])
                    if s > 0
                ][: request.size]
            if request.vector is not None:
                scored = [(cosine(request.vector, d["content_vector"]), d) for d in docs]
                result.vector = [
                    self._hit(request.index, d, s) for s, d in sorted(scored, key=lambda t: -t[0])
                ][: request.size]
            results.append(result)
        return results

    async def fetch(self, indices: Sequence[str], chunk_ids: Sequence[str]) -> list[RawHit]:
        self.fetch_calls.append((tuple(indices), tuple(chunk_ids)))
        return [
            self._hit(index, self.docs[index][cid], 0.0)
            for index in indices
            for cid in chunk_ids
            if cid in self.docs.get(index, {})
        ]

    async def sample(self, index: str, size: int, seed: int) -> list[RawHit]:
        import random

        docs = sorted(self.docs.get(index, {}).values(), key=lambda d: d["chunk_id"])
        picked = random.Random(seed).sample(docs, min(size, len(docs)))
        return [self._hit(index, d, 0.0) for d in picked]

    @staticmethod
    def _hit(index: str, source: dict, score: float) -> RawHit:
        return RawHit(
            id=source["chunk_id"],
            index=index,
            score=score,
            source={k: v for k, v in source.items() if k != "content_vector"},
        )
