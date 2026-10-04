from __future__ import annotations

from typing import Protocol

from src.core.types import RetrievedDocument


class QueryExpander(Protocol):
    async def expand(self, query: str) -> list[str]:
        """Return query variants with the original query first."""


class Reranker(Protocol):
    def rerank(self, documents: list[RetrievedDocument], query: str) -> list[RetrievedDocument]:
        """Set ``rerank_score`` on each document and return them (order not significant)."""
