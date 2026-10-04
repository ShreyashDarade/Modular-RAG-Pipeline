from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.core.types import RetrievedDocument


class QueryExpander(Protocol):
    async def expand(self, query: str) -> list[str]:
        """Return query variants with the original query first."""


class Reranker(Protocol):
    """Orders retrieval candidates by relevance to the query.

    Runs once per request on the merged candidates of all query variants. A failure raises a typed
    error; the pipeline never quietly returns the un-reranked candidates instead.
    """

    async def start(self) -> None:
        """Load whatever is expensive (model weights). Called once at start-up."""

    async def rerank(self, documents: Sequence[RetrievedDocument], query: str) -> list[RetrievedDocument]:
        """Set ``rerank_score`` (higher is more relevant) on every document and return them."""

    async def close(self) -> None: ...
