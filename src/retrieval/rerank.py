"""Built-in rerankers that need no model: ``identity`` (keep the fused order) and ``heuristic``
(cheap lexical signals). Model-based rerankers live in :mod:`src.models.rerankers`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.core.registry import Registries
from src.core.types import RetrievedDocument

if TYPE_CHECKING:
    from src.core.config import Settings
    from src.core.specs import RerankerSpec

_TABLE_TERMS = ("table", "data", "numbers", "statistics")


class _NoModel:
    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


class IdentityReranker(_NoModel):
    async def rerank(self, documents: Sequence[RetrievedDocument], query: str) -> list[RetrievedDocument]:
        for doc in documents:
            doc.rerank_score = doc.score
        return list(documents)


class HeuristicReranker(_NoModel):
    """Multiplies the fused score by cheap relevance signals: keyword overlap, content kind,
    page context, exact phrase and an early-page bonus. These are hand-set weights, not learned
    relevance - use a model-based reranker where answer quality matters, and measure the
    difference with ``rag eval``."""

    async def rerank(self, documents: Sequence[RetrievedDocument], query: str) -> list[RetrievedDocument]:
        lowered = query.lower()
        terms = set(lowered.split())
        for doc in documents:
            meta = doc.metadata
            boost = 1.0
            overlap = len(terms & {str(k).lower() for k in meta.get("keywords") or []})
            boost += 0.1 * overlap
            if doc.kind == "text":
                boost += 0.05
            elif doc.kind == "table" and any(term in lowered for term in _TABLE_TERMS):
                boost += 0.15
            if meta.get("has_table_on_page"):
                boost += 0.03
            if meta.get("has_image_on_page"):
                boost += 0.02
            if lowered in doc.content.lower():
                boost += 0.2
            page = meta.get("page") or 1
            if page <= 5:
                boost += 0.05 * (1 - page / 10)
            doc.rerank_score = doc.score * boost
        return list(documents)


def register_builtin_rerankers(registries: Registries) -> None:
    def heuristic(name: str, spec: RerankerSpec, settings: Settings) -> HeuristicReranker:
        return HeuristicReranker()

    def identity(name: str, spec: RerankerSpec, settings: Settings) -> IdentityReranker:
        return IdentityReranker()

    registries.rerankers.register("heuristic", heuristic)
    registries.rerankers.register("identity", identity)
