from __future__ import annotations

from src.core.registry import Registries
from src.core.types import RetrievedDocument

_TABLE_TERMS = ("table", "data", "numbers", "statistics")


class IdentityReranker:
    def rerank(self, documents: list[RetrievedDocument], query: str) -> list[RetrievedDocument]:
        for doc in documents:
            doc.rerank_score = doc.score
        return documents


class HeuristicReranker:
    """Multiplies the fused score by cheap relevance signals: keyword overlap, content kind,
    page context, exact phrase and an early-page bonus."""

    def rerank(self, documents: list[RetrievedDocument], query: str) -> list[RetrievedDocument]:
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
        return documents


def register_builtin_rerankers(registries: Registries) -> None:
    registries.rerankers.register("heuristic", lambda settings: HeuristicReranker())
    registries.rerankers.register("identity", lambda settings: IdentityReranker())
