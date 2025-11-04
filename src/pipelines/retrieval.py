from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.services.query_expansion import QueryExpansionService
from src.services.retrieval import HybridRetriever, RetrievedDocument


@dataclass
class RetrievalResult:
    query: str
    expanded_queries: List[str]
    documents: List[RetrievedDocument]


class RetrievalPipeline:
    def __init__(self) -> None:
        self.expander = QueryExpansionService()
        self.retriever = HybridRetriever()

    def retrieve(self, query: str) -> RetrievalResult:
        expansion = self.expander.expand(query)
        seen = {}
        for variant in expansion.expanded:
            hits = self.retriever.retrieve(variant)
            for hit in hits:
                doc_hash = (
                    hit.document.metadata.get("source"),
                    hit.document.metadata.get("page"),
                    hit.document.page_content[:100],
                )
                existing = seen.get(doc_hash)
                if existing is None or hit.score > existing.score:
                    seen[doc_hash] = hit
        aggregated = list(seen.values())
        aggregated.sort(key=lambda x: x.score, reverse=True)

        # Ensure at least one document is surfaced from each backing index so image-only
        # answers are not overshadowed by denser PDF text content.
        selected: List[RetrievedDocument] = []
        covered_indexes = set()

        for hit in aggregated:
            if hit.source_index not in covered_indexes:
                selected.append(hit)
                covered_indexes.add(hit.source_index)
            if len(selected) >= self.retriever.top_k:
                break

        if len(selected) < self.retriever.top_k:
            for hit in aggregated:
                if hit in selected:
                    continue
                selected.append(hit)
                if len(selected) >= self.retriever.top_k:
                    break

        return RetrievalResult(query=query, expanded_queries=expansion.expanded, documents=selected)


__all__ = ["RetrievalPipeline", "RetrievalResult"]
