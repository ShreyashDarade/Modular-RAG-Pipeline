from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.core.errors import InvalidRequestError
from src.core.types import CONTENT_KINDS, RetrievalResult, RetrievalScope, RetrievedDocument
from src.ports.retrieval import QueryExpander
from src.retrieval.hybrid import HybridRetriever
from src.runtime.cache import CachedCall, CorpusVersion

if TYPE_CHECKING:
    from src.core.config import Settings


def _dump(result: RetrievalResult) -> bytes:
    return json.dumps(
        {
            "query": result.query,
            "expanded": result.expanded_queries,
            "documents": [
                {
                    "content": d.content,
                    "metadata": d.metadata,
                    "score": d.score,
                    "rerank_score": d.rerank_score,
                    "collection": d.collection,
                    "kind": d.kind,
                    "index": d.index,
                }
                for d in result.documents
            ],
        },
        ensure_ascii=False,
    ).encode()


def _load(raw: bytes) -> RetrievalResult:
    data = json.loads(raw)
    return RetrievalResult(
        query=data["query"],
        expanded_queries=data["expanded"],
        documents=[RetrievedDocument(**d) for d in data["documents"]],
    )


class RetrievalPipeline:
    """expand -> hybrid search (all variants, collections, kinds at once) -> merge.

    Results are cached per (query, scope, corpus version); any ingest or delete bumps the corpus
    version, which retires every cached result on every replica at once.
    """

    def __init__(
        self,
        *,
        expander: QueryExpander,
        retriever: HybridRetriever,
        cache: CachedCall,
        corpus: CorpusVersion,
        settings: Settings,
        fingerprint: str = "",
    ) -> None:
        self._expander = expander
        self._retriever = retriever
        self._cache = cache
        self._corpus = corpus
        self._top_k = settings.retriever_top_k
        self._ttl = settings.cache_ttl_seconds
        self._max_query_chars = settings.max_query_chars
        # A shared cache outlives deploys: results computed under other retrieval settings must not be served.
        self._fingerprint = hashlib.sha256(
            json.dumps(
                [
                    settings.hybrid_alpha,
                    settings.retriever_top_k,
                    settings.rerank_top_k,
                    settings.enable_cross_references,
                    fingerprint,
                ],
            ).encode()
        ).hexdigest()[:12]

    def scope(
        self,
        collections: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
    ) -> RetrievalScope:
        return self._retriever.resolve_scope(collections, kinds, sources)

    async def retrieve(self, query: str, scope: RetrievalScope) -> RetrievalResult:
        query = query.strip()
        if not query:
            raise InvalidRequestError("query must not be empty")
        if len(query) > self._max_query_chars:
            raise InvalidRequestError(f"query is longer than {self._max_query_chars} characters")
        version = await self._corpus.current()
        fingerprint = json.dumps(
            [
                query,
                sorted(scope.collections),
                sorted(scope.kinds),
                sorted(scope.sources),
                version,
                self._fingerprint,
            ],
            ensure_ascii=False,
        )
        key = f"retrieve:{hashlib.sha256(fingerprint.encode()).hexdigest()}"
        raw = await self._cache.get_or_compute(key, self._ttl, lambda: self._compute(query, scope))
        return _load(raw)

    async def _compute(self, query: str, scope: RetrievalScope) -> bytes:
        variants = await self._expander.expand(query)
        per_variant = await self._retriever.retrieve_many(variants, scope)
        best: dict[tuple[str, str | None, int | None, str], RetrievedDocument] = {}
        for docs in per_variant:
            for doc in docs:
                key = (
                    doc.collection,
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.content[:100],
                )
                current = best.get(key)
                if current is None or doc.final_score > current.final_score:
                    best[key] = doc
        ranked = sorted(best.values(), key=lambda d: d.final_score, reverse=True)

        # One document of every content kind first, so image-only answers are not drowned out by
        # denser PDF text; then fill the remaining slots by score.
        selected: list[RetrievedDocument] = []
        for kind in CONTENT_KINDS:
            first = next((d for d in ranked if d.kind == kind), None)
            if first is not None and len(selected) < self._top_k:
                selected.append(first)
        chosen = {id(d) for d in selected}
        selected.extend([d for d in ranked if id(d) not in chosen][: self._top_k - len(selected)])
        selected.sort(key=lambda d: d.final_score, reverse=True)
        return _dump(RetrievalResult(query=query, expanded_queries=variants, documents=selected))
