"""Hybrid (lexical + vector) retrieval over any number of collections and content kinds.

All searches for all query variants, collections and kinds go out in **one** Elasticsearch round
trip, and all query vectors in one embedding call per embedding model - the dominant cost of a
naive implementation is serial network latency, not compute.
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.errors import InvalidRequestError
from src.core.kinds import parse_kinds
from src.core.specs import CollectionSpec, RagConfig
from src.core.types import (
    CONTENT_KINDS,
    ContentKind,
    RawHit,
    RetrievalScope,
    RetrievedDocument,
    SearchRequest,
)
from src.models.registry import ModelRegistry
from src.ports.indexing import Searcher
from src.ports.retrieval import Reranker

if TYPE_CHECKING:
    from src.core.config import Settings

RRF_K = 60
CROSS_REFERENCE_DISCOUNT = 0.7
MAX_SIBLINGS, MAX_ADJACENT = 5, 3
MAX_CHUNKS_PER_PAGE = 2


@dataclass(frozen=True, slots=True)
class _Target:
    collection: CollectionSpec
    kind: ContentKind
    index: str


class HybridRetriever:
    def __init__(
        self,
        *,
        searcher: Searcher,
        models: ModelRegistry,
        config: RagConfig,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self._searcher = searcher
        self._models = models
        self._config = config
        self._reranker = reranker
        self._alpha = settings.hybrid_alpha
        self._search_size = settings.retriever_top_k * 3
        self._final_k = settings.rerank_top_k
        self._cross_references = settings.enable_cross_references

    # --- scope -----------------------------------------------------------------------------
    def resolve_scope(
        self,
        collections: Sequence[str] | None,
        kinds: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
    ) -> RetrievalScope:
        names = tuple(dict.fromkeys(collections)) if collections else (self._config.default_collection,)
        for name in names:
            self._config.collection(name)  # raises NotFoundError for unknown names
        requested = parse_kinds(kinds)
        selected: tuple[ContentKind, ...] = requested or CONTENT_KINDS
        for name in names:
            unsupported = [k for k in (requested or ()) if k not in self._config.collection(name).kinds]
            if unsupported:
                raise InvalidRequestError(f"collection '{name}' does not index {unsupported}")
        return RetrievalScope(collections=names, kinds=selected, sources=tuple(sources or ()))

    def _targets(self, scope: RetrievalScope) -> list[_Target]:
        targets = []
        for name in scope.collections:
            spec = self._config.collection(name)
            for kind in scope.kinds:
                if kind in spec.kinds:
                    targets.append(_Target(spec, kind, spec.index_name(kind)))
        return targets

    # --- retrieval -------------------------------------------------------------------------
    async def retrieve_many(
        self, queries: Sequence[str], scope: RetrievalScope
    ) -> list[list[RetrievedDocument]]:
        targets = self._targets(scope)
        if not targets:
            raise InvalidRequestError("the selected collections index none of the requested content kinds")
        vectors = await self._embed(queries, targets)
        requests = [
            SearchRequest(
                index=target.index,
                text=query,
                vector=tuple(vectors[target.collection.embedding_model][i]),
                size=self._search_size,
                sources=scope.sources,
            )
            for i, query in enumerate(queries)
            for target in targets
        ]
        results = await self._searcher.search(requests)

        per_query: list[list[RetrievedDocument]] = [[] for _ in queries]
        for position, result in enumerate(results):
            query_index, target = divmod(position, len(targets))
            per_query[query_index].extend(self._fuse(targets[target], result.lexical, result.vector))

        if self._cross_references:
            related = await self._related(per_query, targets)
            for query_index, docs in enumerate(related):
                per_query[query_index].extend(docs)

        return [self._diversify(self._reranker.rerank(docs, queries[i])) for i, docs in enumerate(per_query)]

    async def _embed(self, queries: Sequence[str], targets: list[_Target]) -> dict[str, list[list[float]]]:
        names = sorted({t.collection.embedding_model for t in targets})
        vectors = await asyncio.gather(
            *(self._models.embedder(n).embed_queries(list(queries)) for n in names)
        )
        return dict(zip(names, vectors, strict=True))

    def _fuse(self, target: _Target, lexical: list[RawHit], vector: list[RawHit]) -> list[RetrievedDocument]:
        """Reciprocal Rank Fusion of the two ranked lists, weighted by ``alpha``."""
        scored: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(lexical, start=1):
            scored.setdefault(hit.id, {"hit": hit, "lexical": 0.0, "vector": 0.0})["lexical"] = 1.0 / (
                RRF_K + rank
            )
        for rank, hit in enumerate(vector, start=1):
            scored.setdefault(hit.id, {"hit": hit, "lexical": 0.0, "vector": 0.0})["vector"] = 1.0 / (
                RRF_K + rank
            )
        documents = []
        for entry in scored.values():
            score = self._alpha * entry["lexical"] + (1 - self._alpha) * entry["vector"]
            doc = self._to_document(entry["hit"], target, score)
            doc.metadata["score_breakdown"] = {"bm25_rrf": entry["lexical"], "knn_rrf": entry["vector"]}
            documents.append(doc)
        return documents

    @staticmethod
    def _to_document(hit: RawHit, target: _Target, score: float) -> RetrievedDocument:
        src = hit.source
        metadata = {
            **(src.get("metadata") or {}),
            "language": src.get("language"),
            "keywords": src.get("keywords") or [],
            "source": src.get("source"),
            "page": src.get("page"),
            "chunk_id": src.get("chunk_id"),
            "document_id": src.get("document_id"),
            "sibling_chunk_ids": src.get("sibling_chunk_ids") or [],
            "adjacent_chunk_ids": src.get("adjacent_chunk_ids") or [],
            "content_type": src.get("content_type"),
            "has_table_on_page": src.get("has_table_on_page", False),
            "has_image_on_page": src.get("has_image_on_page", False),
            "collection": target.collection.name,
        }
        return RetrievedDocument(
            content=src.get("content", ""),
            metadata=metadata,
            score=score,
            collection=target.collection.name,
            kind=target.kind,
            index=hit.index,
        )

    async def _related(
        self, per_query: list[list[RetrievedDocument]], targets: list[_Target]
    ) -> list[list[RetrievedDocument]]:
        """Pull in neighbouring chunks (same page, adjacent pages) of what was found.

        A neighbour scores a fixed fraction of the best hit that points at it, so context
        supports - and never outranks - direct matches.
        """
        by_index = {t.index: t for t in targets}
        indices_of: dict[str, list[str]] = defaultdict(list)  # only kinds inside the requested scope
        for target in targets:
            indices_of[target.collection.name].append(target.index)
        wanted: dict[str, dict[str, float]] = defaultdict(dict)  # collection -> chunk_id -> parent score
        per_query_ids: list[list[tuple[str, str, float]]] = []
        for docs in per_query:
            have = {d.metadata.get("chunk_id") for d in docs}
            siblings: dict[tuple[str, str], float] = {}
            adjacent: dict[tuple[str, str], float] = {}
            for doc in docs:
                for chunk_id in doc.metadata.get("sibling_chunk_ids", []):
                    if chunk_id not in have:
                        key = (doc.collection, chunk_id)
                        siblings[key] = max(siblings.get(key, 0.0), doc.score)
                for chunk_id in doc.metadata.get("adjacent_chunk_ids", []):
                    if chunk_id not in have:
                        key = (doc.collection, chunk_id)
                        adjacent[key] = max(adjacent.get(key, 0.0), doc.score)
            chosen = [
                (*key, score)
                for table, limit in ((siblings, MAX_SIBLINGS), (adjacent, MAX_ADJACENT))
                for key, score in sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
            ]
            per_query_ids.append(chosen)
            for collection, chunk_id, score in chosen:
                wanted[collection][chunk_id] = max(wanted[collection].get(chunk_id, 0.0), score)

        async def fetch(collection: str) -> tuple[str, list[RawHit]]:
            return collection, await self._searcher.fetch(indices_of[collection], sorted(wanted[collection]))

        fetched = dict(await asyncio.gather(*(fetch(c) for c in wanted))) if wanted else {}
        lookup = {(c, h.source.get("chunk_id")): h for c, hits in fetched.items() for h in hits}
        out: list[list[RetrievedDocument]] = []
        for chosen in per_query_ids:
            docs = []
            for collection, chunk_id, parent_score in chosen:
                hit = lookup.get((collection, chunk_id))
                if hit is None:
                    continue
                doc = self._to_document(hit, by_index[hit.index], parent_score * CROSS_REFERENCE_DISCOUNT)
                doc.metadata["is_cross_reference"] = True
                docs.append(doc)
            out.append(docs)
        return out

    def _diversify(self, docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
        """Best ``final_k`` by score, taking at most ``MAX_CHUNKS_PER_PAGE`` chunks per page
        first and only then filling remaining slots from the overflow."""
        ranked = sorted(docs, key=lambda d: d.final_score, reverse=True)
        if len(ranked) <= self._final_k:
            return ranked
        per_page: Counter[tuple[str, str | None, int | None]] = Counter()
        chosen: list[RetrievedDocument] = []
        overflow: list[RetrievedDocument] = []
        for doc in ranked:
            key = (doc.collection, doc.metadata.get("source"), doc.metadata.get("page"))
            if per_page[key] < MAX_CHUNKS_PER_PAGE and len(chosen) < self._final_k:
                per_page[key] += 1
                chosen.append(doc)
            else:
                overflow.append(doc)
        chosen.extend(overflow[: self._final_k - len(chosen)])
        return sorted(chosen, key=lambda d: d.final_score, reverse=True)
