from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from langchain_core.documents import Document
from elasticsearch import ApiError

from src.core.config import settings
from src.core.logger import logger
from src.services.elastic import ElasticClient
from src.services.embedding import EmbeddingService


@dataclass
class RetrievedDocument:
    document: Document
    score: float
    source_index: str


class HybridRetriever:
    def __init__(self, alpha: float | None = None, top_k: int | None = None) -> None:
        self.alpha = alpha or settings.hybrid_alpha
        self.top_k = top_k or settings.retriever_top_k
        self.client = ElasticClient()
        self.embedding = EmbeddingService()
        self.indexes = [
            settings.es_index_text,
            settings.es_index_tables,
            settings.es_index_images,
        ]

    def retrieve(self, query: str) -> List[RetrievedDocument]:
        query_vector = self.embedding.embed_query(query)
        combined: List[RetrievedDocument] = []
        for index in self.indexes:
            bm25_hits = self._bm25_search(index, query)
            knn_hits = self._knn_search(index, query_vector)
            combined.extend(self._fuse(index, bm25_hits, knn_hits))
        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[: self.top_k]

    def _bm25_search(self, index: str, query: str) -> List[Dict]:
        body = {
            "size": self.top_k * 3,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content^3", "keywords", "metadata.source"],
                    "fuzziness": "AUTO",
                }
            },
        }
        response = self.client.search(index, body)
        return response.get("hits", {}).get("hits", [])

    def _knn_search(self, index: str, vector: List[float]) -> List[Dict]:
        body = {
            "size": self.top_k * 3,
            "query": {"match_all": {}},
            "knn": {
                "field": "content_vector",
                "query_vector": vector,
                "k": self.top_k * 3,
                "num_candidates": self.top_k * 6,
            },
        }
        try:
            response = self.client.search(index, body)
            return response.get("hits", {}).get("hits", [])
        except ApiError as exc:
            logger.warning("kNN search unsupported on index %s: %s", index, exc)
            return []

    def _fuse(self, index: str, bm25_hits: List[Dict], knn_hits: List[Dict]) -> List[RetrievedDocument]:
        scores: Dict[str, Dict[str, float]] = {}
        max_bm25 = max((hit.get("_score", 0.0) for hit in bm25_hits), default=1.0)
        max_knn = max((hit.get("_score", 0.0) for hit in knn_hits), default=1.0)

        for hit in bm25_hits:
            doc_id = hit.get("_id")
            normalized = (hit.get("_score", 0.0) / max_bm25) if max_bm25 else 0.0
            scores.setdefault(doc_id, {})["bm25"] = normalized
        for hit in knn_hits:
            doc_id = hit.get("_id")
            normalized = (hit.get("_score", 0.0) / max_knn) if max_knn else 0.0
            scores.setdefault(doc_id, {})["knn"] = normalized

        fused: List[RetrievedDocument] = []
        for doc_id, parts in scores.items():
            bm25 = parts.get("bm25", 0.0)
            knn = parts.get("knn", 0.0)
            score = self.alpha * bm25 + (1 - self.alpha) * knn
            source_hit = next(
                (hit for hit in bm25_hits + knn_hits if hit.get("_id") == doc_id),
                None,
            )
            if not source_hit:
                continue
            source = source_hit.get("_source", {})
            document = Document(page_content=source.get("content", ""), metadata=source.get("metadata", {}))
            document.metadata.update(
                {
                    "language": source.get("language"),
                    "keywords": source.get("keywords", []),
                    "source": source.get("source"),
                    "score_breakdown": parts,
                }
            )
            fused.append(RetrievedDocument(document=document, score=score, source_index=index))
        return fused


__all__ = ["HybridRetriever", "RetrievedDocument"]
