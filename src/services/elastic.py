from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List

from elasticsearch import ApiError, Elasticsearch

from src.core.config import settings
from src.core.logger import logger


class ElasticClient:
    def __init__(self) -> None:
        auth = None
        if settings.es_username and settings.es_password:
            auth = (settings.es_username, settings.es_password)
        self.client = Elasticsearch(settings.es_host, basic_auth=auth)

    def ensure_index(self, index_name: str, dims: int) -> None:
        if self.client.indices.exists(index=index_name):
            return
        logger.info("Creating index %s with dimension %s", index_name, dims)
        mappings = {
            "properties": {
                "content": {"type": "text"},
                "content_vector": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "keywords": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": True},
                "language": {"type": "keyword"},
                "source": {"type": "keyword"},
                "page": {"type": "integer"},
                "created_at": {"type": "date"},
            }
        }
        settings_body = {"index": {"knn": True}}
        try:
            self.client.indices.create(index=index_name, mappings=mappings, settings=settings_body)
        except ApiError as exc:
            message = str(exc)
            if "unknown setting [index.knn]" in message or "illegal_argument_exception" in message:
                logger.warning("Index setting 'index.knn' not supported by cluster; retrying without kNN flag.")
                fallback_settings = {"number_of_shards": 1, "number_of_replicas": 0}
                self.client.indices.create(index=index_name, mappings=mappings, settings=fallback_settings)
            else:
                raise

    def bulk_index(self, index_name: str, docs: Iterable[Dict[str, Any]]) -> None:
        operations: List[Dict[str, Any]] = []
        for doc in docs:
            operations.append({"index": {"_index": index_name}})
            operations.append(doc)
        if not operations:
            return
        response = self.client.bulk(operations=operations, refresh=True)
        if response.get("errors"):
            logger.error("Bulk index encountered errors: %s", response)
        else:
            logger.info("Indexed %s documents into %s", len(operations) // 2, index_name)

    def search(self, index_name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.search(index=index_name, body=body)

    def count(self, index_name: str, query: Dict[str, Any]) -> int:
        try:
            response = self.client.count(index=index_name, body={"query": query})
            return int(response.get("count", 0))
        except ApiError as exc:
            logger.warning("Failed to count documents in %s: %s", index_name, exc)
            return 0


class ElasticIndexer:
    def __init__(self, embedding_dim: int) -> None:
        self.elastic = ElasticClient()
        self.embedding_dim = embedding_dim
        self.elastic.ensure_index(settings.es_index_text, embedding_dim)
        self.elastic.ensure_index(settings.es_index_tables, embedding_dim)
        self.elastic.ensure_index(settings.es_index_images, embedding_dim)
        self.index_names = [
            settings.es_index_text,
            settings.es_index_tables,
            settings.es_index_images,
        ]

    def index_text_documents(self, docs: List[Dict[str, Any]]) -> None:
        self.elastic.bulk_index(settings.es_index_text, docs)

    def index_table_documents(self, docs: List[Dict[str, Any]]) -> None:
        self.elastic.bulk_index(settings.es_index_tables, docs)

    def index_image_documents(self, docs: List[Dict[str, Any]]) -> None:
        self.elastic.bulk_index(settings.es_index_images, docs)

    def source_document_count(self, source: str) -> int:
        total = 0
        query = {"term": {"source": source}}
        for index_name in self.index_names:
            total += self.elastic.count(index_name, query)
        return total


__all__ = ["ElasticIndexer", "ElasticClient"]
