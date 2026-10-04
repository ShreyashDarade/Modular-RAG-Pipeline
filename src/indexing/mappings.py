"""Index settings and mappings (Elasticsearch)."""

from __future__ import annotations

from typing import Any

#: Fields added after the first release; applied additively to indices that already exist.
ADDITIVE_FIELDS: dict[str, dict[str, Any]] = {
    "file_checksum": {"type": "keyword"},
    "kind": {"type": "keyword"},
}


def chunk_properties(dims: int, vector_index_type: str) -> dict[str, Any]:
    vector: dict[str, Any] = {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"}
    if vector_index_type != "auto":
        vector["index_options"] = {"type": vector_index_type}
    return {
        "content": {"type": "text", "analyzer": "standard"},
        "content_vector": vector,
        "keywords": {"type": "keyword"},
        "metadata": {
            "type": "object",
            # Everything else lives in _source for retrieval but is not indexed: unbounded
            # user-supplied keys cannot explode the mapping.
            "dynamic": False,
            "properties": {"table_summary": {"type": "text"}, "source": {"type": "text"}},
        },
        "language": {"type": "keyword"},
        "source": {"type": "keyword"},
        "page": {"type": "integer"},
        "chunk_id": {"type": "keyword"},
        "document_id": {"type": "keyword"},
        "sibling_chunk_ids": {"type": "keyword"},
        "adjacent_chunk_ids": {"type": "keyword"},
        "content_type": {"type": "keyword"},
        "has_table_on_page": {"type": "boolean"},
        "has_image_on_page": {"type": "boolean"},
        "created_at": {"type": "date"},
        **ADDITIVE_FIELDS,
    }


REGISTRY_PROPERTIES: dict[str, Any] = {
    "collection": {"type": "keyword"},
    "source": {"type": "keyword"},
    "file_checksum": {"type": "keyword"},
    "document_id": {"type": "keyword"},
    "status": {"type": "keyword"},
    "parser": {"type": "keyword"},
    "kinds": {"type": "keyword"},
    "text_chunks": {"type": "integer"},
    "table_chunks": {"type": "integer"},
    "image_chunks": {"type": "integer"},
    "total_pages": {"type": "integer"},
    "completed_at": {"type": "date"},
}


def index_body(
    dims: int | None,
    *,
    shards: int,
    replicas: int,
    refresh_interval: str,
    vector_index_type: str,
) -> dict[str, Any]:
    properties = REGISTRY_PROPERTIES if dims is None else chunk_properties(dims, vector_index_type)
    return {
        "settings": {
            "number_of_shards": shards,
            "number_of_replicas": replicas,
            "refresh_interval": refresh_interval,
        },
        "mappings": {"dynamic": False, "properties": properties},
    }
