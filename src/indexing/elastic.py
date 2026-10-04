"""Elasticsearch adapter: one pooled async connection shared by a writer, a searcher and the
document registry (each implements one small port).

Every call goes through :meth:`ElasticConnection.request`, which bounds concurrency, records
latency, and translates client exceptions into the domain's ``UpstreamError`` family. Nothing is
swallowed: a failed search is a ``SearchError``, a partly failed bulk is an ``IndexingError``.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from elasticsearch import ApiError, AsyncElasticsearch, NotFoundError, TransportError
from elasticsearch.helpers import async_streaming_bulk

from src.core.errors import ConfigError, IndexingError, RagError, SearchError, UpstreamError
from src.core.logger import logger
from src.core.types import DocumentRecord, RawHit, SearchRequest, SearchResult
from src.indexing.mappings import ADDITIVE_FIELDS, index_body
from src.ports.indexing import IndexDoc, IndexSpec
from src.runtime.concurrency import Bulkhead
from src.runtime.metrics import UPSTREAM_ERRORS, UPSTREAM_LATENCY

if TYPE_CHECKING:
    from src.core.config import Settings

_SOURCE_EXCLUDES = {"excludes": ["content_vector"]}
_LEXICAL_FIELDS = ["content^3", "keywords^2", "metadata.table_summary", "metadata.source"]


class ElasticConnection:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        kwargs: dict[str, Any] = {
            "request_timeout": settings.es_request_timeout,
            "max_retries": settings.es_max_retries,
            "retry_on_timeout": True,
            "retry_on_status": (429, 502, 503, 504),
            "connections_per_node": settings.es_connections_per_node,
            "http_compress": True,
            "verify_certs": settings.es_verify_certs,
        }
        if settings.es_ca_certs:
            kwargs["ca_certs"] = settings.es_ca_certs
        if settings.es_cloud_id:
            kwargs["cloud_id"] = settings.es_cloud_id
        elif settings.es_host:
            kwargs["hosts"] = [h.strip() for h in settings.es_host.split(",") if h.strip()]
        else:
            raise ConfigError(
                "Elasticsearch is not configured: set ES_CLOUD_ID (Elastic Cloud) or ES_HOST (self-hosted)"
            )
        api_key = settings.es_api_key.get_secret_value()
        if api_key:
            kwargs["api_key"] = api_key
        elif settings.es_username:
            kwargs["basic_auth"] = (settings.es_username, settings.es_password.get_secret_value())
        self.client = AsyncElasticsearch(**kwargs)
        self._bulkhead = Bulkhead(settings.es_max_concurrency, name="elasticsearch")

    @asynccontextmanager
    async def request(
        self, operation: str, error: type[UpstreamError] = UpstreamError
    ) -> AsyncIterator[None]:
        started = time.perf_counter()
        try:
            async with self._bulkhead:
                yield
        except RagError:
            raise
        except (ApiError, TransportError) as exc:
            UPSTREAM_ERRORS.labels("elasticsearch", operation).inc()
            raise error(f"elasticsearch {operation} failed: {exc}") from exc
        finally:
            UPSTREAM_LATENCY.labels("elasticsearch", operation).observe(time.perf_counter() - started)

    async def ping(self) -> None:
        async with self.request("ping"):
            if not await self.client.ping():
                raise UpstreamError("elasticsearch did not answer ping")

    async def close(self) -> None:
        await self.client.close()


class ElasticIndexWriter:
    def __init__(self, connection: ElasticConnection) -> None:
        self._conn = connection
        self._settings = connection.settings

    async def ensure_indices(self, specs: Sequence[IndexSpec]) -> None:
        for spec in specs:
            await self._ensure(spec)

    async def _ensure(self, spec: IndexSpec) -> None:
        client, s = self._conn.client, self._settings
        async with self._conn.request("ensure_index"):
            exists = bool(await client.indices.exists(index=spec.name))
            if not exists:
                if not s.es_auto_create_indices:
                    raise ConfigError(f"index '{spec.name}' does not exist and ES_AUTO_CREATE_INDICES is off")
                body = index_body(
                    spec.dims,
                    shards=spec.shards if spec.shards is not None else s.es_number_of_shards,
                    replicas=spec.replicas if spec.replicas is not None else s.es_number_of_replicas,
                    refresh_interval=s.es_refresh_interval,
                    vector_index_type=spec.vector_index_type or s.es_vector_index_type,
                )
                try:
                    await client.indices.create(
                        index=spec.name, settings=body["settings"], mappings=body["mappings"]
                    )
                    logger.info("created index %s (dims=%s)", spec.name, spec.dims)
                except ApiError as exc:
                    if exc.error != "resource_already_exists_exception":  # lost a race with another replica
                        raise
                return
            if spec.dims is None:
                return
            mapping = await client.indices.get_mapping(index=spec.name)
            properties = mapping[spec.name]["mappings"].get("properties", {})
            actual = properties.get("content_vector", {}).get("dims")
            if actual != spec.dims:
                raise ConfigError(
                    f"index '{spec.name}' stores {actual}-dimensional vectors but its collection's embedding model "
                    f"produces {spec.dims}. Use a different index name, or re-create the index."
                )
            missing = {name: body for name, body in ADDITIVE_FIELDS.items() if name not in properties}
            if missing:
                await client.indices.put_mapping(index=spec.name, properties=missing)
                logger.info("added fields %s to index %s", sorted(missing), spec.name)

    async def write(self, index: str, docs: Sequence[IndexDoc]) -> None:
        if not docs:
            return
        s = self._settings
        actions = ({"_op_type": "index", "_index": index, "_id": d.id, "_source": d.source} for d in docs)
        failures: list[dict[str, Any]] = []
        async with self._conn.request("bulk", IndexingError):
            async for ok, item in async_streaming_bulk(
                self._conn.client,
                actions,
                chunk_size=s.es_bulk_chunk_size,
                max_chunk_bytes=s.es_bulk_max_chunk_mb * 1024 * 1024,
                raise_on_error=False,
                raise_on_exception=True,
                max_retries=s.es_max_retries,
                initial_backoff=1,
                max_backoff=30,
                yield_ok=False,
            ):
                if not ok:
                    failures.append(item)
        if failures:
            first: Any = next(iter(failures[0].values()), {})
            raise IndexingError(
                f"{len(failures)} of {len(docs)} documents were rejected by {index}; first: {first}"
            )
        logger.info("indexed %d documents into %s", len(docs), index)

    async def _delete(
        self, indices: Sequence[str], query: dict[str, Any], operation: str, *, refresh: bool
    ) -> int:
        if not indices:
            return 0
        async with self._conn.request(operation, IndexingError):
            response = await self._conn.client.delete_by_query(
                index=list(indices),
                query=query,
                conflicts="proceed",
                refresh=refresh,
                wait_for_completion=True,
            )
        if response.get("failures"):
            raise IndexingError(f"{operation} had failures: {response['failures'][:3]}")
        return int(response.get("deleted", 0))

    async def delete_other_generations(
        self, indices: Sequence[str], source: str, keep_document_id: str
    ) -> int:
        query = {
            "bool": {
                "filter": [{"term": {"source": source}}],
                "must_not": [{"term": {"document_id": keep_document_id}}],
            }
        }
        # no refresh here: the ingestion commit refreshes once, after this sweep (or leaves it to the interval)
        return await self._delete(indices, query, "delete_stale", refresh=False)

    async def delete_source(self, indices: Sequence[str], source: str) -> int:
        # a user-facing delete must be visible to the very next search
        return await self._delete(indices, {"term": {"source": source}}, "delete_source", refresh=True)

    async def refresh(self, indices: Sequence[str]) -> None:
        async with self._conn.request("refresh"):
            await self._conn.client.indices.refresh(index=list(indices))


class ElasticSearcher:
    def __init__(self, connection: ElasticConnection) -> None:
        self._conn = connection
        self._fuzziness = connection.settings.es_bm25_fuzziness

    def _filter(self, request: SearchRequest) -> list[dict[str, Any]]:
        return [{"terms": {"source": list(request.sources)}}] if request.sources else []

    def _lexical_body(self, request: SearchRequest) -> dict[str, Any]:
        assert request.text is not None
        return {
            "size": request.size,
            "track_total_hits": False,
            "_source": _SOURCE_EXCLUDES,
            "query": {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {
                                        "multi_match": {
                                            "query": request.text,
                                            "fields": _LEXICAL_FIELDS,
                                            "fuzziness": self._fuzziness,
                                            "operator": "or",
                                        }
                                    },
                                    {
                                        "match_phrase": {
                                            "content": {"query": request.text, "boost": 2.0, "slop": 2}
                                        }
                                    },
                                ],
                                "minimum_should_match": 1,
                            }
                        }
                    ],
                    "filter": self._filter(request),
                }
            },
        }

    def _vector_body(self, request: SearchRequest) -> dict[str, Any]:
        assert request.vector is not None
        knn: dict[str, Any] = {
            "field": "content_vector",
            "query_vector": list(request.vector),
            "k": request.size,
            "num_candidates": max(request.size * 2, 50),
        }
        if request.sources:
            knn["filter"] = {"bool": {"filter": self._filter(request)}}
        return {"size": request.size, "track_total_hits": False, "_source": _SOURCE_EXCLUDES, "knn": knn}

    @staticmethod
    def _hits(response: Any) -> list[RawHit]:
        return [
            RawHit(
                id=h["_id"],
                index=h["_index"],
                score=float(h.get("_score") or 0.0),
                source=h.get("_source", {}),
            )
            for h in response["hits"]["hits"]
        ]

    async def search(self, requests: Sequence[SearchRequest]) -> list[SearchResult]:
        searches: list[dict[str, Any]] = []
        slots: list[tuple[int, str]] = []
        for position, request in enumerate(requests):
            if request.text is not None:
                searches += [{"index": request.index}, self._lexical_body(request)]
                slots.append((position, "lexical"))
            if request.vector is not None:
                searches += [{"index": request.index}, self._vector_body(request)]
                slots.append((position, "vector"))
        results = [SearchResult() for _ in requests]
        if not searches:
            return results
        async with self._conn.request("msearch", SearchError):
            response = await self._conn.client.msearch(searches=searches)
        for (position, which), item in zip(slots, response["responses"], strict=True):
            if "error" in item:
                raise SearchError(f"{which} search on '{requests[position].index}' failed: {item['error']}")
            setattr(results[position], which, self._hits(item))
        return results

    async def fetch(self, indices: Sequence[str], chunk_ids: Sequence[str]) -> list[RawHit]:
        if not chunk_ids or not indices:
            return []
        async with self._conn.request("fetch", SearchError):
            response = await self._conn.client.search(
                index=list(indices),
                query={"terms": {"chunk_id": list(chunk_ids)}},
                size=len(chunk_ids),
                source_excludes=["content_vector"],
                track_total_hits=False,
            )
        return self._hits(response)


class ElasticDocumentRegistry:
    def __init__(self, connection: ElasticConnection, index: str) -> None:
        self._conn = connection
        self._index = index

    @staticmethod
    def _id(collection: str, source: str) -> str:
        return hashlib.sha256(f"{collection}\x1f{source}".encode()).hexdigest()[:40]

    async def get(self, collection: str, source: str) -> DocumentRecord | None:
        async with self._conn.request("registry_get"):
            try:
                response = await self._conn.client.get(index=self._index, id=self._id(collection, source))
            except NotFoundError:
                return None
        return DocumentRecord(**response["_source"])

    async def put(self, record: DocumentRecord) -> None:
        async with self._conn.request("registry_put"):
            await self._conn.client.index(
                index=self._index,
                id=self._id(record.collection, record.source),
                document=asdict(record),
            )  # no refresh: GET (used by the skip check) is real-time, and list() refreshes before it searches

    async def delete(self, collection: str, source: str) -> None:
        async with self._conn.request("registry_delete"):
            try:
                await self._conn.client.delete(index=self._index, id=self._id(collection, source))
            except NotFoundError:
                return

    async def list(self, collection: str, *, limit: int, offset: int) -> tuple[list[DocumentRecord], int]:
        async with self._conn.request("registry_list"):
            # a tiny index, listed rarely: a refresh makes the ledger exactly current without making
            # every ingested document wait for the refresh interval
            await self._conn.client.indices.refresh(index=self._index)
            response = await self._conn.client.search(
                index=self._index,
                query={"term": {"collection": collection}},
                sort=[{"completed_at": "desc"}, {"source": "asc"}],
                size=limit,
                from_=offset,
                track_total_hits=True,
            )
        total = int(response["hits"]["total"]["value"])
        return [DocumentRecord(**hit["_source"]) for hit in response["hits"]["hits"]], total
