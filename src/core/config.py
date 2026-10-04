"""Environment-driven runtime settings.

Settings describe *infrastructure and limits* (Elasticsearch, Redis, credentials, concurrency).
What the system is *made of* - models and collections - is described by
:class:`src.core.specs.RagConfig`, loaded from ``RAG_CONFIG`` (TOML) or synthesised from the
classic ``OPENAI_*`` / ``CHUNK_*`` / ``ES_INDEX_*`` variables by :func:`load_rag_config`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.specs import ChatModelSpec, ChunkerSpec, CollectionSpec, EmbeddingModelSpec, RagConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- application -------------------------------------------------------------------------
    app_name: str = "OCR-rag"
    environment: str = "production"
    data_dir: Path = Path("data").resolve()
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    #: Modules exposing ``register(registries)``; the plug-in hook for new parsers, providers, ...
    plugins: list[str] = Field(default_factory=list)
    rag_config: Path | None = None

    # --- Elasticsearch ------------------------------------------------------------------------
    es_cloud_id: str = ""
    es_host: str = ""  # self-hosted: one URL or a comma separated list
    es_api_key: SecretStr = SecretStr("")
    es_username: str = ""
    es_password: SecretStr = SecretStr("")
    es_verify_certs: bool = True
    es_ca_certs: str | None = None
    es_index_text: str = "doc-text"
    es_index_tables: str = "doc-tables"
    es_index_images: str = "doc-images"
    es_index_registry: str = "doc-registry"
    #: Create missing indices on start-up. Off => a missing index is an error.
    es_auto_create_indices: bool = True
    es_number_of_shards: int = Field(default=1, gt=0)
    es_number_of_replicas: int = Field(default=1, ge=0)
    es_refresh_interval: str = "5s"
    #: dense_vector index type: int8_hnsw (8.12+), bbq_hnsw (8.18+/9.x, ~32x smaller), hnsw, ...
    es_vector_index_type: str = "int8_hnsw"
    es_request_timeout: float = 60.0
    es_max_retries: int = 3
    es_connections_per_node: int = Field(default=32, gt=0)
    es_max_concurrency: int = Field(default=64, gt=0)
    es_bulk_chunk_size: int = Field(default=500, gt=0)
    es_bulk_max_chunk_mb: int = Field(default=10, gt=0)
    es_bm25_fuzziness: str = "AUTO"

    # --- model providers (credentials; the models themselves live in RagConfig) ----------------
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    ollama_base_url: str | None = None
    model_timeout_seconds: float = 30.0
    model_max_retries: int = 4
    #: Concurrent in-flight calls per provider client; excess callers queue instead of piling on.
    model_max_concurrency: int = Field(default=32, gt=0)
    embedding_batch_size: int = Field(default=256, gt=0)
    embedding_cache_entries: int = Field(default=10_000, ge=0)

    # --- chunking (defaults for the synthesised "default" collection) --------------------------
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    min_chunk_size: int = Field(default=100, ge=0)
    keyword_top_k: int = Field(default=20, gt=0)

    # --- retrieval ----------------------------------------------------------------------------
    retriever_top_k: int = Field(default=10, gt=0)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rerank_enabled: bool = True
    rerank_top_k: int = Field(default=6, gt=0)
    query_expansion_enabled: bool = True
    query_expansion_timeout_seconds: float = 5.0
    query_expansion_cache_ttl_seconds: int = 86_400
    enable_cross_references: bool = True
    page_context_window: int = Field(default=1, ge=0)
    max_query_chars: int = Field(default=2000, gt=0)

    # --- chat ---------------------------------------------------------------------------------
    chat_store: Literal["memory", "redis"] = "memory"
    chat_history_messages: int = Field(default=20, ge=0)
    #: Character budget for the history sent to the model; oldest turns are dropped first.
    chat_history_max_chars: int = Field(default=24_000, gt=0)
    chat_history_ttl_seconds: int = 86_400
    chat_memory_conversations: int = Field(default=10_000, gt=0)
    chat_condense_questions: bool = True

    # --- ingestion ----------------------------------------------------------------------------
    max_upload_mb: int = Field(default=100, gt=0)
    ingest_backend: Literal["inprocess", "redis"] = "inprocess"
    #: Concurrent ingestion jobs per worker process.
    ingest_concurrency: int = Field(default=2, gt=0)
    #: Run ingestion workers inside the API process. Turn off on API replicas of a distributed deployment.
    ingest_embedded_worker: bool = True
    ingest_queue_max_size: int = Field(default=1000, gt=0)
    ingest_wait_default: bool = True
    ingest_max_attempts: int = Field(default=3, gt=0)
    #: "each": refresh the indices when every document finishes, so it is searchable the moment its job
    #: reports success. "interval": rely on ES_REFRESH_INTERVAL - less work for the cluster when bulk-loading.
    ingest_refresh: Literal["each", "interval"] = "each"
    #: Embedding+indexing slices in flight per document; bounds the memory held as vectors.
    ingest_pipeline_depth: int = Field(default=4, gt=0)
    job_ttl_seconds: int = 86_400
    job_visibility_timeout_seconds: int = Field(default=300, gt=0)
    shutdown_grace_seconds: int = 120
    #: Prometheus metrics port of the standalone worker (0 disables).
    worker_metrics_port: int = 9100
    watch_data_dir: bool = False
    watch_debounce_seconds: float = 2.0
    pdf_extract_tables: bool = True

    # --- OCR ----------------------------------------------------------------------------------
    ocr_enabled: bool = True
    ocr_engine: str = "easyocr"
    ocr_gpu_enabled: bool = True
    ocr_model_dir: Path = Path("models")
    #: Fetch missing model weights on first use. Turn off in images that bake the weights in.
    ocr_download_models: bool = True
    ocr_batch_size: int = Field(default=5, gt=0)
    ocr_concurrency: int = Field(default=1, gt=0)
    ocr_min_image_side: int = Field(default=64, ge=0)
    ocr_max_side: int = Field(default=2560, gt=0)
    #: Skip the (slow) pre-processed second pass when the raw image already reads this well.
    ocr_early_exit_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    #: Same, for the Hindi/Marathi model. Its confidence is poorly calibrated (0.26-0.47 on text that was
    #: read 97% correctly) and on the degraded scans measured the second pass never improved the result, so
    #: by default the first pass is accepted. Raise it (up to 1.0) for workloads of very poor scans.
    ocr_early_exit_confidence_devanagari: float = Field(default=0.0, ge=0.0, le=1.0)
    supported_ocr_languages: list[str] = Field(default_factory=lambda: ["en", "mr", "hi"])

    # --- API ----------------------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    web_concurrency: int = Field(default=1, gt=0)
    rate_limit_per_minute: int = Field(default=100, ge=0)  # 0 disables
    rate_limit_backend: Literal["memory", "redis"] = "memory"
    request_timeout_seconds: int = Field(default=120, gt=0)
    #: Admission control: above this many in-flight requests new ones get 503 + Retry-After. 0 disables.
    max_concurrent_requests: int = Field(default=1024, ge=0)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    metrics_enabled: bool = True
    mcp_enabled: bool = False
    mcp_path: str = "/mcp"
    #: Host header values the MCP endpoint accepts (DNS-rebinding protection). Empty = localhost only.
    mcp_allowed_hosts: list[str] = Field(default_factory=list)
    mcp_allowed_origins: list[str] = Field(default_factory=list)

    # --- cache / redis ------------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379"
    #: Key prefix for everything this system stores in Redis, so environments can share one Redis.
    redis_namespace: str = "rag"
    #: Connect/read timeout of every Redis client. Must exceed the 1 s queue poll of the workers.
    redis_socket_timeout_seconds: float = Field(default=5.0, ge=2.0)
    cache_backend: Literal["memory", "redis", "tiered"] = "memory"
    cache_ttl_seconds: int = 3600
    cache_memory_entries: int = Field(default=2000, gt=0)
    use_redis_cache: bool = False  # legacy switch, equivalent to cache_backend=tiered

    @model_validator(mode="after")
    def _legacy_redis_cache_flag(self) -> Self:
        if self.use_redis_cache and "cache_backend" not in self.model_fields_set:
            self.cache_backend = "tiered"
        return self

    @model_validator(mode="after")
    def _refresh_interval_is_usable(self) -> Self:
        if self.es_refresh_interval != "-1" and _duration_seconds(self.es_refresh_interval) is None:
            raise ValueError(
                f"ES_REFRESH_INTERVAL must look like 500ms, 1s, 5s, 1m or -1: {self.es_refresh_interval!r}"
            )
        if self.ingest_refresh == "interval" and self.es_refresh_interval == "-1":
            raise ValueError("INGEST_REFRESH=interval needs a real ES_REFRESH_INTERVAL; -1 never refreshes")
        return self

    @property
    def search_settle_seconds(self) -> float:
        """How long after an ingest the new chunks may still be invisible to search (0: they are not)."""
        if self.ingest_refresh == "each":
            return 0.0
        return (_duration_seconds(self.es_refresh_interval) or 0.0) + 1.0

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def uses_redis(self) -> bool:
        return (
            self.cache_backend != "memory"
            or self.rate_limit_backend == "redis"
            or self.ingest_backend == "redis"
            or self.chat_store == "redis"
        )


_DURATION = re.compile(r"^(\d+)(ms|s|m|h)$")
_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def _duration_seconds(value: str) -> float | None:
    """Seconds in an Elasticsearch time value (``500ms``, ``5s``, ``1m``); None if it is not one."""
    match = _DURATION.match(value)
    return int(match[1]) * _UNITS[match[2]] if match else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_rag_config(settings: Settings) -> RagConfig:
    """The TOML file if ``RAG_CONFIG`` is set, otherwise one OpenAI chat model, one OpenAI
    embedding model and one collection built from the classic environment variables."""
    if settings.rag_config is not None:
        return RagConfig.from_toml(settings.rag_config)
    return RagConfig(
        default_chat_model="default",
        default_collection="default",
        utility_model="default",
        query_expander="llm" if settings.query_expansion_enabled else "identity",
        reranker="heuristic" if settings.rerank_enabled else "identity",
        chat_models={
            "default": ChatModelSpec(
                provider="openai",
                model=settings.openai_model,
                temperature=0.1,
                max_tokens=2048,
                base_url=settings.openai_base_url,
            )
        },
        embedding_models={
            "default": EmbeddingModelSpec(
                provider="openai",
                model=settings.openai_embedding_model,
                dimensions=settings.openai_embedding_dimensions,
                base_url=settings.openai_base_url,
            )
        },
        collections={
            "default": CollectionSpec(
                name="default",
                description="Default collection",
                embedding_model="default",
                data_subdir="",
                chunker=ChunkerSpec(
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    min_chunk_size=settings.min_chunk_size,
                ),
                indices={
                    "text": settings.es_index_text,
                    "table": settings.es_index_tables,
                    "image": settings.es_index_images,
                },
            )
        },
    )
