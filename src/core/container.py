"""Composition root: the only module that knows which concrete class implements which port.

Everything else receives its collaborators through constructors, so any component can be
replaced - by configuration, by a plug-in, or by a fake in a test - without touching its users.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from src.chat.answer import AnswerService
from src.chat.service import ChatService
from src.core.bootstrap import build_registries
from src.core.config import Settings, load_rag_config
from src.core.errors import ConfigError
from src.core.logger import logger
from src.core.registry import Registries
from src.core.specs import RagConfig
from src.core.types import JobSpec
from src.indexing.elastic import (
    ElasticConnection,
    ElasticDocumentRegistry,
    ElasticIndexWriter,
    ElasticSearcher,
)
from src.ingestion.documents import DocumentService
from src.ingestion.service import IngestionService
from src.ingestion.storage import DataStore
from src.models.registry import ModelRegistry
from src.parsing.registry import ParserSet
from src.ports.indexing import IndexSpec
from src.ports.runtime import Cache, ConversationStore, JobBackend, RateLimiter
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.pipeline import RetrievalPipeline
from src.runtime.cache import CachedCall, CorpusVersion

if TYPE_CHECKING:
    from src.ingestion.watcher import DataDirectoryWatcher

Role = Literal["api", "worker", "cli", "mcp"]


@dataclass
class Container:
    settings: Settings
    config: RagConfig
    registries: Registries
    cache: Cache
    corpus: CorpusVersion
    models: ModelRegistry
    elastic: ElasticConnection
    writer: ElasticIndexWriter
    jobs: JobBackend
    conversations: ConversationStore
    retrieval: RetrievalPipeline
    answers: AnswerService
    chat: ChatService
    documents: DocumentService
    store: DataStore
    parsers: ParserSet
    rate_limiter: RateLimiter | None
    ingestion: IngestionService | None = None
    watchers: list[DataDirectoryWatcher] = field(default_factory=list)
    _closers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)

    # --- construction ----------------------------------------------------------------------
    @classmethod
    async def build(
        cls,
        settings: Settings,
        *,
        role: Role,
        with_ingestion: bool | None = None,
        registries: Registries | None = None,
        config: RagConfig | None = None,
    ) -> Container:
        registries = registries or build_registries(settings)
        config = config or load_rag_config(settings)
        cache = registries.caches.create(settings.cache_backend, settings)
        corpus = CorpusVersion(cache)
        models = ModelRegistry(config, settings, registries, query_cache=cache)

        elastic = ElasticConnection(settings)
        writer = ElasticIndexWriter(elastic)
        searcher = ElasticSearcher(elastic)
        doc_registry = ElasticDocumentRegistry(elastic, settings.es_index_registry)

        expander = registries.query_expanders.create(
            config.query_expander, settings, models.chat(config.utility_model), cache
        )
        reranker = registries.rerankers.create(config.reranker, settings)
        retriever = HybridRetriever(
            searcher=searcher, models=models, config=config, reranker=reranker, settings=settings
        )
        retrieval = RetrievalPipeline(
            expander=expander,
            retriever=retriever,
            cache=CachedCall(cache, "retrieval"),
            corpus=corpus,
            settings=settings,
            fingerprint=f"{config.query_expander}/{config.reranker}/{config.utility_model}",
        )
        answers = AnswerService(retrieval=retrieval, models=models, settings=settings)
        conversations = registries.conversation_stores.create(settings.chat_store, settings)
        chat = ChatService(
            answers=answers,
            models=models,
            store=conversations,
            utility_model=config.utility_model,
            settings=settings,
        )
        jobs = registries.job_backends.create(settings.ingest_backend, settings)
        documents = DocumentService(
            config=config, writer=writer, registry=doc_registry, lock=jobs.lock, on_change=corpus.bump
        )
        limiter = (
            registries.rate_limiters.create(settings.rate_limit_backend, settings)
            if role == "api" and settings.rate_limit_per_minute > 0
            else None
        )
        container = cls(
            settings=settings,
            config=config,
            registries=registries,
            cache=cache,
            corpus=corpus,
            models=models,
            elastic=elastic,
            writer=writer,
            jobs=jobs,
            conversations=conversations,
            retrieval=retrieval,
            answers=answers,
            chat=chat,
            documents=documents,
            store=DataStore(settings.data_dir, settings.max_upload_bytes),
            parsers=ParserSet(registries, settings),
            rate_limiter=limiter,
        )
        container._closers = [
            *(c.close for c in (cache, conversations, jobs)),
            *([limiter.close] if limiter else []),
            elastic.close,
        ]
        wants_ingestion = (
            with_ingestion
            if with_ingestion is not None
            else (role == "worker" or (role == "api" and settings.ingest_embedded_worker))
        )
        if wants_ingestion:
            container.ingestion = container._build_ingestion(doc_registry)
        return container

    def _build_ingestion(self, doc_registry: ElasticDocumentRegistry) -> IngestionService:
        # imported here: pulls in scikit-learn / the parsing stack, which API-only images omit
        try:
            from src.chunking.keywords import KeywordExtractor
        except ImportError as exc:
            raise ConfigError(
                "this process is configured to run ingestion but the worker dependencies are not installed: "
                "pip install 'turinton-rag[worker]', or set INGEST_EMBEDDED_WORKER=false and run `rag-worker` separately"
            ) from exc
        from src.ingestion.ocr import LazyOcr

        s = self.settings
        ocr = (
            LazyOcr(lambda: self.registries.ocr_engines.create(s.ocr_engine, s), s.ocr_concurrency)
            if s.ocr_enabled
            else None
        )
        if ocr is not None:
            self._closers.append(ocr.close)
        return IngestionService(
            settings=s,
            config=self.config,
            parsers=self.parsers,
            chunkers={
                n: self.registries.chunkers.create(c.chunker.name, c.chunker)
                for n, c in self.config.collections.items()
            },
            models=self.models,
            writer=self.writer,
            registry=doc_registry,
            ocr=ocr,
            keywords=KeywordExtractor(s.keyword_top_k),
            lock=self.jobs.lock,
            on_change=self.corpus.bump,
        )

    # --- lifecycle -------------------------------------------------------------------------
    async def start(self) -> None:
        specs: list[IndexSpec] = [IndexSpec(self.settings.es_index_registry, None)]
        for collection in self.config.collections.values():
            dims = self.models.embedder(collection.embedding_model).dimensions
            specs += [IndexSpec(name, dims) for name in collection.index_names().values()]
        await self.writer.ensure_indices(specs)
        await self.jobs.start()

    async def start_watchers(self) -> None:
        if not (self.settings.watch_data_dir and self.ingestion):
            return
        extensions = self.ingestion._parsers.extensions
        for name, collection in self.config.collections.items():

            async def submit(path, _name=name) -> None:
                await self.jobs.submit(JobSpec(collection=_name, path=str(path)))

            watcher = DataDirectoryWatcher(
                self.store.directory(collection), extensions, submit, self.settings.watch_debounce_seconds
            )
            await watcher.start()
            self.watchers.append(watcher)

    async def handle_job(self, spec: JobSpec) -> dict:
        assert self.ingestion is not None, "this process was built without the ingestion stack"
        return (await self.ingestion.ingest(spec)).to_dict()

    def start_embedded_worker(self, stop: asyncio.Event) -> asyncio.Task[None]:
        return asyncio.create_task(
            self.jobs.run_worker(self.handle_job, concurrency=self.settings.ingest_concurrency, stop=stop)
        )

    async def readiness(self) -> dict[str, str]:
        """Per-dependency status, ``"ok"`` or the error text. Never raises."""
        checks: dict[str, Callable[[], Awaitable[None]]] = {
            "elasticsearch": self.elastic.ping,
            "cache": self.cache.ping,
            "conversations": self.conversations.ping,
            "jobs": self.jobs.ping,
        }
        if self.rate_limiter:
            checks["rate_limiter"] = self.rate_limiter.ping
        report: dict[str, str] = {}
        for name, check in checks.items():
            try:
                await check()
                report[name] = "ok"
            except Exception as exc:
                report[name] = f"{type(exc).__name__}: {exc}"
        return report

    async def close(self) -> None:
        for watcher in self.watchers:
            await watcher.stop()
        for close in reversed(self._closers):
            try:
                await close()
            except Exception as exc:
                logger.error("error while closing: %s", exc)
