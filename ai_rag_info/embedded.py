"""The in-process SDK: run the whole pipeline inside your application.

Needs the engine: ``pip install 'ai-rag-info[engine]'`` (plus ``worker`` to ingest files, ``local`` for
self-hosted models, ...). The interface is the one :class:`~ai_rag_info.AsyncRagClient` exposes over HTTP -
it is the same facade over :class:`~src.application.RagService` - plus :meth:`AsyncRag.evaluate`, which only
makes sense where the engine is.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

_ENGINE = ("elasticsearch", "pydantic_settings", "redis", "langchain_core", "prometheus_client")
if missing := [m for m in _ENGINE if importlib.util.find_spec(m) is None]:
    raise ImportError(
        f"ai_rag_info.embedded needs the engine ({', '.join(missing)} not installed): "
        "pip install 'ai-rag-info[engine]'"
    )

from src.application import RagService
from src.contracts.models import (
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    CollectionInfo,
    ConversationResponse,
    DeleteResponse,
    DocumentList,
    IngestResponse,
    JobResponse,
    ModelsResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from src.core.bootstrap import build_registries
from src.core.config import Settings, get_settings, load_rag_config
from src.core.container import Container
from src.core.errors import ConfigError, RequestTimeoutError
from src.core.logger import logger
from src.core.registry import Registries
from src.core.specs import RagConfig

from ai_rag_info._backend import IngestOptions, Upload
from ai_rag_info._compat import experimental, internal_init
from ai_rag_info._facade import AsyncRagAPI
from ai_rag_info._sync import RagAPI, bridge_for

if TYPE_CHECKING:
    from src.evaluation.dataset import Dataset
    from src.evaluation.report import EvalReport
    from src.evaluation.runner import Granularity


class EmbeddedBackend:
    """Calls the use-case layer directly: no HTTP, same models, same typed errors."""

    def __init__(self, container: Container, *, owns_container: bool) -> None:
        self._container = container
        self._service = RagService(container)
        self._owns = owns_container
        self._stop = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._owns:
            await self._container.start()
            if self._container.ingestion is not None:
                self._worker = self._container.start_embedded_worker(self._stop)

    # --- operations ------------------------------------------------------------------------------
    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        return await self._service.retrieve(request)

    async def ask(self, request: AskRequest) -> AskResponse:
        return await self._service.ask(request)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        return await self._service.chat(request)

    def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        return self._service.chat_stream(request)

    async def get_conversation(self, conversation_id: str) -> ConversationResponse:
        return await self._service.conversation(conversation_id)

    async def delete_conversation(self, conversation_id: str) -> None:
        await self._service.delete_conversation(conversation_id)

    async def ingest(self, upload: Upload, options: IngestOptions) -> IngestResponse:
        c = self._container
        if c.ingestion is None and c.settings.ingest_backend == "inprocess":
            raise ConfigError(
                "this engine has no ingestion worker (it was created with ingestion=False); "
                "create it with ingestion=True, or use INGEST_BACKEND=redis with separate workers"
            )
        work = self._service.ingest(
            upload.filename,
            upload.stream,
            collection=options.collection,
            image_language=options.image_language,
            kinds=options.kinds,
            force=options.force,
            wait=options.wait,
        )
        try:
            outcome = await (asyncio.wait_for(work, options.timeout) if options.timeout else work)
        except TimeoutError:
            raise RequestTimeoutError(f"ingestion did not return within {options.timeout:g}s") from None
        outcome.raise_if_failed()
        return outcome.response()

    async def get_job(self, job_id: str) -> JobResponse:
        return await self._service.job(job_id)

    async def list_documents(self, collection: str | None, limit: int, offset: int) -> DocumentList:
        return await self._service.documents(collection, limit=limit, offset=offset)

    async def delete_document(self, source: str, collection: str | None) -> DeleteResponse:
        return await self._service.delete_document(source, collection)

    async def list_collections(self) -> list[CollectionInfo]:
        return await self._service.collections()

    async def list_models(self) -> ModelsResponse:
        return await self._service.models()

    async def aclose(self) -> None:
        if not self._owns:
            return
        self._stop.set()
        if self._worker is not None:
            grace = self._container.settings.shutdown_grace_seconds
            try:
                await asyncio.wait_for(self._worker, grace)
            except TimeoutError:
                logger.error("ingestion worker did not drain within %ss", grace)
                self._worker.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._worker
            except Exception as exc:  # the worker had died earlier; closing must still happen
                logger.error("ingestion worker had stopped with an error", exc_info=exc)
        await self._container.close()


@internal_init
class AsyncRag(AsyncRagAPI):
    """The pipeline, in-process. Create it with :meth:`create` (it builds the engine and starts ingestion)::

        async with await AsyncRag.create() as rag:
            await rag.documents.ingest("report.pdf")
            answer = await rag.ask("What changed in Q2?")

    Configuration is the engine's usual: environment / ``.env`` for infrastructure, ``RAG_CONFIG`` (TOML) for
    models and collections - or pass ``settings`` / ``config`` explicitly.
    """

    _backend: EmbeddedBackend

    def __init__(self, backend: EmbeddedBackend, container: Container) -> None:
        super().__init__(backend)
        self.engine = container  # the composition root, for applications that need to reach below the SDK

    @classmethod
    async def create(
        cls,
        settings: Settings | None = None,
        config: RagConfig | None = None,
        *,
        registries: Registries | None = None,
        ingestion: bool = True,
    ) -> AsyncRag:
        """Build and start the engine. ``ingestion=False`` skips the ingestion stack (no parsers, OCR or
        worker dependencies needed); then ``documents.ingest`` works only with ``INGEST_BACKEND=redis``
        and separate workers."""
        settings = settings or get_settings()
        container = await Container.build(
            settings,
            role="api",
            with_ingestion=ingestion,
            config=config or load_rag_config(settings),
            registries=registries or build_registries(settings),
        )
        backend = EmbeddedBackend(container, owns_container=True)
        try:
            await backend.start()
        except BaseException:
            await backend.aclose()
            raise
        return cls(backend, container)

    @classmethod
    def from_container(cls, container: Container) -> AsyncRag:
        """Wrap an engine somebody else built and runs (its lifecycle stays theirs; ``aclose`` does nothing)."""
        return cls(EmbeddedBackend(container, owns_container=False), container)

    @experimental
    async def evaluate(
        self,
        dataset: Dataset | str | Path,
        *,
        name: str = "run",
        collection: str | None = None,
        ks: Sequence[int] = (1, 3, 5, 10),
        granularity: Granularity = "chunk",
        answers: bool = False,
        judge_model: str | None = None,
        answer_model: str | None = None,
        concurrency: int = 4,
    ) -> EvalReport:
        """Score retrieval (and, with ``answers``, judged answers) on a labelled question set.

        Runs against this engine exactly as configured - including its cache, so repeated queries may be
        answered from it; use ``rag eval`` for isolated, cache-free configuration comparisons.
        """
        from src.evaluation.dataset import Dataset as DatasetType
        from src.evaluation.dataset import load_dataset
        from src.evaluation.session import evaluate_container

        data = dataset if isinstance(dataset, DatasetType) else load_dataset(Path(dataset))
        return await evaluate_container(
            self.engine,
            data,
            name=name,
            collection=collection,
            ks=ks,
            granularity=granularity,
            answers=answers,
            judge_model=judge_model,
            answer_model=answer_model,
            concurrency=concurrency,
        )


class Rag(RagAPI):
    """Blocking version of :class:`AsyncRag` (one background event loop; see :class:`RagClient`)."""

    def __init__(
        self,
        settings: Settings | None = None,
        config: RagConfig | None = None,
        *,
        registries: Registries | None = None,
        ingestion: bool = True,
    ) -> None:
        bridge = bridge_for("Rag")
        try:
            engine = bridge.run(AsyncRag.create(settings, config, registries=registries, ingestion=ingestion))
        except BaseException:
            bridge.close()
            raise
        super().__init__(engine, bridge)
        self._engine = engine

    @property
    def engine(self) -> Container:
        return self._engine.engine

    @experimental
    def evaluate(
        self,
        dataset: Dataset | str | Path,
        *,
        name: str = "run",
        collection: str | None = None,
        ks: Sequence[int] = (1, 3, 5, 10),
        granularity: Granularity = "chunk",
        answers: bool = False,
        judge_model: str | None = None,
        answer_model: str | None = None,
        concurrency: int = 4,
    ) -> EvalReport:
        """See :meth:`AsyncRag.evaluate`."""
        return self._bridge.run(
            self._engine.evaluate(
                dataset,
                name=name,
                collection=collection,
                ks=ks,
                granularity=granularity,
                answers=answers,
                judge_model=judge_model,
                answer_model=answer_model,
                concurrency=concurrency,
            )
        )


__all__ = ["AsyncRag", "Rag"]
