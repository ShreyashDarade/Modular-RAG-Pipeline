"""The use cases of the RAG backend.

Every capability exposed to a client - over HTTP, from the embedded SDK, and (in time) from the CLI and
MCP server - is implemented here, once, and returns :mod:`src.contracts` models. Driving adapters translate
their input into one of these calls and the result into their own format; they hold no pipeline logic.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from src.application import mappers
from src.contracts.models import (
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    CollectionInfo,
    ConversationMessage,
    ConversationResponse,
    DeleteResponse,
    DocumentList,
    IngestResponse,
    JobResponse,
    ModelsResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from src.core.container import Container
from src.core.errors import InvalidRequestError, NotFoundError, error_from_code
from src.core.kinds import parse_kinds
from src.core.types import JobRecord, JobSpec, RetrievalScope
from src.ingestion.hints import normalize_language_hint
from src.ingestion.storage import DataStore
from src.runtime.concurrency import deadline, deadline_iter

MAX_PAGE = 500


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    """The state of an ingestion job after the call returned (it may still be running)."""

    record: JobRecord
    #: Where the upload was stored.
    source: str

    @property
    def failed(self) -> bool:
        return self.record.status == "failed"

    @property
    def finished(self) -> bool:
        return self.record.status == "succeeded"

    def response(self, status_url: str | None = None) -> IngestResponse:
        return mappers.ingest_response(self.record, status_url, self.source)

    def raise_if_failed(self) -> None:
        """Raise the job's own typed error (same class as over HTTP), carrying the job id."""
        if self.failed:
            raise error_from_code(
                self.record.error_code or "job_failed",
                self.record.error or "ingestion failed",
                status=self.record.error_status,
                details={"job_id": self.record.id},
            )


class RagService:
    def __init__(self, container: Container) -> None:
        self._c = container
        self._timeout = container.settings.request_timeout_seconds

    # --- search ----------------------------------------------------------------------------------
    def _scope(self, request: RetrieveRequest | AskRequest | ChatRequest) -> RetrievalScope:
        return self._c.retrieval.scope(request.collections, request.kinds, request.sources)

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        scope = self._scope(request)
        async with deadline(self._timeout):
            result = await self._c.retrieval.retrieve(request.query, scope)
        return RetrieveResponse(
            query=result.query,
            expanded_queries=result.expanded_queries,
            documents=[mappers.chunk_of(d) for d in result.documents],
        )

    async def ask(self, request: AskRequest) -> AskResponse:
        scope = self._scope(request)
        async with deadline(self._timeout):
            result = await self._c.answers.ask(request.query, scope, request.model)
        return AskResponse(
            query=result.query,
            expanded_queries=result.expanded_queries,
            answer=result.answer,
            model=result.model,
            context=mappers.ranked_context(result.documents),
        )

    # --- chat ------------------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResponse:
        scope = self._scope(request)
        async with deadline(self._timeout):
            result = await self._c.chat.chat(
                request.message, scope, conversation_id=request.conversation_id, model=request.model
            )
        return ChatResponse(
            conversation_id=result.conversation_id,
            answer=result.answer,
            standalone_query=result.standalone_query,
            model=result.model,
            context=mappers.ranked_context(result.documents),
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        """``start`` (sources, conversation id), ``delta`` (text), ``end``. The whole stream shares one
        deadline. Errors - including those found before the first event (unknown conversation, bad scope) -
        are raised, not emitted as events: the caller decides how to present them."""
        scope = self._scope(request)
        events = self._c.chat.stream(
            request.message, scope, conversation_id=request.conversation_id, model=request.model
        )
        async for event in deadline_iter(events, self._timeout):
            yield mappers.stream_event(event)

    async def conversation(self, conversation_id: str) -> ConversationResponse:
        messages = await self._c.chat.history(conversation_id)
        return ConversationResponse(
            conversation_id=conversation_id,
            messages=[ConversationMessage(role=m.role, content=m.content) for m in messages],
        )

    async def delete_conversation(self, conversation_id: str) -> None:
        await self._c.chat.delete(conversation_id)

    # --- documents and jobs ----------------------------------------------------------------------
    async def ingest(
        self,
        filename: str,
        stream: BinaryIO,
        *,
        collection: str | None = None,
        image_language: str | None = None,
        kinds: Sequence[str] | None = None,
        force: bool = False,
        wait: bool | None = None,
    ) -> IngestOutcome:
        """Store the file and queue it for ingestion; with ``wait`` (default from ``INGEST_WAIT_DEFAULT``)
        return the finished job, or the still-running one if it outlasts the request timeout. Everything
        that can be refused (file type, language, kinds, unknown collection) is refused before anything is stored."""
        c, settings = self._c, self._c.settings
        spec = c.config.collection(collection or c.config.default_collection)
        name = DataStore.safe_name(filename)
        c.parsers.for_path(Path(name), spec.parsers)
        normalize_language_hint(image_language, settings.supported_ocr_languages)
        selected = parse_kinds(list(kinds)) if kinds else None

        stored = await asyncio.to_thread(c.store.save, spec, name, stream)
        record = await c.jobs.submit(
            JobSpec(
                collection=spec.name,
                path=str(stored),
                force=force,
                image_language=image_language,
                kinds=selected,
            )
        )
        if settings.ingest_wait_default if wait is None else wait:
            record = await c.jobs.wait(record.id, max(1, settings.request_timeout_seconds - 5))
        return IngestOutcome(record, str(stored))

    async def job(self, job_id: str) -> JobResponse:
        record = await self._c.jobs.get(job_id)
        if record is None:
            raise NotFoundError(f"job '{job_id}' not found (unknown or expired)")
        return mappers.job_response(record)

    async def documents(
        self, collection: str | None = None, *, limit: int = 50, offset: int = 0
    ) -> DocumentList:
        if not 1 <= limit <= MAX_PAGE:
            raise InvalidRequestError(f"limit must be between 1 and {MAX_PAGE}")
        if offset < 0:
            raise InvalidRequestError("offset must not be negative")
        name = collection or self._c.config.default_collection
        records, total = await self._c.documents.list(name, limit=limit, offset=offset)
        return DocumentList(
            collection=name,
            total=total,
            limit=limit,
            offset=offset,
            documents=[mappers.document_info(r) for r in records],
        )

    async def delete_document(self, source: str, collection: str | None = None) -> DeleteResponse:
        name = collection or self._c.config.default_collection
        deleted = await self._c.documents.delete(name, source)
        return DeleteResponse(success=True, source=source, collection=name, deleted_count=deleted)

    # --- catalog ---------------------------------------------------------------------------------
    async def collections(self) -> list[CollectionInfo]:
        return mappers.collection_infos(self._c.config)

    async def models(self) -> ModelsResponse:
        return mappers.models_response(self._c.config, self._c.models)
