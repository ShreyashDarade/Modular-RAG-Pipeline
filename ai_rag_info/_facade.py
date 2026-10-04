"""The public interface, written once.

Method names, argument names, defaults, request validation and the typed errors for bad input are defined
here and nowhere else; the transport (HTTP or in-process) sits behind :class:`~ai_rag_info._backend.Backend`.
That is what keeps the two modes from drifting apart (``docs/adr/0005``).
"""

from __future__ import annotations

import asyncio
import io
import os
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Self

from pydantic import BaseModel, ValidationError
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
from src.core.errors import InvalidRequestError, NotFoundError, RequestTimeoutError, error_from_code
from src.core.types import ContentKind

from ai_rag_info._backend import Backend, IngestOptions, Upload
from ai_rag_info._compat import internal_init

#: What ``documents.ingest`` accepts as its source.
IngestSource = str | os.PathLike[str] | bytes | bytearray | memoryview | BinaryIO


def build_request[M: BaseModel](cls: type[M], /, **fields: Any) -> M:
    """Validate request arguments; bad input is the same typed 400 as when a server refuses it."""
    try:
        return cls(**fields)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'request'}: {e['msg']}" for e in exc.errors())
        raise InvalidRequestError(f"invalid request: {problems}") from exc


def open_upload(source: IngestSource, filename: str | None) -> tuple[Upload, bool]:
    """Normalise a source to an :class:`Upload`; the flag says whether we opened (and must close) it."""
    if isinstance(source, str | os.PathLike):
        path = Path(source)
        try:
            handle = path.open("rb")
        except FileNotFoundError:
            raise NotFoundError(f"file not found: {path}") from None
        except OSError as exc:
            raise InvalidRequestError(f"cannot read {path}: {exc}") from exc
        return Upload(filename or path.name, handle), True
    if isinstance(source, bytes | bytearray | memoryview):
        if not filename:
            raise InvalidRequestError("`filename` is required when ingesting bytes (it selects the parser)")
        return Upload(filename, io.BytesIO(bytes(source))), True
    if hasattr(source, "read"):
        name = filename or os.path.basename(str(getattr(source, "name", "")))
        if not name:
            raise InvalidRequestError("`filename` is required when ingesting a stream without a name")
        seekable = getattr(source, "seekable", None)
        if callable(seekable) and seekable() and source.tell() != 0:
            # the file is ingested from where the stream is positioned: HTTP uploads rewind to the start, so
            # hand both transports the same bytes (the caller's stream is left where it ended up)
            return Upload(name, io.BytesIO(source.read())), True
        return Upload(name, source), False
    raise InvalidRequestError(
        f"cannot ingest a {type(source).__name__}: pass a path, bytes or a binary stream"
    )


@internal_init
class AsyncDocuments:
    def __init__(self, backend: Backend) -> None:
        self._b = backend

    async def ingest(
        self,
        source: IngestSource,
        *,
        filename: str | None = None,
        collection: str | None = None,
        image_language: str | None = None,
        kinds: Sequence[ContentKind] | None = None,
        force: bool = False,
        wait: bool | None = None,
        timeout: float | None = None,
    ) -> IngestResponse:
        """Add a file to a collection.

        ``source`` is a path, bytes (then ``filename`` is required) or a binary stream. With ``wait`` (the
        server's default when ``None``) the call returns the finished result - or, if the job outlasts the
        request timeout, a response whose ``status`` is not ``succeeded``: follow it with ``jobs.wait``.
        A failed job raises its own typed error. Re-ingesting an unchanged file is a no-op (``skipped_reason``).
        """
        upload, owned = open_upload(source, filename)
        try:
            options = IngestOptions(
                collection=collection,
                image_language=image_language,
                kinds=tuple(kinds) if kinds else None,
                force=force,
                wait=wait,
                timeout=timeout,
            )
            return await self._b.ingest(upload, options)
        finally:
            if owned:
                upload.stream.close()

    async def list(self, collection: str | None = None, *, limit: int = 50, offset: int = 0) -> DocumentList:
        """The documents fully ingested into a collection (from the ingestion ledger)."""
        if not 1 <= limit <= 500:
            raise InvalidRequestError("limit must be between 1 and 500")
        if offset < 0:
            raise InvalidRequestError("offset must not be negative")
        return await self._b.list_documents(collection, limit, offset)

    async def delete(self, source: str, collection: str | None = None) -> DeleteResponse:
        """Remove every chunk of one document (by its ``source``) from a collection."""
        if not source:
            raise InvalidRequestError("source must not be empty")
        return await self._b.delete_document(source, collection)


@internal_init
class AsyncChat:
    def __init__(self, backend: Backend) -> None:
        self._b = backend

    @staticmethod
    def _request(
        message: str,
        conversation_id: str | None,
        model: str | None,
        collections: Sequence[str] | None,
        kinds: Sequence[ContentKind] | None,
        sources: Sequence[str] | None,
    ) -> ChatRequest:
        return build_request(
            ChatRequest,
            message=message,
            conversation_id=conversation_id,
            model=model,
            collections=list(collections) if collections else None,
            kinds=list(kinds) if kinds else None,
            sources=list(sources) if sources else None,
        )

    async def send(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        model: str | None = None,
        collections: Sequence[str] | None = None,
        kinds: Sequence[ContentKind] | None = None,
        sources: Sequence[str] | None = None,
    ) -> ChatResponse:
        """One turn. Omit ``conversation_id`` to start a conversation; the response carries the new id.
        An unknown or expired id is an error (``not_found``), never a silently new conversation."""
        return await self._b.chat(self._request(message, conversation_id, model, collections, kinds, sources))

    def stream(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        model: str | None = None,
        collections: Sequence[str] | None = None,
        kinds: Sequence[ContentKind] | None = None,
        sources: Sequence[str] | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        """One turn as events: ``ChatStartEvent`` (sources, conversation id), many ``ChatDeltaEvent``
        (text), then ``ChatEndEvent``. A failure - before or during the stream - is raised as its typed
        error; a stream that ends without ``ChatEndEvent`` is a ``ResponseError``."""
        request = self._request(message, conversation_id, model, collections, kinds, sources)
        return self._b.chat_stream(request)

    async def get(self, conversation_id: str) -> ConversationResponse:
        return await self._b.get_conversation(conversation_id)

    async def delete(self, conversation_id: str) -> None:
        await self._b.delete_conversation(conversation_id)


@internal_init
class AsyncJobs:
    def __init__(self, backend: Backend) -> None:
        self._b = backend

    async def get(self, job_id: str) -> JobResponse:
        """The job's current state (including a failed one - its ``error`` / ``error_code`` fields say why)."""
        return await self._b.get_job(job_id)

    async def wait(self, job_id: str, *, timeout: float = 300.0, poll_interval: float = 0.5) -> JobResponse:
        """Block until the job succeeds and return it. A failed job raises its own typed error (with
        ``details['job_id']``); a job still running after ``timeout`` seconds raises ``RequestTimeoutError``."""
        deadline = time.monotonic() + timeout
        while True:
            job = await self._b.get_job(job_id)
            if job.status == "succeeded":
                return job
            if job.status == "failed":
                raise error_from_code(
                    job.error_code or "job_failed",
                    job.error or "ingestion failed",
                    details={"job_id": job_id},
                )
            if time.monotonic() >= deadline:
                raise RequestTimeoutError(f"job '{job_id}' still {job.status} after {timeout:g}s")
            await asyncio.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


@internal_init
class AsyncCollections:
    def __init__(self, backend: Backend) -> None:
        self._b = backend

    async def list(self) -> list[CollectionInfo]:
        """Every collection: its embedding model, content kinds, accepted parsers and indices."""
        return await self._b.list_collections()


@internal_init
class AsyncRagAPI:
    """The interface of the SDK. Use :class:`~ai_rag_info.AsyncRagClient` (HTTP) or
    :class:`~ai_rag_info.AsyncRag` (in-process); annotate with this class to accept either."""

    documents: AsyncDocuments
    chat: AsyncChat
    jobs: AsyncJobs
    collections: AsyncCollections

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self.documents = AsyncDocuments(backend)
        self.chat = AsyncChat(backend)
        self.jobs = AsyncJobs(backend)
        self.collections = AsyncCollections(backend)

    async def retrieve(
        self,
        query: str,
        *,
        collections: Sequence[str] | None = None,
        kinds: Sequence[ContentKind] | None = None,
        sources: Sequence[str] | None = None,
    ) -> RetrieveResponse:
        """Hybrid (BM25 + vector) search. ``collections`` / ``kinds`` / ``sources`` narrow what is searched;
        omit all three for the default collection."""
        request = build_request(
            RetrieveRequest,
            query=query,
            collections=list(collections) if collections else None,
            kinds=list(kinds) if kinds else None,
            sources=list(sources) if sources else None,
        )
        return await self._backend.retrieve(request)

    async def ask(
        self,
        query: str,
        *,
        collections: Sequence[str] | None = None,
        kinds: Sequence[ContentKind] | None = None,
        sources: Sequence[str] | None = None,
        model: str | None = None,
    ) -> AskResponse:
        """Answer one question from retrieved context with the chosen chat model (default model if ``None``)."""
        request = build_request(
            AskRequest,
            query=query,
            model=model,
            collections=list(collections) if collections else None,
            kinds=list(kinds) if kinds else None,
            sources=list(sources) if sources else None,
        )
        return await self._backend.ask(request)

    async def models(self) -> ModelsResponse:
        """The named chat and embedding models that can be selected."""
        return await self._backend.list_models()

    async def aclose(self) -> None:
        await self._backend.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.aclose()
