"""Blocking wrappers over the async facade (internal).

One daemon thread runs one event loop per sync client; every blocking call is a coroutine submitted to it.
So the sync interface has exactly one implementation to test - the async one - and objects that are tied
to an event loop (HTTP connection pools, the Elasticsearch client) always run on the loop they were
created on. A blocking call made from inside a running loop would stall that loop, so it is a ``UsageError``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Coroutine, Iterator, Sequence
from types import TracebackType
from typing import Any, Self, TypeVar

from src.contracts.models import (
    AskResponse,
    ChatResponse,
    ChatStreamEvent,
    CollectionInfo,
    ConversationResponse,
    DeleteResponse,
    DocumentList,
    IngestResponse,
    JobResponse,
    ModelsResponse,
    RetrieveResponse,
)
from src.core.errors import UsageError
from src.core.types import ContentKind

from ai_rag_info._compat import internal_init
from ai_rag_info._facade import AsyncRagAPI, IngestSource

T = TypeVar("T")


class LoopThread:
    def __init__(self, owner: str) -> None:
        self._owner = owner
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"ai-rag-info-{owner}", daemon=True)
        self._thread.start()
        self._ready.wait()
        self._closed = False

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()
        self._loop.close()

    def _refuse_inside_a_loop(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise UsageError(
            f"{self._owner} blocks the calling thread, which is running an event loop; "
            "use the async class (Async...) inside async code"
        )

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        try:
            self._refuse_inside_a_loop()
        except UsageError:
            coro.close()  # never started: close it so Python does not warn it was never awaited
            raise
        if self._closed:
            coro.close()
            raise UsageError(f"{self._owner} is closed")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def iterate(self, source: AsyncIterator[T]) -> Iterator[T]:
        """A blocking iterator over an async one. Abandoning it (``break``, an exception) closes the source."""
        self._refuse_inside_a_loop()
        try:
            while True:
                try:
                    yield self.run(_next(source))
                except StopAsyncIteration:
                    return
        finally:
            aclose = getattr(source, "aclose", None)
            if aclose is not None and not self._closed:
                self.run(aclose())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=10)


async def _next[T](source: AsyncIterator[T]) -> T:
    return await source.__anext__()


def bridge_for(owner: str) -> LoopThread:
    return LoopThread(owner)


@internal_init
class Documents:
    def __init__(self, api: AsyncRagAPI, bridge: LoopThread) -> None:
        self._a, self._bridge = api.documents, bridge

    def ingest(
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
        """See :meth:`ai_rag_info.AsyncRagAPI.documents` ``ingest``."""
        return self._bridge.run(
            self._a.ingest(
                source,
                filename=filename,
                collection=collection,
                image_language=image_language,
                kinds=kinds,
                force=force,
                wait=wait,
                timeout=timeout,
            )
        )

    def list(self, collection: str | None = None, *, limit: int = 50, offset: int = 0) -> DocumentList:
        return self._bridge.run(self._a.list(collection, limit=limit, offset=offset))

    def delete(self, source: str, collection: str | None = None) -> DeleteResponse:
        return self._bridge.run(self._a.delete(source, collection))


@internal_init
class Chat:
    def __init__(self, api: AsyncRagAPI, bridge: LoopThread) -> None:
        self._a, self._bridge = api.chat, bridge

    def send(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        model: str | None = None,
        collections: Sequence[str] | None = None,
        kinds: Sequence[ContentKind] | None = None,
        sources: Sequence[str] | None = None,
    ) -> ChatResponse:
        return self._bridge.run(
            self._a.send(
                message,
                conversation_id=conversation_id,
                model=model,
                collections=collections,
                kinds=kinds,
                sources=sources,
            )
        )

    def stream(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        model: str | None = None,
        collections: Sequence[str] | None = None,
        kinds: Sequence[ContentKind] | None = None,
        sources: Sequence[str] | None = None,
    ) -> Iterator[ChatStreamEvent]:
        return self._bridge.iterate(
            self._a.stream(
                message,
                conversation_id=conversation_id,
                model=model,
                collections=collections,
                kinds=kinds,
                sources=sources,
            )
        )

    def get(self, conversation_id: str) -> ConversationResponse:
        return self._bridge.run(self._a.get(conversation_id))

    def delete(self, conversation_id: str) -> None:
        self._bridge.run(self._a.delete(conversation_id))


@internal_init
class Jobs:
    def __init__(self, api: AsyncRagAPI, bridge: LoopThread) -> None:
        self._a, self._bridge = api.jobs, bridge

    def get(self, job_id: str) -> JobResponse:
        return self._bridge.run(self._a.get(job_id))

    def wait(self, job_id: str, *, timeout: float = 300.0, poll_interval: float = 0.5) -> JobResponse:
        return self._bridge.run(self._a.wait(job_id, timeout=timeout, poll_interval=poll_interval))


@internal_init
class Collections:
    def __init__(self, api: AsyncRagAPI, bridge: LoopThread) -> None:
        self._a, self._bridge = api.collections, bridge

    def list(self) -> list[CollectionInfo]:
        return self._bridge.run(self._a.list())


@internal_init
class RagAPI:
    """The blocking interface: the same operations as :class:`~ai_rag_info.AsyncRagAPI`."""

    documents: Documents
    chat: Chat
    jobs: Jobs
    collections: Collections

    def __init__(self, api: AsyncRagAPI, bridge: LoopThread) -> None:
        self._api, self._bridge = api, bridge
        self.documents = Documents(api, bridge)
        self.chat = Chat(api, bridge)
        self.jobs = Jobs(api, bridge)
        self.collections = Collections(api, bridge)

    def retrieve(
        self,
        query: str,
        *,
        collections: Sequence[str] | None = None,
        kinds: Sequence[ContentKind] | None = None,
        sources: Sequence[str] | None = None,
    ) -> RetrieveResponse:
        return self._bridge.run(
            self._api.retrieve(query, collections=collections, kinds=kinds, sources=sources)
        )

    def ask(
        self,
        query: str,
        *,
        collections: Sequence[str] | None = None,
        kinds: Sequence[ContentKind] | None = None,
        sources: Sequence[str] | None = None,
        model: str | None = None,
    ) -> AskResponse:
        return self._bridge.run(
            self._api.ask(query, collections=collections, kinds=kinds, sources=sources, model=model)
        )

    def models(self) -> ModelsResponse:
        return self._bridge.run(self._api.models())

    def close(self) -> None:
        try:
            self._bridge.run(self._api.aclose())
        finally:
            self._bridge.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
