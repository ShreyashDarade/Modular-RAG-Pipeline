"""Adapters from LangChain model objects to the ``ChatModel`` / ``Embedder`` ports.

This is the only place LangChain's message and model types meet the domain. Provider SDK
errors are translated into :class:`ModelError` (original kept as ``__cause__``), never
swallowed and never replaced by a degraded answer.
"""

from __future__ import annotations

import hashlib
import time
from array import array
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.core.errors import ModelError, RagError
from src.core.logger import logger
from src.core.types import ChatMessage
from src.ports.models import Embedder
from src.ports.runtime import Cache
from src.runtime.concurrency import Bulkhead, run_all
from src.runtime.metrics import UPSTREAM_ERRORS, UPSTREAM_LATENCY

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel


def _to_langchain(messages: Sequence[ChatMessage]) -> list[BaseMessage]:
    mapping = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return [mapping[m.role](content=m.content) for m in messages]


class _Instrumented:
    """Shared bulkhead + metrics + error translation for one provider client."""

    def __init__(self, service: str, bulkhead: Bulkhead) -> None:
        self._service = service
        self._bulkhead = bulkhead

    def _fail(self, operation: str, model_id: str, exc: Exception) -> ModelError:
        UPSTREAM_ERRORS.labels(self._service, operation).inc()
        logger.error("model call failed", extra={"model": model_id, "operation": operation}, exc_info=exc)
        return ModelError(f"{model_id}: {operation} failed: {type(exc).__name__}: {exc}")


class LangChainChatModel(_Instrumented):
    def __init__(self, model_id: str, chat: BaseChatModel, *, service: str, bulkhead: Bulkhead) -> None:
        super().__init__(service, bulkhead)
        self.model_id = model_id
        self._chat = chat

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        started = time.perf_counter()
        try:
            async with self._bulkhead:
                response = await self._chat.ainvoke(_to_langchain(messages))
        except RagError:
            raise
        except Exception as exc:
            raise self._fail("complete", self.model_id, exc) from exc
        UPSTREAM_LATENCY.labels(self._service, "complete").observe(time.perf_counter() - started)
        return str(response.text)

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        started = time.perf_counter()
        try:
            async with self._bulkhead:
                async for chunk in self._chat.astream(_to_langchain(messages)):
                    text = str(chunk.text)
                    if text:
                        yield text
        except RagError:
            raise
        except Exception as exc:
            raise self._fail("stream", self.model_id, exc) from exc
        UPSTREAM_LATENCY.labels(self._service, "stream").observe(time.perf_counter() - started)


class LangChainEmbedder(_Instrumented):
    """Splits work into ``batch_size`` requests and runs them concurrently (bounded by the
    bulkhead) instead of letting the SDK walk the batches one after another."""

    def __init__(
        self,
        model_id: str,
        embeddings: Embeddings,
        *,
        dimensions: int,
        batch_size: int,
        service: str,
        bulkhead: Bulkhead,
        batch_queries: bool = True,
    ) -> None:
        super().__init__(service, bulkhead)
        self.model_id = model_id
        self.dimensions = dimensions
        self._embeddings = embeddings
        self._batch_size = batch_size
        #: True when the model encodes queries and documents identically (OpenAI): queries then go in
        #: one batched request. Providers with distinct query encodings (task types, prefixes) must
        #: use their own ``aembed_query`` for each query.
        self._batch_queries = batch_queries

    async def _call(
        self, operation: str, count: int, call: Callable[[], Awaitable[list[list[float]]]]
    ) -> list[list[float]]:
        started = time.perf_counter()
        try:
            async with self._bulkhead:
                vectors = await call()
        except RagError:
            raise
        except Exception as exc:
            raise self._fail(operation, self.model_id, exc) from exc
        UPSTREAM_LATENCY.labels(self._service, operation).observe(time.perf_counter() - started)
        if len(vectors) != count or any(len(v) != self.dimensions for v in vectors):
            raise ModelError(
                f"{self.model_id}: expected {count} vectors of {self.dimensions} dimensions, "
                f"got {len(vectors)} of {{{', '.join(sorted({str(len(v)) for v in vectors}))}}}"
            )
        return vectors

    async def _embed_batch(self, operation: str, batch: list[str]) -> list[list[float]]:
        async def call() -> list[list[float]]:
            return await self._embeddings.aembed_documents(batch)

        return await self._call(operation, len(batch), call)

    async def _embed_one_query(self, text: str) -> list[float]:
        async def call() -> list[list[float]]:
            return [await self._embeddings.aembed_query(text)]

        return (await self._call("embed_queries", 1, call))[0]

    async def _embed(self, texts: Sequence[str], *, query: bool) -> list[list[float]]:
        if not texts:
            return []
        if query and not self._batch_queries:
            return list(await run_all(self._embed_one_query(t) for t in texts))
        operation = "embed_queries" if query else "embed_documents"
        batches = [list(texts[i : i + self._batch_size]) for i in range(0, len(texts), self._batch_size)]
        results = await run_all(self._embed_batch(operation, batch) for batch in batches)
        return [vector for batch_vectors in results for vector in batch_vectors]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, query=False)

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, query=True)


class CachingEmbedder:
    """Decorator adding a bounded in-process LRU (documents and queries) and, for queries, an
    optional shared cache so replicas do not re-embed the same question.

    Keys cover model, dimensions, the *role* (a model may embed queries and documents differently:
    prefixes, instructions) and the *full* text, so two inputs only share a vector if they are
    byte-identical and were embedded the same way.
    """

    def __init__(
        self,
        inner: Embedder,
        *,
        max_entries: int,
        shared: Cache | None = None,
        shared_ttl: int = 86_400,
        identity: str | None = None,
    ) -> None:
        self._inner = inner
        #: What the vectors depend on (provider, model, size, endpoint). The profile *name* is not
        #: enough: re-pointing a profile at another model must not reuse cached vectors.
        self._identity = identity or inner.model_id
        self.model_id = inner.model_id
        self.dimensions = inner.dimensions
        self._max = max_entries
        self._shared = shared
        self._ttl = shared_ttl
        self._lru: OrderedDict[str, bytes] = OrderedDict()

    def _key(self, text: str, query: bool) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()
        return f"emb:{self._identity}:{self.dimensions}:{'q' if query else 'd'}:{digest}"

    def _remember(self, key: str, packed: bytes) -> None:
        if self._max == 0:
            return
        self._lru[key] = packed
        self._lru.move_to_end(key)
        while len(self._lru) > self._max:
            self._lru.popitem(last=False)

    @staticmethod
    def _pack(vector: Sequence[float]) -> bytes:
        return array("f", vector).tobytes()

    @staticmethod
    def _unpack(packed: bytes) -> list[float]:
        values = array("f")
        values.frombytes(packed)
        return values.tolist()

    async def _embed(self, texts: Sequence[str], *, query: bool) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, list[int]] = {}
        for index, text in enumerate(texts):
            key = self._key(text, query)
            packed = self._lru.get(key)
            if packed is None and query and self._shared is not None:
                packed = await self._shared.get(key)
                if packed is not None:
                    self._remember(key, packed)
            if packed is not None:
                if key in self._lru:
                    self._lru.move_to_end(key)
                results[index] = self._unpack(packed)
            else:
                missing.setdefault(text, []).append(index)
        if missing:
            unique = list(missing)
            embed = self._inner.embed_queries if query else self._inner.embed_documents
            vectors = await embed(unique)
            for text, vector in zip(unique, vectors, strict=True):
                packed = self._pack(vector)
                key = self._key(text, query)
                self._remember(key, packed)
                if query and self._shared is not None:
                    await self._shared.set(key, packed, self._ttl)
                for index in missing[text]:
                    results[index] = vector
        return results  # type: ignore[return-value]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, query=False)

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, query=True)
