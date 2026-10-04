from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from src.core.specs import RagConfig
from src.core.types import DocumentRecord
from src.ports.indexing import DocumentRegistry, IndexWriter


class DocumentService:
    """List and delete what a collection holds. Needs no parsers or OCR, so API replicas can
    serve it without the ingestion stack."""

    def __init__(
        self,
        *,
        config: RagConfig,
        writer: IndexWriter,
        registry: DocumentRegistry,
        lock: Callable[[str], AbstractAsyncContextManager[Any]],
        on_change: Callable[[], Awaitable[object]],
    ) -> None:
        self._config = config
        self._writer = writer
        self._registry = registry
        self._lock = lock
        self._on_change = on_change

    async def list(self, collection: str, *, limit: int, offset: int) -> tuple[list[DocumentRecord], int]:
        self._config.collection(collection)
        return await self._registry.list(collection, limit=limit, offset=offset)

    async def delete(self, collection: str, source: str) -> int:
        spec = self._config.collection(collection)
        async with self._lock(f"{collection}:{source}"):
            deleted = await self._writer.delete_source(list(spec.index_names().values()), source)
            await self._registry.delete(collection, source)
        await self._on_change()
        return deleted
