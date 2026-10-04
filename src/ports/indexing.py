from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.core.types import DocumentRecord, RawHit, SearchRequest, SearchResult


@dataclass(frozen=True, slots=True)
class IndexSpec:
    """An index to create. ``dims`` is None for indices that hold no vectors."""

    name: str
    dims: int | None
    #: Per-index overrides of the global ES_* settings (``None`` = use the global value).
    shards: int | None = None
    replicas: int | None = None
    vector_index_type: str | None = None


@dataclass(frozen=True, slots=True)
class IndexDoc:
    id: str
    source: dict[str, Any]


class IndexWriter(Protocol):
    async def ensure_indices(self, specs: Sequence[IndexSpec]) -> None: ...

    async def write(self, index: str, docs: Sequence[IndexDoc]) -> None:
        """Idempotent upsert keyed by ``IndexDoc.id``. Raises ``IndexingError`` if any doc fails."""

    async def delete_other_generations(
        self, indices: Sequence[str], source: str, keep_document_id: str
    ) -> int:
        """Remove chunks of ``source`` that were written by a different ingestion run."""

    async def delete_source(self, indices: Sequence[str], source: str) -> int: ...

    async def refresh(self, indices: Sequence[str]) -> None: ...


class Searcher(Protocol):
    async def search(self, requests: Sequence[SearchRequest]) -> list[SearchResult]:
        """One round trip for all requests. Raises ``SearchError`` if any of them fails."""

    async def fetch(self, indices: Sequence[str], chunk_ids: Sequence[str]) -> list[RawHit]: ...

    async def sample(self, index: str, size: int, seed: int) -> list[RawHit]:
        """A reproducible pseudo-random sample of up to ``size`` chunks (used to build evaluation sets)."""


class DocumentRegistry(Protocol):
    async def get(self, collection: str, source: str) -> DocumentRecord | None: ...

    async def put(self, record: DocumentRecord) -> None: ...

    async def delete(self, collection: str, source: str) -> None: ...

    async def list(self, collection: str, *, limit: int, offset: int) -> tuple[list[DocumentRecord], int]: ...
