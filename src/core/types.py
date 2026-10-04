"""Domain value objects shared by every layer. No I/O, no third-party imports."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

ContentKind = Literal["text", "table", "image"]
CONTENT_KINDS: tuple[ContentKind, ...] = ("text", "table", "image")

Role = Literal["system", "user", "assistant"]


def content_type_label(parser: str, kind: ContentKind) -> str:
    """Stable ``type`` label stored with each chunk (``pdf_text``, ``pdf_table``, ``image``, ...)."""
    return kind if parser == kind else f"{parser}_{kind}"


# --- chat ----------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


# --- indexing ------------------------------------------------------------------------------
@dataclass(slots=True)
class ChunkRecord:
    """One searchable chunk exactly as stored in an index."""

    chunk_id: str
    content: str
    kind: ContentKind
    content_type: str
    vector: list[float]
    keywords: list[str]
    language: str
    source: str
    page: int | None
    document_id: str
    file_checksum: str
    metadata: dict[str, Any]
    sibling_chunk_ids: list[str] = field(default_factory=list)
    adjacent_chunk_ids: list[str] = field(default_factory=list)
    has_table_on_page: bool = False
    has_image_on_page: bool = False
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_source(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "content_vector": self.vector,
            "keywords": self.keywords,
            "metadata": self.metadata,
            "language": self.language,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "file_checksum": self.file_checksum,
            "sibling_chunk_ids": self.sibling_chunk_ids,
            "adjacent_chunk_ids": self.adjacent_chunk_ids,
            "content_type": self.content_type,
            "kind": self.kind,
            "has_table_on_page": self.has_table_on_page,
            "has_image_on_page": self.has_image_on_page,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class DocumentRecord:
    """Ingestion ledger entry: one per (collection, source). Written last, so its presence with
    ``status == "complete"`` and a matching checksum proves the document is fully indexed."""

    collection: str
    source: str
    file_checksum: str
    document_id: str
    status: Literal["complete"] = "complete"
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    total_pages: int = 0
    parser: str = ""
    kinds: list[str] = field(default_factory=list)
    completed_at: int = field(default_factory=lambda: int(time.time() * 1000))


# --- searching -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Backend-neutral request: rank ``index`` by lexical match to ``text`` and by similarity to
    ``vector``. Either may be omitted. ``sources`` restricts the search to those documents."""

    index: str
    text: str | None
    vector: tuple[float, ...] | None
    size: int
    sources: tuple[str, ...] = ()


@dataclass(slots=True)
class RawHit:
    id: str
    index: str
    score: float
    source: dict[str, Any]


@dataclass(slots=True)
class SearchResult:
    lexical: list[RawHit] = field(default_factory=list)
    vector: list[RawHit] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RetrievalScope:
    """What a query is allowed to see. Empty ``sources`` means every document."""

    collections: tuple[str, ...]
    kinds: tuple[ContentKind, ...] = CONTENT_KINDS
    sources: tuple[str, ...] = ()


@dataclass(slots=True)
class RetrievedDocument:
    content: str
    metadata: dict[str, Any]
    score: float
    collection: str
    kind: ContentKind
    index: str
    rerank_score: float = 0.0

    @property
    def final_score(self) -> float:
        return self.rerank_score or self.score


@dataclass(slots=True)
class RetrievalResult:
    query: str
    expanded_queries: list[str]
    documents: list[RetrievedDocument]


@dataclass(slots=True)
class AnswerResult:
    query: str
    expanded_queries: list[str]
    answer: str
    model: str
    documents: list[RetrievedDocument]


@dataclass(slots=True)
class ChatResult:
    conversation_id: str
    answer: str
    standalone_query: str
    model: str
    documents: list[RetrievedDocument]


# --- jobs ----------------------------------------------------------------------------------
@dataclass(slots=True)
class JobSpec:
    collection: str
    path: str
    force: bool = False
    image_language: str | None = None
    kinds: tuple[ContentKind, ...] | None = None


JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass(slots=True)
class JobRecord:
    id: str
    spec: JobSpec
    status: JobStatus = "queued"
    attempts: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None
    error_status: int | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def finished(self) -> bool:
        return self.status in ("succeeded", "failed")
