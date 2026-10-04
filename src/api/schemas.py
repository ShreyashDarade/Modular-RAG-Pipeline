from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.types import ContentKind, JobRecord, RetrievedDocument


class _Scoped(BaseModel):
    """Which part of the corpus a request may see. Omit everything for the default collection."""

    collections: list[str] | None = Field(
        default=None, description="Collections to search (default: the default collection)"
    )
    kinds: list[ContentKind] | None = Field(
        default=None, description="Restrict to text, table and/or image content"
    )
    sources: list[str] | None = Field(default=None, description="Restrict to these documents (source paths)")


class RetrieveRequest(_Scoped):
    query: str = Field(min_length=1, max_length=20000)


class AskRequest(_Scoped):
    query: str = Field(min_length=1, max_length=20000)
    model: str | None = Field(default=None, description="Named chat model (see GET /api/v1/models)")


class ChatRequest(_Scoped):
    message: str = Field(min_length=1, max_length=20000)
    conversation_id: str | None = Field(default=None, description="Omit to start a new conversation")
    model: str | None = None


class RetrievedDocumentSchema(BaseModel):
    content: str
    score: float
    source: str | None = None
    page: int | None = None
    type: str | None = None
    keywords: list[str] | None = None
    collection: str | None = None
    kind: ContentKind | None = None

    @classmethod
    def of(cls, doc: RetrievedDocument) -> RetrievedDocumentSchema:
        meta = doc.metadata
        return cls(
            content=doc.content,
            score=doc.final_score,
            source=meta.get("source"),
            page=meta.get("page"),
            type=meta.get("type") or meta.get("content_type"),
            keywords=meta.get("keywords"),
            collection=doc.collection,
            kind=doc.kind,
        )


class RetrieveResponse(BaseModel):
    query: str
    expanded_queries: list[str]
    documents: list[RetrievedDocumentSchema]


class AskContextItem(RetrievedDocumentSchema):
    rank: int

    @classmethod
    def ranked(cls, rank: int, doc: RetrievedDocument) -> AskContextItem:
        return cls(rank=rank, **RetrievedDocumentSchema.of(doc).model_dump())


class AskResponseSchema(BaseModel):
    query: str
    expanded_queries: list[str]
    answer: str
    model: str
    context: list[AskContextItem]


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    standalone_query: str
    model: str
    context: list[AskContextItem]


class ChatMessageSchema(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: list[ChatMessageSchema]


class IngestResponse(BaseModel):
    source: str
    collection: str
    text_chunks: int = Field(ge=0)
    table_chunks: int = Field(ge=0)
    image_chunks: int = Field(ge=0)
    skipped_reason: str | None = None
    reindexed: bool = True
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    status_url: str
    document_id: str | None = None
    total_pages: int | None = None
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def of(cls, record: JobRecord, status_url: str, source: str) -> IngestResponse:
        result: dict[str, Any] = record.result or {}
        return cls(
            source=result.get("source", source),
            collection=record.spec.collection,
            text_chunks=result.get("text_chunks", 0),
            table_chunks=result.get("table_chunks", 0),
            image_chunks=result.get("image_chunks", 0),
            skipped_reason=result.get("skipped_reason"),
            reindexed=result.get("reindexed", record.status != "succeeded"),
            job_id=record.id,
            status=record.status,
            status_url=status_url,
            document_id=result.get("document_id"),
            total_pages=result.get("total_pages"),
            warnings=result.get("warnings", []),
        )


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    collection: str
    source: str
    attempts: int
    result: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None

    @classmethod
    def of(cls, record: JobRecord) -> JobResponse:
        return cls(
            job_id=record.id,
            status=record.status,
            collection=record.spec.collection,
            source=record.spec.path,
            attempts=record.attempts,
            result=record.result,
            error=record.error,
            error_code=record.error_code,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
        )


class CollectionInfo(BaseModel):
    name: str
    description: str
    embedding_model: str
    kinds: list[ContentKind]
    parsers: list[str] | None
    indices: dict[str, str]
    default: bool


class ModelInfo(BaseModel):
    name: str
    provider: str
    model: str
    default: bool = False
    dimensions: int | None = None


class ModelsResponse(BaseModel):
    chat: list[ModelInfo]
    embedding: list[ModelInfo]


class DocumentInfo(BaseModel):
    source: str
    collection: str
    document_id: str
    file_checksum: str
    parser: str
    kinds: list[str]
    text_chunks: int
    table_chunks: int
    image_chunks: int
    total_pages: int
    completed_at: int


class DocumentList(BaseModel):
    collection: str
    total: int
    limit: int
    offset: int
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    success: bool
    source: str
    collection: str
    deleted_count: int
