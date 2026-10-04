"""Request and response models of the ``/api/v1`` REST API (and of the SDK).

Responses are read tolerantly by clients (unknown fields are ignored) but validated strictly (required fields
and types), so a newer server never breaks an older SDK and a malformed response is never accepted quietly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.types import ContentKind


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


class RetrievedChunk(BaseModel):
    content: str
    score: float
    source: str | None = None
    page: int | None = None
    type: str | None = None
    keywords: list[str] | None = None
    collection: str | None = None
    kind: ContentKind | None = None


class RetrieveResponse(BaseModel):
    query: str
    expanded_queries: list[str]
    documents: list[RetrievedChunk]


class ContextChunk(RetrievedChunk):
    rank: int


class AskResponse(BaseModel):
    query: str
    expanded_queries: list[str]
    answer: str
    model: str
    context: list[ContextChunk]


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    standalone_query: str
    model: str
    context: list[ContextChunk]


class ConversationMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: list[ConversationMessage]


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
    #: Where to poll the job over HTTP; absent when the engine runs in-process (use ``jobs.get``).
    status_url: str | None = None
    document_id: str | None = None
    total_pages: int | None = None
    warnings: list[str] = Field(default_factory=list)


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


# --- chat stream events (server-sent events: ``event: <name>`` + JSON ``data``) --------------------------
class ChatStartEvent(BaseModel):
    """First event: the conversation id and the sources the answer will be written from."""

    conversation_id: str
    standalone_query: str
    expanded_queries: list[str]
    model: str
    context: list[ContextChunk]


class ChatDeltaEvent(BaseModel):
    text: str


class ChatEndEvent(BaseModel):
    answer: str


ChatStreamEvent = ChatStartEvent | ChatDeltaEvent | ChatEndEvent
STREAM_EVENT_NAMES: dict[str, type[ChatStartEvent] | type[ChatDeltaEvent] | type[ChatEndEvent]] = {
    "start": ChatStartEvent,
    "delta": ChatDeltaEvent,
    "end": ChatEndEvent,
}


class ErrorBody(BaseModel):
    """The JSON body of every error response (and of a mid-stream ``error`` event)."""

    detail: str
    code: str
