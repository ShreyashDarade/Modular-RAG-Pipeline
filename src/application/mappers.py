"""Domain objects -> wire models. The only place that knows both."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.chat.service import ChatDelta, ChatEvent, ChatFinished, ChatStarted
from src.contracts.models import (
    ChatDeltaEvent,
    ChatEndEvent,
    ChatStartEvent,
    ChatStreamEvent,
    CollectionInfo,
    ContextChunk,
    DocumentInfo,
    IngestResponse,
    JobResponse,
    ModelInfo,
    ModelsResponse,
    RetrievedChunk,
)
from src.core.specs import RagConfig
from src.core.types import DocumentRecord, JobRecord, RetrievedDocument
from src.models.registry import ModelRegistry


def chunk_of(doc: RetrievedDocument) -> RetrievedChunk:
    meta = doc.metadata
    return RetrievedChunk(
        content=doc.content,
        score=doc.final_score,
        source=meta.get("source"),
        page=meta.get("page"),
        type=meta.get("type") or meta.get("content_type"),
        keywords=meta.get("keywords"),
        collection=doc.collection,
        kind=doc.kind,
    )


def context_of(rank: int, doc: RetrievedDocument) -> ContextChunk:
    return ContextChunk(rank=rank, **chunk_of(doc).model_dump())


def ranked_context(documents: list[RetrievedDocument]) -> list[ContextChunk]:
    return [context_of(i, d) for i, d in enumerate(documents, start=1)]


def job_response(record: JobRecord) -> JobResponse:
    return JobResponse(
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


def ingest_response(record: JobRecord, status_url: str | None, source: str) -> IngestResponse:
    result: dict[str, Any] = record.result or {}
    return IngestResponse(
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


def collection_infos(config: RagConfig) -> list[CollectionInfo]:
    return [
        CollectionInfo(
            name=name,
            description=spec.description,
            embedding_model=spec.embedding_model,
            kinds=list(spec.kinds),
            parsers=list(spec.parsers) if spec.parsers is not None else None,
            indices=dict(spec.index_names().items()),
            default=name == config.default_collection,
        )
        for name, spec in sorted(config.collections.items())
    ]


def models_response(config: RagConfig, models: ModelRegistry) -> ModelsResponse:
    return ModelsResponse(
        chat=[
            ModelInfo(name=n, provider=s.provider, model=s.model, default=n == config.default_chat_model)
            for n, s in sorted(config.chat_models.items())
        ],
        embedding=[
            ModelInfo(name=n, provider=s.provider, model=s.model, dimensions=models.embedder(n).dimensions)
            for n, s in sorted(config.embedding_models.items())
        ],
    )


def document_info(record: DocumentRecord) -> DocumentInfo:
    return DocumentInfo(**{k: v for k, v in asdict(record).items() if k != "status"})


def stream_event(event: ChatEvent) -> ChatStreamEvent:
    match event:
        case ChatStarted():
            return ChatStartEvent(
                conversation_id=event.conversation_id,
                standalone_query=event.standalone_query,
                expanded_queries=event.expanded_queries,
                model=event.model,
                context=ranked_context(event.documents),
            )
        case ChatDelta():
            return ChatDeltaEvent(text=event.text)
        case ChatFinished():
            return ChatEndEvent(answer=event.answer)
    raise AssertionError(f"unhandled chat event {event!r}")
