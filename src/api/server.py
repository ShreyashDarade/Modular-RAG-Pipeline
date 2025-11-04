from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from src.api.schemas import (
    AskContextItem,
    AskRequest,
    AskResponseSchema,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedDocumentSchema,
)
from src.core.config import settings
from src.core.logger import logger
from src.pipelines.ask import AskPipeline
from src.pipelines.ingestion import IngestionPipeline
from src.pipelines.retrieval import RetrievalPipeline
from src.services.reindexer import DataDirectoryWatcher


app = FastAPI(title="RAG APIs", version="0.1.0")

_ingestion = IngestionPipeline()
_retrieval = RetrievalPipeline()
_ask = AskPipeline()
_watcher = DataDirectoryWatcher()


def _reindex_callback(path: Path) -> None:
    if path.suffix.lower() not in settings.allowed_file_extensions:
        return
    logger.info("Watcher triggered reindex for %s", path)
    _ingestion.ingest_path(path, force=True)


@app.on_event("startup")
def startup_event() -> None:
    logger.info("Starting background watcher on %s", settings.data_dir)
    _watcher.start(_reindex_callback)


@app.on_event("shutdown")
def shutdown_event() -> None:
    logger.info("Shutting down background watcher")
    _watcher.stop()


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    force: bool = False,
    image_language: str | None = Form(None),
):
    file.file.seek(0)
    summary = await run_in_threadpool(_ingestion.ingest_upload, file, force, image_language)
    return IngestResponse(
        source=str(summary.source),
        text_chunks=summary.text_chunks,
        table_chunks=summary.table_chunks,
        image_chunks=summary.image_chunks,
        skipped_reason=summary.skipped_reason,
        reindexed=summary.reindexed,
    )


@app.post("/api/v1/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(payload: RetrieveRequest):
    result = await run_in_threadpool(_retrieval.retrieve, payload.query)
    documents = [
        RetrievedDocumentSchema(
            content=hit.document.page_content,
            score=hit.score,
            source=hit.document.metadata.get("source"),
            page=hit.document.metadata.get("page"),
            type=hit.document.metadata.get("type"),
            keywords=hit.document.metadata.get("keywords"),
        )
        for hit in result.documents
    ]
    return RetrieveResponse(
        query=payload.query,
        expanded_queries=result.expanded_queries,
        documents=documents,
    )


@app.post("/api/v1/ask", response_model=AskResponseSchema)
async def ask_question(payload: AskRequest):
    response = await run_in_threadpool(_ask.ask, payload.query)
    context_items = [
        AskContextItem(
            rank=item["rank"],
            score=item["score"],
            source=item["source"],
            page=item["page"],
            type=item["type"],
            keywords=item["keywords"],
            content=item["content"],
        )
        for item in response.context
    ]
    return AskResponseSchema(
        query=response.query,
        expanded_queries=response.expanded_queries,
        answer=response.answer,
        context=context_items,
    )


__all__ = ["app"]
