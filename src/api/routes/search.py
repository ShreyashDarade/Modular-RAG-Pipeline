from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import ContainerDep, RateLimited, deadline
from src.api.schemas import (
    AskContextItem,
    AskRequest,
    AskResponseSchema,
    RetrievedDocumentSchema,
    RetrieveRequest,
    RetrieveResponse,
)

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/retrieve", response_model=RetrieveResponse, dependencies=[RateLimited])
async def retrieve_documents(payload: RetrieveRequest, container: ContainerDep):
    """Hybrid (BM25 + vector) search over the selected collections and content kinds."""
    scope = container.retrieval.scope(payload.collections, payload.kinds, payload.sources)
    async with deadline(container):
        result = await container.retrieval.retrieve(payload.query, scope)
    return RetrieveResponse(
        query=result.query,
        expanded_queries=result.expanded_queries,
        documents=[RetrievedDocumentSchema.of(d) for d in result.documents],
    )


@router.post("/ask", response_model=AskResponseSchema, dependencies=[RateLimited])
async def ask_question(payload: AskRequest, container: ContainerDep):
    """Answer a single question from retrieved context with the chosen chat model."""
    scope = container.retrieval.scope(payload.collections, payload.kinds, payload.sources)
    async with deadline(container):
        result = await container.answers.ask(payload.query, scope, payload.model)
    return AskResponseSchema(
        query=result.query,
        expanded_queries=result.expanded_queries,
        answer=result.answer,
        model=result.model,
        context=[AskContextItem.ranked(i, d) for i, d in enumerate(result.documents, start=1)],
    )
