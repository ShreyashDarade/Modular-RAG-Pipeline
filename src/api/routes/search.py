from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import RateLimited, ServiceDep
from src.contracts.models import AskRequest, AskResponse, RetrieveRequest, RetrieveResponse

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/retrieve", response_model=RetrieveResponse, dependencies=[RateLimited])
async def retrieve_documents(payload: RetrieveRequest, service: ServiceDep):
    """Hybrid (BM25 + vector) search over the selected collections and content kinds."""
    return await service.retrieve(payload)


@router.post("/ask", response_model=AskResponse, dependencies=[RateLimited])
async def ask_question(payload: AskRequest, service: ServiceDep):
    """Answer a single question from retrieved context with the chosen chat model."""
    return await service.ask(payload)
