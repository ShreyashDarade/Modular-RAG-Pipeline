from __future__ import annotations

from fastapi import APIRouter, Query

from src.api.deps import RateLimited, ServiceDep
from src.contracts.models import CollectionInfo, DeleteResponse, DocumentList, ModelsResponse

router = APIRouter(prefix="/api/v1", tags=["catalog"])


@router.get("/collections", response_model=list[CollectionInfo])
async def list_collections(service: ServiceDep):
    return await service.collections()


@router.get("/models", response_model=ModelsResponse)
async def list_models(service: ServiceDep):
    return await service.models()


@router.get("/documents", response_model=DocumentList)
async def list_documents(
    service: ServiceDep,
    collection: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Documents fully ingested into a collection (from the ingestion ledger)."""
    return await service.documents(collection, limit=limit, offset=offset)


@router.delete("/documents", response_model=DeleteResponse, dependencies=[RateLimited])
async def delete_documents(service: ServiceDep, source: str, collection: str | None = None):
    """Remove every chunk of one document (by its source path) from a collection."""
    return await service.delete_document(source, collection)
