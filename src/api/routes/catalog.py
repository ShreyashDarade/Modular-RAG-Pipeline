from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Query

from src.api.deps import ContainerDep, RateLimited
from src.api.schemas import (
    CollectionInfo,
    DeleteResponse,
    DocumentInfo,
    DocumentList,
    ModelInfo,
    ModelsResponse,
)

router = APIRouter(prefix="/api/v1", tags=["catalog"])


@router.get("/collections", response_model=list[CollectionInfo])
async def list_collections(container: ContainerDep):
    config = container.config
    return [
        CollectionInfo(
            name=name,
            description=spec.description,
            embedding_model=spec.embedding_model,
            kinds=list(spec.kinds),
            parsers=list(spec.parsers) if spec.parsers is not None else None,
            indices={kind: index for kind, index in spec.index_names().items()},
            default=name == config.default_collection,
        )
        for name, spec in sorted(config.collections.items())
    ]


@router.get("/models", response_model=ModelsResponse)
async def list_models(container: ContainerDep):
    config = container.config
    return ModelsResponse(
        chat=[
            ModelInfo(name=n, provider=s.provider, model=s.model, default=n == config.default_chat_model)
            for n, s in sorted(config.chat_models.items())
        ],
        embedding=[
            ModelInfo(
                name=n, provider=s.provider, model=s.model, dimensions=container.models.embedder(n).dimensions
            )
            for n, s in sorted(config.embedding_models.items())
        ],
    )


@router.get("/documents", response_model=DocumentList)
async def list_documents(
    container: ContainerDep,
    collection: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Documents fully ingested into a collection (from the ingestion ledger)."""
    name = collection or container.config.default_collection
    records, total = await container.documents.list(name, limit=limit, offset=offset)
    return DocumentList(
        collection=name,
        total=total,
        limit=limit,
        offset=offset,
        documents=[DocumentInfo(**{k: v for k, v in asdict(r).items() if k != "status"}) for r in records],
    )


@router.delete("/documents", response_model=DeleteResponse, dependencies=[RateLimited])
async def delete_documents(container: ContainerDep, source: str, collection: str | None = None):
    """Remove every chunk of one document (by its source path) from a collection."""
    name = collection or container.config.default_collection
    deleted = await container.documents.delete(name, source)
    return DeleteResponse(success=True, source=source, collection=name, deleted_count=deleted)
