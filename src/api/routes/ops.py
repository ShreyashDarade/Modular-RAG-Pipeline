from __future__ import annotations

import time

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.deps import ContainerDep
from src.api.version import VERSION
from src.core.errors import NotFoundError

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health(container: ContainerDep):
    """Liveness: the process is up and serving. Checks no dependency (use /ready for that)."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": VERSION,
        "environment": container.settings.environment,
    }


@router.get("/ready")
async def ready(container: ContainerDep):
    """Readiness: every dependency answers. 503 otherwise, so load balancers stop routing here."""
    checks = await container.readiness()
    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(status_code=200 if all_ok else 503, content={"ready": all_ok, "checks": checks})


@router.get("/metrics")
async def metrics(container: ContainerDep):
    if not container.settings.metrics_enabled:
        raise NotFoundError("metrics are disabled")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/api/v1/status")
async def status(container: ContainerDep):
    """Effective configuration (no secrets)."""
    s, config = container.settings, container.config
    return {
        "app_name": s.app_name,
        "environment": s.environment,
        "version": VERSION,
        "default_collection": config.default_collection,
        "collections": sorted(config.collections),
        "chat_models": sorted(config.chat_models),
        "embedding_models": sorted(config.embedding_models),
        "parsers": container.parsers.names(),
        "retrieval": {
            "query_expander": config.query_expander,
            "reranker": {
                "name": config.reranker,
                "provider": config.reranker_spec().provider,
                "model": config.reranker_spec().model or None,
                "candidates": s.rerank_candidates,
            },
            "top_k": s.retriever_top_k,
            "rerank_top_k": s.rerank_top_k,
            "cross_references": s.enable_cross_references,
        },
        "runtime": {
            "cache": s.cache_backend,
            "rate_limit": f"{s.rate_limit_per_minute}/minute ({s.rate_limit_backend})"
            if s.rate_limit_per_minute
            else "off",
            "ingest_backend": s.ingest_backend,
            "embedded_worker": container.ingestion is not None,
            "chat_store": s.chat_store,
            "ocr": {"enabled": s.ocr_enabled, "engine": s.ocr_engine, "languages": s.supported_ocr_languages},
        },
    }
