"""FastAPI application factory.

``create_app`` wires nothing itself: the lifespan asks the composition root
(:class:`src.core.container.Container`) for the collaborators and stores them on ``app.state``.
Tests can pass a prebuilt container; ``uvicorn src.api.server:app`` uses the module-level app.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.errors import install_error_handlers
from src.api.middleware import MaxBodySizeMiddleware, RequestContextMiddleware
from src.api.routes import catalog, chat, ingest, ops, search
from src.api.version import VERSION
from src.application import RagService
from src.core.config import Settings, get_settings
from src.core.container import Container
from src.core.logger import configure_logging, logger


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or (container.settings if container else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_format)
        owned = container is None
        active = container or await Container.build(settings, role="api")
        if settings.web_concurrency > 1 and settings.ingest_backend == "inprocess":
            logger.warning(
                "WEB_CONCURRENCY=%d with INGEST_BACKEND=inprocess: job status is per process. "
                "Use INGEST_BACKEND=redis (or one process per container).",
                settings.web_concurrency,
            )
        try:
            await active.start()
        except BaseException:
            if owned:
                await active.close()
            raise
        app.state.container = active
        app.state.service = RagService(active)
        stop = asyncio.Event()
        worker = active.start_embedded_worker(stop) if active.ingestion else None
        await active.start_watchers()
        logger.info(
            "ready: collections=%s chat_models=%s ingest=%s%s",
            sorted(active.config.collections),
            active.models.chat_names(),
            settings.ingest_backend,
            " (embedded worker)" if worker else " (queue only)",
        )
        mcp_lifespan = app.state.mcp_lifespan if hasattr(app.state, "mcp_lifespan") else None
        try:
            async with mcp_lifespan() if mcp_lifespan else contextlib.nullcontext():
                yield
        finally:
            stop.set()
            if worker is not None:
                try:
                    await asyncio.wait_for(worker, settings.shutdown_grace_seconds)
                except TimeoutError:
                    logger.error(
                        "ingestion workers did not drain within %ss", settings.shutdown_grace_seconds
                    )
                    worker.cancel()
                except Exception as exc:  # the worker died earlier; cleanup below must still run
                    logger.error("ingestion worker had stopped with an error", exc_info=exc)
            if owned:
                await active.close()

    app = FastAPI(
        title="RAG-OCR Pipeline API",
        version=VERSION,
        description="Modular multilingual RAG: multi-format ingestion with OCR, selective multi-collection "
        "hybrid retrieval, multi-model chat.",
        lifespan=lifespan,
    )
    install_error_handlers(app)
    origins = settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
    )
    app.add_middleware(
        MaxBodySizeMiddleware,
        path="/api/v1/ingest",
        max_bytes=settings.max_upload_bytes + 1024 * 1024,  # + multipart overhead
    )
    app.add_middleware(RequestContextMiddleware, max_in_flight=settings.max_concurrent_requests)
    for router in (ops.router, ingest.router, search.router, chat.router, catalog.router):
        app.include_router(router)
    if settings.mcp_enabled:
        from src.mcp_server.server import mount_mcp

        mount_mcp(app, settings)
    if container is not None:
        app.state.container = container  # visible before lifespan for tests that skip it
        app.state.service = RagService(container)
    return app


app = create_app()

__all__ = ["app", "create_app"]
