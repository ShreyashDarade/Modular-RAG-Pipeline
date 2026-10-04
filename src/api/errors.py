from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.errors import OverloadedError, RagError, RateLimitedError
from src.core.logger import logger


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RagError)
    async def rag_error(request: Request, exc: RagError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("%s on %s %s", type(exc).__name__, request.method, request.url.path, exc_info=exc)
        headers = {}
        if isinstance(exc, RateLimitedError | OverloadedError):
            headers["Retry-After"] = str(exc.retry_after)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.public_message, "code": exc.code},
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path, exc_info=exc
        )
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error", "code": "internal_error"}
        )
