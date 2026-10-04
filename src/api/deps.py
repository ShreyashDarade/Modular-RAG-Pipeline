from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, Request, Response

from src.core.container import Container
from src.core.errors import RateLimitedError
from src.runtime.concurrency import deadline as request_deadline
from src.runtime.metrics import RATE_LIMITED


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


async def rate_limit(request: Request, response: Response, container: ContainerDep) -> None:
    """Per-client, per-route fixed-window limit. Backend failures propagate (no fail-open)."""
    limiter = container.rate_limiter
    if limiter is None:
        return
    route = getattr(request.scope.get("route"), "path", request.url.path)
    client = request.client.host if request.client else "unknown"
    limit = container.settings.rate_limit_per_minute
    decision = await limiter.hit(f"{client}|{route}", limit, 60)
    response.headers["X-RateLimit-Limit"] = str(decision.limit)
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
    if not decision.allowed:
        RATE_LIMITED.labels(route).inc()
        raise RateLimitedError(f"rate limit of {limit}/minute exceeded", retry_after=decision.retry_after)


RateLimited = Depends(rate_limit)


@asynccontextmanager
async def deadline(container: Container) -> AsyncIterator[None]:
    """Bound a request's total time to ``REQUEST_TIMEOUT_SECONDS`` (504 beyond that)."""
    async with request_deadline(container.settings.request_timeout_seconds):
        yield
