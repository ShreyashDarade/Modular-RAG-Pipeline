"""Pure-ASGI request middleware (no ``BaseHTTPMiddleware`` overhead): request ids, admission
control (load shedding) and metrics."""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.core.logger import logger, request_id_var
from src.runtime.metrics import HTTP_IN_FLIGHT, HTTP_LATENCY, HTTP_REQUESTS, HTTP_SHED

_EXEMPT = {"/health", "/ready", "/metrics"}


class RequestContextMiddleware:
    """* assigns / propagates ``X-Request-ID`` (also in every log line of the request)
    * rejects new work with ``503 + Retry-After`` above ``max_in_flight`` concurrent requests,
      so an overloaded replica sheds load instead of queueing it into timeouts
    * records request count, latency and in-flight gauge by route *template* (bounded labels)
    """

    def __init__(self, app: ASGIApp, *, max_in_flight: int) -> None:
        self.app = app
        self._max = max_in_flight
        self._in_flight = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        request_id = headers.get(b"x-request-id", b"").decode()[:64] or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        method, path = scope["method"], scope["path"]
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            if self._max and self._in_flight >= self._max and path not in _EXEMPT:
                HTTP_SHED.inc()
                body = json.dumps(
                    {"detail": "server is at capacity, retry shortly", "code": "overloaded"}
                ).encode()
                status = 503
                await send_wrapper(
                    {
                        "type": "http.response.start",
                        "status": 503,
                        "headers": [(b"content-type", b"application/json"), (b"retry-after", b"1")],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
            self._in_flight += 1
            HTTP_IN_FLIGHT.inc()
            started = time.perf_counter()
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                self._in_flight -= 1
                HTTP_IN_FLIGHT.dec()
                route: Any = scope.get("route")
                template = getattr(route, "path", None) or "unmatched"
                HTTP_REQUESTS.labels(method, template, str(status)).inc()
                HTTP_LATENCY.labels(method, template).observe(time.perf_counter() - started)
                if path not in _EXEMPT:
                    logger.info(
                        "%s %s -> %s in %.0fms", method, path, status, (time.perf_counter() - started) * 1000
                    )
        finally:
            request_id_var.reset(token)


class _BodyTooLarge(Exception):
    pass


class MaxBodySizeMiddleware:
    """Refuse an oversized upload *before* the framework spools it to disk.

    ``DataStore.save`` enforces the limit while copying, but by then Starlette has already received
    and spooled the whole multipart body - a client could make the server buffer gigabytes. This
    rejects on ``Content-Length`` immediately and counts bytes for chunked bodies without one.
    ``max_bytes`` should be the upload limit plus a little multipart overhead.
    """

    def __init__(self, app: ASGIApp, *, path: str, max_bytes: int) -> None:
        self.app = app
        self._path = path
        self._max = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != self._path:
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self._max:
            await self._reject(send)
            return
        received = 0
        exceeded = False
        rejected = False

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            # Whatever the framework makes of the aborted read (FastAPI turns it into a generic
            # "error parsing the body" 400), the caller gets the accurate 413.
            nonlocal rejected
            if exceeded:
                if not rejected:
                    rejected = True
                    await self._reject(send)
                return
            await send(message)

        with contextlib.suppress(_BodyTooLarge):
            await self.app(scope, counting_receive, guarded_send)
        if exceeded and not rejected:
            await self._reject(send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps(
            {
                "detail": f"upload exceeds the {self._max / (1024 * 1024):g} MB request limit",
                "code": "payload_too_large",
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})
