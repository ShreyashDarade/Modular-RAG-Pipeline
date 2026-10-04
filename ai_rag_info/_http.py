"""The HTTP transport: requests, retries, error mapping, SSE (internal).

* Errors are rebuilt from the response's ``code`` into the SDK's typed classes (a ``404 not_found`` becomes
  ``NotFoundError``), carrying ``request_id`` and any extra fields as ``details``.
* A response that is not the documented shape is a :class:`ResponseError` - never a best-effort object.
* Retries: a request that certainly never reached the server (connection refused, connect timeout) and a
  refusal before any work (429, 503) are retried for every operation. 408/502/504 and connection resets - the
  server may have done the work - are retried only for idempotent operations (reads, deletes, and ingestion,
  which is idempotent by content checksum), never for a chat turn (it appends to the conversation).
  A *read timeout* is never retried - the server may still be working, and a retry would repeat an LLM call
  or an upload - and surfaces as the same ``RequestTimeoutError`` the in-process engine raises.
  ``Retry-After`` is honoured; backoff is exponential with jitter.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator, Mapping
from typing import Any, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError
from src.contracts.models import (
    STREAM_EVENT_NAMES,
    AskRequest,
    AskResponse,
    ChatEndEvent,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    CollectionInfo,
    ConversationResponse,
    DeleteResponse,
    DocumentList,
    ErrorBody,
    IngestResponse,
    JobResponse,
    ModelsResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from src.core.errors import (
    ConnectionFailedError,
    RagError,
    RequestTimeoutError,
    ResponseError,
    error_from_code,
)

from ai_rag_info._backend import IngestOptions, Upload
from ai_rag_info._sse import LineSplitter, SSEEvent, SSEParser

M = TypeVar("M", bound=BaseModel)

#: Statuses that mean "retry might help" for any operation: the server refused before doing work.
_PRE_WORK = frozenset({429, 503})
#: Additional statuses that are safe to retry only when the operation is idempotent.
_IDEMPOTENT_ONLY = frozenset({408, 502, 504})
#: What an error response without a ``code`` means, by status.
_CODE_BY_STATUS = {
    400: "invalid_request",
    404: "not_found",
    408: "timeout",
    413: "payload_too_large",
    415: "unsupported_type",
    422: "invalid_request",
    429: "rate_limited",
    502: "upstream_error",
    503: "overloaded",
    504: "timeout",
}
_MAX_RETRY_AFTER = 30.0
#: The request was never sent: always safe to retry.
_NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
#: The request was sent and nothing came back in time: it may still be running. Never retried.
_NO_ANSWER = (httpx.ReadTimeout, httpx.WriteTimeout)

_COLLECTIONS = TypeAdapter(list[CollectionInfo])


def _backoff(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, _MAX_RETRY_AFTER)
    return float(min(8.0, 0.5 * 2**attempt) * random.uniform(0.5, 1.0))


def _seekable(stream: Any) -> bool:
    check = getattr(stream, "seekable", None)
    return bool(check()) if callable(check) else False


def _timed_out(exc: BaseException) -> RequestTimeoutError:
    return RequestTimeoutError(
        f"the server did not answer in time ({type(exc).__name__}); it may still be working"
    )


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except ValueError:
        return None


def error_from_response(response: httpx.Response) -> RagError:
    """The typed error for a non-success response."""
    status, request_id = response.status_code, response.headers.get("x-request-id")
    try:
        body: Any = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and isinstance(body.get("code"), str):
        extra = {k: v for k, v in body.items() if k not in ("detail", "code")}
        return error_from_code(
            body["code"],
            str(body.get("detail", "")),
            status=status,
            retry_after=int(r) if (r := _retry_after(response)) is not None else None,
            request_id=request_id,
            details=extra,
        )
    if isinstance(body, dict) and isinstance(
        body.get("detail"), list
    ):  # the framework's own validation error
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}"
            for e in body["detail"]
            if isinstance(e, dict)
        )
        return error_from_code(
            "invalid_request", f"invalid request: {problems}", status=status, request_id=request_id
        )
    code = _CODE_BY_STATUS.get(status, "internal_error")
    detail = body.get("detail") if isinstance(body, dict) else None
    return error_from_code(code, str(detail or f"HTTP {status}"), status=status, request_id=request_id)


class HttpBackend:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        *,
        max_retries: int,
        owns_client: bool,
    ) -> None:
        self._client = client
        self._base = base_url.rstrip("/")
        self._retries = max_retries
        self._owns = owns_client

    # --- plumbing --------------------------------------------------------------------------------
    async def _send(
        self,
        method: str,
        path: str,
        *,
        idempotent: bool,
        json_body: Any = None,
        data: Mapping[str, Any] | None = None,
        upload: Upload | None = None,
        params: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        accept: tuple[int, ...] = (200,),
        gone_ok: bool = False,
    ) -> httpx.Response:
        """``gone_ok``: a 404 on a *retry* means an earlier, unanswered attempt already did the delete."""
        rewindable = upload is None or _seekable(upload.stream)
        attempts = (self._retries if rewindable else 0) + 1
        last: BaseException | None = None
        for attempt in range(attempts):
            if upload is not None and attempt:
                upload.stream.seek(0)
            retry_after: float | None = None
            try:
                response = await self._client.request(
                    method,
                    self._base + path,
                    json=json_body,
                    data=data,
                    files={"file": (upload.filename, upload.stream)} if upload else None,
                    params=params,
                    timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
                )
            except _NEVER_SENT as exc:
                last = exc
            except _NO_ANSWER as exc:
                raise _timed_out(exc) from exc
            except httpx.TransportError as exc:  # a reset or protocol error: the server may have acted
                last = exc
                if not idempotent:
                    break
            else:
                if response.status_code in accept:
                    return response
                if gone_ok and attempt and response.status_code == 404:
                    return response
                retriable = response.status_code in _PRE_WORK or (
                    idempotent and response.status_code in _IDEMPOTENT_ONLY
                )
                if not retriable or attempt == attempts - 1:
                    raise error_from_response(response)
                retry_after = _retry_after(response)
                last = None
            if attempt < attempts - 1:
                await asyncio.sleep(_backoff(attempt, retry_after))
        raise ConnectionFailedError(f"cannot reach the server: {type(last).__name__}: {last}") from last

    @staticmethod
    def _parse(model: type[M], response: httpx.Response) -> M:
        try:
            return model.model_validate_json(response.content)
        except ValidationError as exc:
            raise ResponseError(
                f"the server's response is not a valid {model.__name__}: {_summarise(exc)}"
            ) from exc

    # --- operations ------------------------------------------------------------------------------
    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        r = await self._send(
            "POST", "/api/v1/retrieve", idempotent=True, json_body=request.model_dump(mode="json")
        )
        return self._parse(RetrieveResponse, r)

    async def ask(self, request: AskRequest) -> AskResponse:
        r = await self._send(
            "POST", "/api/v1/ask", idempotent=True, json_body=request.model_dump(mode="json")
        )
        return self._parse(AskResponse, r)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        r = await self._send(
            "POST", "/api/v1/chat", idempotent=False, json_body=request.model_dump(mode="json")
        )
        return self._parse(ChatResponse, r)

    async def _open_stream(self, url: str, body: Any) -> tuple[Any, httpx.Response]:
        """Open the SSE request, retrying only what is certainly safe: it never reached the server, or the
        server refused (429/503) before starting the turn. Returns the context manager and the response."""
        for attempt in range(self._retries + 1):
            manager = self._client.stream("POST", url, json=body)
            try:
                response = await manager.__aenter__()
            except _NEVER_SENT as exc:
                if attempt < self._retries:
                    await asyncio.sleep(_backoff(attempt, None))
                    continue
                raise ConnectionFailedError(f"cannot reach the server: {type(exc).__name__}: {exc}") from exc
            except _NO_ANSWER as exc:
                raise _timed_out(exc) from exc
            except httpx.TransportError as exc:
                raise ConnectionFailedError(f"the connection broke: {type(exc).__name__}: {exc}") from exc
            if response.status_code in _PRE_WORK and attempt < self._retries:
                await response.aread()
                await manager.__aexit__(None, None, None)
                await asyncio.sleep(_backoff(attempt, _retry_after(response)))
                continue
            return manager, response
        raise AssertionError("unreachable")  # the last attempt returns or raises

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        manager, response = await self._open_stream(
            self._base + "/api/v1/chat/stream", request.model_dump(mode="json")
        )
        try:
            if response.status_code != 200:
                await response.aread()
                raise error_from_response(response)
            request_id = response.headers.get("x-request-id")
            splitter, parser, ended = LineSplitter(), SSEParser(), False
            try:
                async for chunk in response.aiter_bytes():
                    for line in splitter.feed(chunk):
                        event = parser.feed(line)
                        if (decoded := _decode_event(event, request_id) if event else None) is not None:
                            ended = ended or isinstance(decoded, ChatEndEvent)
                            yield decoded
                for line in splitter.flush():
                    parser.feed(line)
                if (tail := parser.finish()) is not None and (decoded := _decode_event(tail, request_id)):
                    ended = ended or isinstance(decoded, ChatEndEvent)
                    yield decoded
            except UnicodeDecodeError as exc:
                raise ResponseError(f"the stream is not valid UTF-8: {exc}") from exc
            if not ended:
                raise ResponseError("the stream ended before its final event")
        except _NO_ANSWER as exc:
            raise _timed_out(exc) from exc
        except httpx.TransportError as exc:
            raise ConnectionFailedError(f"the connection broke: {type(exc).__name__}: {exc}") from exc
        finally:
            await manager.__aexit__(None, None, None)

    async def get_conversation(self, conversation_id: str) -> ConversationResponse:
        r = await self._send("GET", f"/api/v1/chat/{quote(conversation_id, safe='')}", idempotent=True)
        return self._parse(ConversationResponse, r)

    async def delete_conversation(self, conversation_id: str) -> None:
        await self._send(
            "DELETE",
            f"/api/v1/chat/{quote(conversation_id, safe='')}",
            idempotent=True,
            accept=(204, 200),
            gone_ok=True,
        )

    async def ingest(self, upload: Upload, options: IngestOptions) -> IngestResponse:
        data: dict[str, Any] = {}
        if options.collection:
            data["collection"] = options.collection
        if options.image_language:
            data["image_language"] = options.image_language
        if options.kinds:
            data["kinds"] = ",".join(options.kinds)
        params: dict[str, Any] = {"force": str(options.force).lower()}
        if options.wait is not None:
            params["wait"] = str(options.wait).lower()
        r = await self._send(
            "POST",
            "/api/v1/ingest",
            idempotent=True,  # re-sending the same file is a checksum no-op
            data=data,
            upload=upload,
            params=params,
            timeout=options.timeout,
            accept=(200, 202),
        )
        return self._parse(IngestResponse, r)

    async def get_job(self, job_id: str) -> JobResponse:
        r = await self._send("GET", f"/api/v1/jobs/{quote(job_id, safe='')}", idempotent=True)
        return self._parse(JobResponse, r)

    async def list_documents(self, collection: str | None, limit: int, offset: int) -> DocumentList:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if collection:
            params["collection"] = collection
        r = await self._send("GET", "/api/v1/documents", idempotent=True, params=params)
        return self._parse(DocumentList, r)

    async def delete_document(self, source: str, collection: str | None) -> DeleteResponse:
        params: dict[str, Any] = {"source": source}
        if collection:
            params["collection"] = collection
        r = await self._send("DELETE", "/api/v1/documents", idempotent=True, params=params)
        return self._parse(DeleteResponse, r)

    async def list_collections(self) -> list[CollectionInfo]:
        r = await self._send("GET", "/api/v1/collections", idempotent=True)
        try:
            return _COLLECTIONS.validate_json(r.content)
        except ValidationError as exc:
            raise ResponseError(
                f"the server's response is not a valid collection list: {_summarise(exc)}"
            ) from exc

    async def list_models(self) -> ModelsResponse:
        r = await self._send("GET", "/api/v1/models", idempotent=True)
        return self._parse(ModelsResponse, r)

    async def aclose(self) -> None:
        if self._owns:
            await self._client.aclose()


def _summarise(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:3])


def _decode_event(event: SSEEvent, request_id: str | None = None) -> ChatStreamEvent | None:
    """A stream event as its model. ``error`` raises the typed error; an event name this SDK does not know is
    ignored (a newer server may add some)."""
    if event.event == "error":
        try:
            body = ErrorBody.model_validate_json(event.data)
        except ValidationError as exc:
            raise ResponseError(f"malformed error event: {_summarise(exc)}") from exc
        raise error_from_code(body.code, body.detail, request_id=request_id)
    model = STREAM_EVENT_NAMES.get(event.event)
    if model is None:
        return None
    try:
        return model.model_validate(json.loads(event.data))
    except (ValueError, ValidationError) as exc:
        raise ResponseError(f"malformed '{event.event}' event: {exc}") from exc
