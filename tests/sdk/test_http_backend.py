"""The HTTP transport on its own: retries, error mapping, response validation, SSE - against a mock server."""

from __future__ import annotations

import io
import json

import httpx
import pytest
from turinton_rag import AsyncRagClient
from turinton_rag._http import error_from_response
from turinton_rag._sse import SSEParser
from turinton_rag.errors import (
    ConnectionFailedError,
    InvalidRequestError,
    NotFoundError,
    OverloadedError,
    RagError,
    RateLimitedError,
    ResponseError,
    UpstreamError,
)

RETRIEVE_OK = {"query": "q", "expanded_queries": ["q"], "documents": []}
CHAT_OK = {"conversation_id": "c1", "answer": "a", "standalone_query": "q", "model": "m", "context": []}
INGEST_OK = {
    "source": "/d/a.txt",
    "collection": "alpha",
    "text_chunks": 1,
    "table_chunks": 0,
    "image_chunks": 0,
    "job_id": "j1",
    "status": "succeeded",
    "status_url": "http://x/jobs/j1",
}


def client(handler, *, retries: int = 2, **kwargs) -> AsyncRagClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://rag.test")
    return AsyncRagClient("http://rag.test", http_client=http, max_retries=retries, **kwargs)


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    sleeps: list[float] = []

    async def fake(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("turinton_rag._http.asyncio.sleep", fake)
    return sleeps


def sse(*events: tuple[str, dict | str]) -> bytes:
    return b"".join(
        f"event: {name}\ndata: {data if isinstance(data, str) else json.dumps(data)}\n\n".encode()
        for name, data in events
    )


START = {
    "conversation_id": "c1",
    "standalone_query": "q",
    "expanded_queries": ["q"],
    "model": "m",
    "context": [],
}


# --- retries -----------------------------------------------------------------------------------------
async def test_a_rate_limited_read_waits_for_retry_after_then_succeeds(no_sleeping):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                429, json={"detail": "slow down", "code": "rate_limited"}, headers={"Retry-After": "7"}
            )
        return httpx.Response(200, json=RETRIEVE_OK)

    result = await client(handler).retrieve("q")
    assert result.query == "q" and len(calls) == 2 and no_sleeping == [7.0]


async def test_server_errors_are_retried_for_reads_with_backoff_then_raise_typed(no_sleeping):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"detail": "a backing service failed", "code": "upstream_error"})

    with pytest.raises(UpstreamError) as caught:
        await client(handler, retries=2).retrieve("q")
    assert caught.value.code == "upstream_error" and caught.value.status_code == 502
    assert len(no_sleeping) == 2 and all(0 < s <= 8 for s in no_sleeping), "two retries, backing off"


async def test_a_chat_turn_is_not_retried_after_a_gateway_error_but_is_after_a_503():
    """A 502/504 may come after the server appended the turn; a 503/429 is a refusal before any work."""
    calls = []

    def bad_gateway(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(502, json={"detail": "x", "code": "upstream_error"})

    with pytest.raises(UpstreamError):
        await client(bad_gateway, retries=3).chat.send("hello")
    assert calls == [1], "not retried: it might have been recorded already"

    calls.clear()

    def overloaded_then_ok(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(
                503, json={"detail": "busy", "code": "overloaded"}, headers={"Retry-After": "1"}
            )
        return httpx.Response(200, json=CHAT_OK)

    assert (await client(overloaded_then_ok, retries=3).chat.send("hello")).conversation_id == "c1"
    assert len(calls) == 3


async def test_client_errors_are_never_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json={"detail": "unknown collection 'x'", "code": "not_found"})

    with pytest.raises(NotFoundError, match="unknown collection"):
        await client(handler, retries=5).retrieve("q", collections=["x"])
    assert calls == [1]


async def test_connection_failures_retry_then_become_a_typed_error(no_sleeping):
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ConnectionFailedError, match="cannot reach the server") as caught:
        await client(handler, retries=2).retrieve("q")
    assert len(attempts) == 3 and isinstance(caught.value.__cause__, httpx.ConnectError)


async def test_a_read_timeout_is_retried_for_reads_but_not_for_a_chat_turn():
    reads, chats = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        (chats if request.url.path.endswith("/chat") else reads).append(1)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ConnectionFailedError):
        await client(handler, retries=2).retrieve("q")
    with pytest.raises(ConnectionFailedError):
        await client(handler, retries=2).chat.send("hi")
    assert len(reads) == 3 and len(chats) == 1


async def test_an_upload_is_rewound_between_attempts():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(503, json={"detail": "busy", "code": "overloaded"})
        return httpx.Response(200, json=INGEST_OK)

    result = await client(handler).documents.ingest(io.BytesIO(b"hello world"), filename="a.txt")
    assert result.status == "succeeded" and len(bodies) == 2
    assert all(b"hello world" in b for b in bodies), "the retry sent the whole file, not an empty one"


async def test_an_unseekable_upload_gets_one_attempt_only():
    class OneShot(io.RawIOBase):
        def readable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return False

        def readinto(self, buffer) -> int:
            return 0

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"detail": "busy", "code": "overloaded"})

    with pytest.raises(OverloadedError):
        await client(handler, retries=3).documents.ingest(OneShot(), filename="a.txt")
    assert calls == [1]


# --- error mapping ---------------------------------------------------------------------------------------
def response(status: int, body=None, *, headers=None, content: bytes | None = None) -> httpx.Response:
    if content is not None:
        return httpx.Response(status, content=content, headers=headers)
    return httpx.Response(status, json=body, headers=headers)


def test_a_known_code_becomes_its_class_with_status_request_id_and_extra_fields():
    err = error_from_response(
        response(
            429,
            {"detail": "limit", "code": "rate_limited", "job_id": "j9"},
            headers={"Retry-After": "12", "X-Request-ID": "req-1"},
        )
    )
    assert isinstance(err, RateLimitedError) and err.retry_after == 12 and err.status_code == 429
    assert err.request_id == "req-1" and dict(err.details) == {"job_id": "j9"} and str(err) == "limit"


def test_an_unknown_code_from_a_newer_server_is_kept_not_misreported():
    err = error_from_response(response(418, {"detail": "teapot", "code": "short_and_stout"}))
    assert type(err) is RagError and err.code == "short_and_stout" and err.status_code == 418


def test_the_frameworks_own_validation_errors_become_invalid_request():
    body = {"detail": [{"loc": ["body", "query"], "msg": "String should have at least 1 character"}]}
    err = error_from_response(response(422, body))
    assert isinstance(err, InvalidRequestError) and "body.query" in str(err) and err.code == "invalid_request"


@pytest.mark.parametrize(
    ("status", "content", "code"),
    [
        (502, b"<html>Bad Gateway</html>", "upstream_error"),
        (503, b"", "overloaded"),
        (504, b"upstream timed out", "timeout"),
        (500, b"boom", "internal_error"),
    ],
)
def test_proxy_errors_without_a_json_body_are_still_typed(status, content, code):
    err = error_from_response(response(status, content=content))
    assert err.code == code and err.status_code == status


# --- response validation -----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [{"query": "q"}, {"query": "q", "expanded_queries": "no", "documents": []}, [], "text"],
)
async def test_a_malformed_success_response_is_a_response_error_not_a_best_effort_object(payload):
    with pytest.raises(ResponseError, match="not a valid RetrieveResponse"):
        await client(lambda request: httpx.Response(200, json=payload)).retrieve("q")


async def test_a_non_json_success_body_is_a_response_error():
    with pytest.raises(ResponseError):
        await client(lambda request: httpx.Response(200, content=b"<html>")).retrieve("q")


async def test_unknown_response_fields_are_ignored_so_a_newer_server_does_not_break_us():
    payload = {**RETRIEVE_OK, "brand_new_field": {"x": 1}}
    result = await client(lambda request: httpx.Response(200, json=payload)).retrieve("q")
    assert result.query == "q" and not hasattr(result, "brand_new_field")


# --- request shape and options -----------------------------------------------------------------------------
async def test_requests_carry_auth_headers_a_user_agent_and_the_documented_body():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=RETRIEVE_OK)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    # headers/api_key are applied when the SDK builds the client itself; with a supplied client they are the caller's
    c = AsyncRagClient("http://rag.test", http_client=http, max_retries=0)
    await c.retrieve("q", collections=["a"], kinds=["text"], sources=["/x"])
    assert json.loads(seen[0].content) == {
        "query": "q",
        "collections": ["a"],
        "kinds": ["text"],
        "sources": ["/x"],
    }

    from turinton_rag.client import _build_http_client

    built = _build_http_client(
        "http://h/", "sekret", {"X-Team": "a"}, None, 5.0, httpx.MockTransport(handler)
    )
    await built.get("http://h/ping")
    assert seen[-1].headers["authorization"] == "Bearer sekret" and seen[-1].headers["x-team"] == "a"
    assert seen[-1].headers["user-agent"].startswith("turinton-rag-python/")
    assert str(seen[0].url) == "http://rag.test/api/v1/retrieve", (
        "URLs are absolute: a supplied client needs no base_url"
    )


async def test_ingest_sends_the_form_fields_and_follow_up_flags():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202, json={**INGEST_OK, "status": "running"})

    result = await client(handler).documents.ingest(
        b"data",
        filename="a.txt",
        collection="beta",
        kinds=["text", "table"],
        image_language="hi",
        force=True,
        wait=False,
    )
    assert result.status == "running", "a 202 is a success: the job is still running"
    body = seen[0].content
    assert (
        b'name="collection"' in body
        and b"beta" in body
        and b"text,table" in body
        and b'name="image_language"' in body
    )
    assert seen[0].url.params["force"] == "true" and seen[0].url.params["wait"] == "false"


async def test_negative_retries_are_refused():
    with pytest.raises(ValueError):
        AsyncRagClient("http://x", max_retries=-1)


# --- streaming -------------------------------------------------------------------------------------------------
async def collect(c: AsyncRagClient, **kwargs):
    return [e async for e in c.chat.stream("hi", **kwargs)]


async def test_a_stream_is_decoded_into_events():
    body = sse(
        ("start", START), ("delta", {"text": "Hel"}), ("delta", {"text": "lo"}), ("end", {"answer": "Hello"})
    )
    events = await collect(client(lambda request: httpx.Response(200, content=body)))
    assert [type(e).__name__ for e in events] == [
        "ChatStartEvent",
        "ChatDeltaEvent",
        "ChatDeltaEvent",
        "ChatEndEvent",
    ]
    assert events[0].conversation_id == "c1" and events[-1].answer == "Hello"


async def test_a_stream_delivered_in_arbitrary_chunks_decodes_the_same():
    body = sse(("start", START), ("delta", {"text": "é€"}), ("end", {"answer": "é€"}))

    class Chunked(httpx.AsyncByteStream):
        async def __aiter__(self):
            for i in range(0, len(body), 5):  # splits lines and even multi-byte characters mid-way
                yield body[i : i + 5]

    events = await collect(client(lambda request: httpx.Response(200, stream=Chunked())))
    assert events[1].text == "é€" and events[-1].answer == "é€"


async def test_a_mid_stream_error_event_raises_the_typed_error():
    body = sse(
        ("start", START),
        ("delta", {"text": "par"}),
        ("error", {"detail": "request did not finish within 60s", "code": "timeout"}),
    )
    seen = []
    with pytest.raises(RagError) as caught:
        async for event in client(lambda request: httpx.Response(200, content=body)).chat.stream("hi"):
            seen.append(event)
    assert caught.value.code == "timeout" and len(seen) == 2


async def test_a_stream_that_ends_without_its_final_event_is_an_error():
    body = sse(("start", START), ("delta", {"text": "par"}))
    with pytest.raises(ResponseError, match="ended before its final event"):
        await collect(client(lambda request: httpx.Response(200, content=body)))


async def test_unknown_stream_events_are_ignored_and_a_malformed_known_one_is_not():
    ok = sse(("start", START), ("keepalive", {"x": 1}), ("end", {"answer": "a"}))
    assert len(await collect(client(lambda request: httpx.Response(200, content=ok)))) == 2
    bad = sse(("start", START), ("delta", "{not json"), ("end", {"answer": "a"}))
    with pytest.raises(ResponseError, match="malformed 'delta' event"):
        await collect(client(lambda request: httpx.Response(200, content=bad)))


async def test_an_http_error_before_the_stream_starts_is_raised_with_its_type():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "conversation 'x' not found", "code": "not_found"})

    with pytest.raises(NotFoundError):
        await collect(client(handler), conversation_id="x")


# --- the SSE parser --------------------------------------------------------------------------------------------
def parse(text: str) -> list[tuple[str, str]]:
    parser, out = SSEParser(), []
    for line in text.splitlines():
        if (event := parser.feed(line)) is not None:
            out.append((event.event, event.data))
    if (tail := parser.finish()) is not None:
        out.append((tail.event, tail.data))
    return out


def test_sse_parsing_follows_the_spec():
    assert parse("event: a\ndata: 1\n\nevent: b\ndata: 2\n\n") == [("a", "1"), ("b", "2")]
    assert parse("data: one\ndata: two\n\n") == [("message", "one\ntwo")], (
        "multi-line data; default event name"
    )
    assert parse(": keep-alive\nevent: a\nid: 7\nretry: 100\ndata:x\n\n") == [("a", "x")], (
        "comments, id, retry ignored"
    )
    assert parse("\n\n\n") == [], "blank lines alone are not events"
    assert parse("event: a\ndata: tail without blank line") == [("a", "tail without blank line")]
