"""The facade, the sync bridge and the deprecation tools - no services needed."""

from __future__ import annotations

import asyncio
import inspect
import io
import threading
import warnings

import pytest
from ai_rag_info import (
    EXPERIMENTAL,
    AsyncRagAPI,
    RagAPI,
    RagDeprecationWarning,
    RagFutureWarning,
    deprecated,
    experimental,
)
from ai_rag_info._backend import IngestOptions, Upload
from ai_rag_info._facade import AsyncRagAPI as Facade
from ai_rag_info._facade import open_upload
from ai_rag_info._sync import LoopThread
from ai_rag_info.errors import (
    InvalidRequestError,
    NotFoundError,
    ParseError,
    RequestTimeoutError,
    UsageError,
)
from ai_rag_info.models import JobResponse


class RecordingBackend:
    """Remembers what the facade handed it; returns canned values."""

    def __init__(self, jobs: list[JobResponse] | None = None) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.jobs = list(jobs or [])

    async def get_job(self, job_id: str) -> JobResponse:
        self.calls.append(("get_job", (job_id,)))
        return self.jobs.pop(0) if len(self.jobs) > 1 else self.jobs[0]

    async def ingest(self, upload: Upload, options: IngestOptions):
        self.calls.append(("ingest", (upload.filename, upload.stream.read(), options)))
        raise NotImplementedError

    async def retrieve(self, request):
        self.calls.append(("retrieve", (request,)))
        raise NotImplementedError

    async def chat(self, request):
        self.calls.append(("chat", (request,)))
        raise NotImplementedError

    async def delete_document(self, source, collection):
        self.calls.append(("delete_document", (source, collection)))
        raise NotImplementedError

    async def list_documents(self, collection, limit, offset):
        self.calls.append(("list_documents", (collection, limit, offset)))
        raise NotImplementedError

    async def aclose(self) -> None:
        self.calls.append(("aclose", ()))


def job(status: str, **kw) -> JobResponse:
    return JobResponse(
        job_id="j", status=status, collection="c", source="/s", attempts=1, created_at=1.0, **kw
    )


# --- request validation, once, in the facade ----------------------------------------------------------
async def test_bad_arguments_are_typed_errors_before_any_transport_is_touched():
    backend = RecordingBackend()
    api = Facade(backend)  # type: ignore[arg-type]
    with pytest.raises(InvalidRequestError, match="kinds"):
        await api.retrieve("q", kinds=["video"])  # type: ignore[list-item]
    with pytest.raises(InvalidRequestError, match="query"):
        await api.retrieve("")
    with pytest.raises(InvalidRequestError, match="message"):
        await api.chat.send("")
    with pytest.raises(InvalidRequestError, match="limit"):
        await api.documents.list(limit=501)
    with pytest.raises(InvalidRequestError, match="offset"):
        await api.documents.list(offset=-1)
    with pytest.raises(InvalidRequestError, match="source"):
        await api.documents.delete("")
    assert backend.calls == []


async def test_empty_scopes_mean_the_default_and_sequences_are_accepted():
    backend = RecordingBackend()
    api = Facade(backend)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        await api.retrieve("q", collections=(), kinds=("text",), sources=iter(["a"]) if False else ["a"])
    (request,) = backend.calls[0][1]
    assert request.collections is None and request.kinds == ["text"] and request.sources == ["a"]


def test_upload_sources_are_normalised(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text("hello")
    upload, owned = open_upload(path, None)
    assert (upload.filename, upload.stream.read(), owned) == ("report.txt", b"hello", True)
    upload.stream.close()
    upload, owned = open_upload(str(path), "renamed.txt")
    assert upload.filename == "renamed.txt" and owned
    upload.stream.close()
    upload, owned = open_upload(b"abc", "a.bin")
    assert (upload.stream.read(), owned) == (b"abc", True)
    stream = io.BytesIO(b"x")
    upload, owned = open_upload(stream, "s.txt")
    assert upload.stream is stream and not owned, "a caller's stream is never closed by the SDK"
    with pytest.raises(InvalidRequestError, match="filename"):
        open_upload(bytearray(b"x"), None)
    with pytest.raises(InvalidRequestError, match="filename"):
        open_upload(io.BytesIO(b"x"), None)
    with pytest.raises(InvalidRequestError, match="cannot ingest a int"):
        open_upload(42, None)  # type: ignore[arg-type]
    with pytest.raises(NotFoundError, match="file not found"):
        open_upload(tmp_path / "missing.txt", None)


async def test_ingest_closes_the_file_it_opened_even_when_the_transport_fails(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("data")
    opened: list = []
    real = type(path).open

    def tracking(self, *a, **k):
        handle = real(self, *a, **k)
        opened.append(handle)
        return handle

    backend = RecordingBackend()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(type(path), "open", tracking)
        with pytest.raises(NotImplementedError):
            await Facade(backend).documents.ingest(path, collection="beta", kinds=["text"], force=True)  # type: ignore[arg-type]
    assert opened and all(h.closed for h in opened)
    _, (name, body, options) = backend.calls[0]
    assert (name, body, options.collection, options.kinds, options.force) == (
        "a.txt",
        b"data",
        "beta",
        ("text",),
        True,
    )


# --- jobs.wait ----------------------------------------------------------------------------------------------
async def test_waiting_for_a_job_polls_until_it_succeeds():
    backend = RecordingBackend([job("queued"), job("running"), job("succeeded")])
    done = await Facade(backend).jobs.wait("j", poll_interval=0.001)  # type: ignore[arg-type]
    assert done.status == "succeeded" and [c[0] for c in backend.calls] == ["get_job"] * 3


async def test_waiting_for_a_failed_job_raises_its_own_error_with_the_job_id():
    backend = RecordingBackend(
        [job("running"), job("failed", error="cannot open PDF", error_code="parse_error")]
    )
    with pytest.raises(ParseError, match="cannot open PDF") as caught:
        await Facade(backend).jobs.wait("j", poll_interval=0.001)  # type: ignore[arg-type]
    assert caught.value.details["job_id"] == "j"


async def test_waiting_gives_up_with_a_timeout_error():
    with pytest.raises(RequestTimeoutError, match="still running after"):
        await Facade(RecordingBackend([job("running")])).jobs.wait("j", timeout=0.05, poll_interval=0.01)  # type: ignore[arg-type]


# --- the two interfaces cannot drift ---------------------------------------------------------------------
def _public(cls) -> dict[str, inspect.Signature]:
    return {
        name: inspect.signature(member)
        for name, member in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_") and name not in {"aclose", "close"}
    }


def _shape(signature: inspect.Signature) -> list[tuple[str, str, object]]:
    return [(p.name, p.kind.name, p.default) for p in signature.parameters.values() if p.name != "self"]


@pytest.mark.parametrize("pair", ["documents", "chat", "jobs", "collections", None])
def test_the_sync_interface_has_exactly_the_async_methods_with_the_same_arguments(pair):
    from ai_rag_info import _facade, _sync

    sync_cls, async_cls = (
        (RagAPI, AsyncRagAPI)
        if pair is None
        else (getattr(_sync, pair.capitalize()), getattr(_facade, f"Async{pair.capitalize()}"))
    )
    sync_methods, async_methods = _public(sync_cls), _public(async_cls)
    assert set(sync_methods) == set(async_methods), f"{sync_cls.__name__} and {async_cls.__name__} differ"
    for name in async_methods:
        assert _shape(sync_methods[name]) == _shape(async_methods[name]), f"{pair or 'root'}.{name} drifted"


# --- the sync bridge -------------------------------------------------------------------------------------------
def test_the_bridge_runs_coroutines_on_one_background_loop_and_closes_cleanly():
    bridge = LoopThread("test")
    try:

        async def where() -> int:
            return threading.get_ident()

        assert bridge.run(where()) == bridge.run(where()) != threading.get_ident()
    finally:
        bridge.close()
    bridge.close()  # idempotent
    with pytest.raises(UsageError, match="closed"):
        bridge.run(asyncio.sleep(0))


async def test_a_blocking_call_from_inside_a_running_loop_is_a_usage_error_not_a_stall():
    bridge = LoopThread("test")
    try:
        with pytest.raises(UsageError, match="running an event loop"):
            bridge.run(asyncio.sleep(0))
        with pytest.raises(UsageError):
            next(iter(bridge.iterate(_numbers())))
    finally:
        bridge.close()


async def _numbers():
    for i in range(3):
        yield i


def test_a_blocking_iterator_yields_in_order_and_closes_the_source_when_abandoned():
    bridge = LoopThread("test")
    closed = []

    async def source():
        try:
            for i in range(10):
                yield i
        finally:
            closed.append(True)

    try:
        assert list(bridge.iterate(_numbers())) == [0, 1, 2]
        for item in bridge.iterate(source()):
            if item == 2:
                break
        assert closed == [True], "breaking out of the loop closes the async generator"
    finally:
        bridge.close()


# --- stability tiers and deprecation -----------------------------------------------------------------------------
def test_deprecation_warns_with_metadata_and_points_at_the_caller():
    @deprecated(since="2.1", remove_in="3.0", alternative="new_thing()")
    def old_thing() -> int:
        """Does it."""
        return 7

    with pytest.warns(
        RagDeprecationWarning, match=r"old_thing is deprecated since 2\.1.*removed in 3\.0.*new_thing\(\)"
    ) as rec:
        assert old_thing() == 7
    assert rec[0].filename == __file__, "stacklevel points at the caller, not at the SDK"
    assert ".. deprecated:: 2.1" in old_thing.__doc__ and old_thing.__rag_deprecated__ == {
        "since": "2.1",
        "remove_in": "3.0",
    }
    assert issubclass(RagDeprecationWarning, DeprecationWarning) and issubclass(
        RagFutureWarning, FutureWarning
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"since": "2.1", "remove_in": "3.0"},  # no replacement and no reason
        {
            "since": "2.1",
            "remove_in": "2.4",
            "alternative": "x",
        },  # removal in a minor: the policy says a major
        {"since": "2.1", "remove_in": "soon", "alternative": "x"},
        {"since": "v2", "remove_in": "3.0", "alternative": "x"},
    ],
)
def test_a_malformed_deprecation_is_refused_when_it_is_written(kwargs):
    with pytest.raises(ValueError, match="deprecated"):
        deprecated(**kwargs)


def test_a_deprecation_may_state_a_reason_instead_of_a_replacement():
    @deprecated(since="2.1", remove_in="3.0", reason="the feature is gone")
    def f() -> None: ...

    with pytest.warns(RagDeprecationWarning, match="the feature is gone"):
        f()


def test_the_suite_treats_our_own_deprecation_warnings_as_errors():
    @deprecated(since="2.1", remove_in="3.0", alternative="g")
    def f() -> None: ...

    with warnings.catch_warnings():
        warnings.simplefilter("error", RagDeprecationWarning)
        with pytest.raises(RagDeprecationWarning):
            f()


def test_experimental_marks_and_registers_names():
    @experimental
    def shiny() -> None:
        """Does something new."""

    assert shiny.__rag_experimental__ is True and "Experimental" in (shiny.__doc__ or "")  # type: ignore[attr-defined]
    assert any(name.endswith("shiny") for name in EXPERIMENTAL)
    from ai_rag_info.embedded import AsyncRag, Rag

    assert AsyncRag.evaluate.__rag_experimental__ and Rag.evaluate.__rag_experimental__  # type: ignore[attr-defined]
