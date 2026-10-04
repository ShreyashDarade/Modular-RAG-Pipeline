"""One suite, two transports. Each test body runs against the embedded engine and against the HTTP client
(through the real FastAPI app); neither mode may behave differently from the other except where a test says so."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from turinton_rag import AsyncRag, AsyncRagAPI, AsyncRagClient
from turinton_rag.errors import (
    InvalidRequestError,
    NotFoundError,
    ParseError,
    RagError,
    UnsupportedTypeError,
)

from tests.sdk.conftest import TEXT

pytestmark = pytest.mark.integration


async def test_ingest_list_and_skip_when_unchanged(rag: AsyncRagAPI, report: Path):
    first = await rag.documents.ingest(report)
    assert first.status == "succeeded" and first.text_chunks >= 1 and first.collection == "alpha"
    assert first.source.endswith("finance.txt") and first.skipped_reason is None
    if isinstance(rag, AsyncRagClient):
        assert first.status_url and first.status_url.endswith(f"/api/v1/jobs/{first.job_id}")
    else:
        assert first.status_url is None, "an in-process engine has no URL to poll"

    listing = await rag.documents.list()
    assert listing.total == 1 and listing.documents[0].source.endswith("finance.txt")
    assert listing.documents[0].text_chunks == first.text_chunks

    again = await rag.documents.ingest(report)
    assert again.skipped_reason == "no_changes_detected" and again.reindexed is False


async def test_ingest_accepts_bytes_and_streams_and_needs_a_filename_for_them(rag: AsyncRagAPI):
    from_bytes = await rag.documents.ingest(TEXT.encode(), filename="bytes.txt")
    from_stream = await rag.documents.ingest(
        io.BytesIO(TEXT.replace("twelve", "thirteen").encode()), filename="stream.txt"
    )
    assert from_bytes.status == from_stream.status == "succeeded"
    assert (await rag.documents.list()).total == 2
    with pytest.raises(InvalidRequestError, match="filename"):
        await rag.documents.ingest(TEXT.encode())
    with pytest.raises(NotFoundError, match="file not found"):
        await rag.documents.ingest("/no/such/file.txt")


async def test_ingest_into_a_named_collection_and_kind_selection(rag: AsyncRagAPI, report: Path):
    response = await rag.documents.ingest(report, collection="beta", kinds=["text"])
    assert response.collection == "beta" and response.text_chunks >= 1 and response.image_chunks == 0
    assert (await rag.documents.list("beta")).total == 1
    assert (await rag.documents.list("alpha")).total == 0


async def test_retrieve_is_scoped_by_collection_kind_and_source(rag: AsyncRagAPI, report: Path):
    await rag.documents.ingest(report)
    result = await rag.retrieve("quarterly revenue growth")
    assert result.query == "quarterly revenue growth" and result.expanded_queries[0] == result.query
    top = result.documents[0]
    assert "revenue" in top.content and top.collection == "alpha" and top.kind == "text"
    assert top.source and top.source.endswith("finance.txt") and top.score > 0

    assert (await rag.retrieve("revenue", collections=["beta"])).documents == []
    assert (await rag.retrieve("revenue", kinds=["table"])).documents == []
    assert (await rag.retrieve("revenue", sources=["/somewhere/else.txt"])).documents == []
    assert (await rag.retrieve("revenue", sources=[top.source])).documents


async def test_ask_answers_with_ranked_context_and_the_chosen_model(rag: AsyncRagAPI, report: Path):
    await rag.documents.ingest(report)
    answer = await rag.ask("what happened to revenue", model="smart")
    assert answer.model == "smart" and "[smart]" in answer.answer
    assert [c.rank for c in answer.context] == list(range(1, len(answer.context) + 1))
    assert answer.context[0].source and answer.context[0].source.endswith("finance.txt")
    with pytest.raises(NotFoundError, match="unknown chat model"):
        await rag.ask("revenue", model="nope")


async def test_chat_keeps_a_conversation_and_streams(rag: AsyncRagAPI, report: Path):
    await rag.documents.ingest(report)
    first = await rag.chat.send("How did revenue change?")
    assert first.conversation_id and first.answer and first.context
    follow = await rag.chat.send("And the margin?", conversation_id=first.conversation_id)
    assert follow.conversation_id == first.conversation_id and follow.standalone_query.startswith(
        "STANDALONE"
    )

    history = await rag.chat.get(first.conversation_id)
    assert [m.role for m in history.messages] == ["user", "assistant", "user", "assistant"]

    events = [e async for e in rag.chat.stream("Summarise the margin", conversation_id=first.conversation_id)]
    kinds = [type(e).__name__ for e in events]
    assert (
        kinds[0] == "ChatStartEvent"
        and kinds[-1] == "ChatEndEvent"
        and set(kinds[1:-1]) == {"ChatDeltaEvent"}
    )
    streamed = "".join(e.text for e in events if type(e).__name__ == "ChatDeltaEvent")
    assert (
        streamed.strip() == events[-1].answer.strip() and events[0].conversation_id == first.conversation_id
    )

    await rag.chat.delete(first.conversation_id)
    with pytest.raises(NotFoundError):
        await rag.chat.get(first.conversation_id)


async def test_a_stream_for_an_unknown_conversation_fails_before_any_event(rag: AsyncRagAPI):
    """Errors found up front are raised, not delivered as an event inside a successful stream."""
    with pytest.raises(NotFoundError):
        async for _ in rag.chat.stream("hello", conversation_id="does-not-exist"):
            pytest.fail("no event may precede the error")


async def test_jobs_documents_and_catalog(rag: AsyncRagAPI, report: Path):
    submitted = await rag.documents.ingest(report)
    job = await rag.jobs.get(submitted.job_id)
    assert job.status == "succeeded" and job.collection == "alpha" and job.result
    assert (await rag.jobs.wait(submitted.job_id)).job_id == submitted.job_id
    with pytest.raises(NotFoundError, match="job"):
        await rag.jobs.get("no-such-job")

    collections = {c.name: c for c in await rag.collections.list()}
    assert (
        set(collections) == {"alpha", "beta"}
        and collections["alpha"].default
        and not collections["beta"].default
    )
    assert collections["beta"].kinds == ["text", "table"]
    models = await rag.models()
    assert {m.name for m in models.chat} == {"fast", "smart"} and {m.name for m in models.embedding} == {
        "hash64",
        "hash32",
    }

    removed = await rag.documents.delete(submitted.source)
    assert removed.success and removed.deleted_count >= 1 and (await rag.documents.list()).total == 0


@pytest.mark.parametrize(
    ("call", "error", "code"),
    [
        (lambda r: r.retrieve("x", collections=["missing"]), NotFoundError, "not_found"),
        (lambda r: r.retrieve("x", kinds=["video"]), InvalidRequestError, "invalid_request"),
        (lambda r: r.retrieve(""), InvalidRequestError, "invalid_request"),
        (lambda r: r.documents.ingest(b"x", filename="notes.xyz"), UnsupportedTypeError, "unsupported_type"),
        (
            lambda r: r.documents.ingest(b"x", filename="a.txt", collection="missing"),
            NotFoundError,
            "not_found",
        ),
        (lambda r: r.documents.list(limit=0), InvalidRequestError, "invalid_request"),
        (lambda r: r.chat.get("nope"), NotFoundError, "not_found"),
    ],
)
async def test_the_same_mistakes_raise_the_same_typed_errors(rag: AsyncRagAPI, call, error, code):
    with pytest.raises(error) as caught:
        await call(rag)
    assert caught.value.code == code and isinstance(caught.value, RagError)


async def test_a_failed_ingestion_job_raises_its_own_error_with_the_job_id(rag: AsyncRagAPI):
    with pytest.raises(ParseError) as caught:
        await rag.documents.ingest(b"this is not a pdf", filename="broken.pdf")
    assert caught.value.code == "parse_error" and caught.value.details["job_id"]
    job = await rag.jobs.get(caught.value.details["job_id"])
    assert job.status == "failed" and job.error_code == "parse_error"
    with pytest.raises(ParseError):
        await rag.jobs.wait(job.job_id, timeout=5)


async def test_transport_specific_details_are_the_only_differences(rag: AsyncRagAPI):
    assert isinstance(rag, AsyncRagAPI)
    assert hasattr(rag, "evaluate") is isinstance(rag, AsyncRag), (
        "evaluate() exists exactly where the engine is"
    )
    if isinstance(rag, AsyncRagClient):
        assert not hasattr(rag, "engine")


# --- one engine, both ways: congruence ----------------------------------------------------------------
async def test_both_transports_return_identical_results_from_one_engine(both, report: Path):
    embedded, remote = both
    ingested = await remote.documents.ingest(report)  # ingest over HTTP...
    for sdk in (embedded, remote):  # ...and read it back through either transport
        assert (await sdk.documents.list()).total == 1

    queries = ["quarterly revenue growth", "operating margin", "something unrelated entirely"]
    for q in queries:
        assert await embedded.retrieve(q) == await remote.retrieve(q), q
        assert await embedded.retrieve(q, collections=["alpha", "beta"]) == await remote.retrieve(
            q, collections=["alpha", "beta"]
        )
    assert await embedded.ask("what happened to revenue") == await remote.ask("what happened to revenue")
    assert await embedded.collections.list() == await remote.collections.list()
    assert await embedded.models() == await remote.models()
    assert await embedded.documents.list() == await remote.documents.list()
    assert await embedded.jobs.get(ingested.job_id) == await remote.jobs.get(ingested.job_id)

    # a conversation started on one transport continues on the other
    turn = await embedded.chat.send("How did revenue change?")
    follow = await remote.chat.send("And the margin?", conversation_id=turn.conversation_id)
    assert follow.conversation_id == turn.conversation_id
    assert await embedded.chat.get(turn.conversation_id) == await remote.chat.get(turn.conversation_id)

    # identical failures, down to the code and message
    errors = []
    for sdk in (embedded, remote):
        with pytest.raises(NotFoundError) as caught:
            await sdk.retrieve("x", collections=["missing"])
        errors.append((type(caught.value), caught.value.code, str(caught.value)))
    assert errors[0] == errors[1]
