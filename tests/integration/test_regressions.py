"""Regressions for defects found in review that need live services to reproduce."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
import redis.asyncio as aioredis
from fastapi.testclient import TestClient
from redis.exceptions import RedisError
from src.api.server import create_app
from src.chat.service import ChatStarted
from src.core.container import Container
from src.core.errors import InvalidRequestError
from src.core.specs import RagConfig
from src.core.types import JobSpec
from src.jobs.redis_streams import RedisStreamsJobBackend

from tests.conftest import REDIS_URL

pytestmark = pytest.mark.integration


async def drop_indices(container: Container, run_id: str) -> None:
    names = list(await container.elastic.client.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
    if names:
        await container.elastic.client.indices.delete(index=names, ignore_unavailable=True)


async def test_the_directory_watcher_ingests_files_dropped_into_a_collection(
    make_settings, rag_config, run_id
):
    """Regression: start_watchers() referenced a class that was only imported for type checking,
    so WATCH_DATA_DIR=true crashed start-up with a NameError."""
    container = await Container.build(
        make_settings(watch_data_dir=True, watch_debounce_seconds=0.3), role="api", config=rag_config
    )
    await container.start()
    stop = asyncio.Event()
    worker = container.start_embedded_worker(stop)
    try:
        await container.start_watchers()
        assert len(container.watchers) == len(rag_config.collections)
        drop = container.store.directory(rag_config.collection("alpha"))
        (drop / "dropped.txt").write_text(
            "Dropped in a watched folder, long enough to become a chunk of text."
        )
        deadline = time.time() + 20
        while time.time() < deadline:
            _, total = await container.documents.list("alpha", limit=10, offset=0)
            if total:
                break
            await asyncio.sleep(0.2)
        records, total = await container.documents.list("alpha", limit=10, offset=0)
        assert total == 1 and records[0].source.endswith("dropped.txt") and records[0].parser == "text"
    finally:
        stop.set()
        await asyncio.wait_for(worker, 20)
        await drop_indices(container, run_id)
        await container.close()


async def test_per_collection_index_settings_override_the_global_ones(make_settings, run_id):
    """Regression: `shards`, `replicas` and `vector_index_type` on a collection were accepted and
    documented but never used; every index got the global values."""
    config = RagConfig.model_validate(
        {
            "default_chat_model": "c",
            "default_collection": "tuned",
            "chat_models": {"c": {"provider": "fake", "model": "c"}},
            "embedding_models": {"e": {"provider": "fake", "model": "e", "dimensions": 16}},
            "collections": {
                "tuned": {
                    "embedding_model": "e",
                    "index_prefix": f"t{run_id}-tuned",
                    "shards": 2,
                    "replicas": 0,
                    "vector_index_type": "int8_flat",
                },
                "plain": {"embedding_model": "e", "index_prefix": f"t{run_id}-plain"},
            },
        }
    )
    container = await Container.build(
        make_settings(es_number_of_shards=1, es_number_of_replicas=0, es_vector_index_type="int8_hnsw"),
        role="api",
        config=config,
    )
    await container.start()
    try:
        client = container.elastic.client
        tuned = await client.indices.get(index=f"t{run_id}-tuned-text")
        plain = await client.indices.get(index=f"t{run_id}-plain-text")
        t = tuned[f"t{run_id}-tuned-text"]
        p = plain[f"t{run_id}-plain-text"]
        assert (
            t["settings"]["index"]["number_of_shards"] == "2"
            and p["settings"]["index"]["number_of_shards"] == "1"
        )
        assert t["mappings"]["properties"]["content_vector"]["index_options"]["type"] == "int8_flat"
        assert p["mappings"]["properties"]["content_vector"]["index_options"]["type"] == "int8_hnsw"
    finally:
        await drop_indices(container, run_id)
        await container.close()


@pytest.mark.needs_redis
async def test_the_redis_lock_survives_a_transient_error_while_it_is_being_renewed(run_id):
    """Regression: a RedisError in the renewal task killed it silently; the lock then expired in the
    middle of the job, and the error resurfaced from lock() when the job had already succeeded."""
    from tests.integration.test_jobs import settings as job_settings

    holder = RedisStreamsJobBackend(job_settings(), namespace=f"t{run_id}")
    contender = RedisStreamsJobBackend(job_settings(), namespace=f"t{run_id}")
    holder._lock_ttl_ms = contender._lock_ttl_ms = 900  # noqa: SLF001  # renews every 0.3 s
    real_renew, failures = holder._renew, []  # noqa: SLF001

    async def flaky_renew(*args, **kwargs):
        if len(failures) < 2:
            failures.append(1)
            raise RedisError("connection reset")
        return await real_renew(*args, **kwargs)

    holder._renew = flaky_renew  # type: ignore[assignment]  # noqa: SLF001
    acquired_at: list[float] = []
    released_at: list[float] = []

    async def hold() -> None:
        async with holder.lock("doc"):  # must not raise, although two renewals failed
            await asyncio.sleep(2.4)  # far longer than the 0.9 s TTL: only renewal keeps it
        released_at.append(time.monotonic())

    async def wait_for_it() -> None:
        await asyncio.sleep(0.5)
        async with contender.lock("doc"):
            acquired_at.append(time.monotonic())

    await asyncio.gather(hold(), wait_for_it())
    assert len(failures) == 2
    assert acquired_at[0] >= released_at[0] - 0.05, (
        "the contender only got the lock after the holder released it"
    )
    client = aioredis.from_url(REDIS_URL)
    for key in [k async for k in client.scan_iter(f"t{run_id}:*")]:
        await client.delete(key)
    await client.aclose()
    await holder.close(), await contender.close()


async def test_queue_jobs_cannot_make_a_worker_read_files_outside_the_data_directory(
    container: Container, tmp_path: Path
):
    outside = tmp_path / "outside.txt"
    outside.write_text("Secret content that lives outside the data directory entirely.")
    with pytest.raises(InvalidRequestError, match="outside the data directory"):
        await container.handle_job(JobSpec(collection="alpha", path=str(outside)))
    with pytest.raises(InvalidRequestError, match="outside the data directory"):
        await container.handle_job(
            JobSpec(collection="alpha", path=str(container.store.data_dir / ".." / "outside.txt"))
        )

    inside = container.store.directory(container.config.collection("alpha")) / "ok.txt"
    inside.write_text("Inside the data directory, so it is ingested as normal, no questions asked.")
    result = await container.handle_job(JobSpec(collection="alpha", path=str(inside)))
    assert result["text_chunks"] >= 1


# --- HTTP surface ------------------------------------------------------------------------------
@pytest.fixture
def client(make_settings, rag_toml: Path, run_id: str):
    settings = make_settings(rag_config=rag_toml, request_timeout_seconds=1, max_upload_mb=1)
    with TestClient(create_app(settings)) as test_client:
        yield test_client
        es = test_client.app.state.container.elastic.client

        async def cleanup():
            names = list(await es.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
            if names:
                await es.indices.delete(index=names, ignore_unavailable=True)

        test_client.portal.call(cleanup)


def test_an_oversized_chunked_upload_is_refused_before_it_is_stored(client: TestClient):
    def body():
        yield b'--B\r\nContent-Disposition: form-data; name="file"; filename="big.txt"\r\nContent-Type: text/plain\r\n\r\n'
        for _ in range(30):  # 3 MB against a 1 MB limit, sent chunked (no Content-Length)
            yield b"x" * 100_000
        yield b"\r\n--B--\r\n"

    response = client.post(
        "/api/v1/ingest", content=body(), headers={"Content-Type": "multipart/form-data; boundary=B"}
    )
    assert response.status_code == 413 and response.json()["code"] == "payload_too_large"
    data_dir = client.app.state.container.settings.data_dir
    assert not any(p.name == "big.txt" for p in data_dir.rglob("*")), "nothing was stored"


def test_a_stalled_chat_stream_is_cut_off_by_the_request_deadline(client: TestClient, monkeypatch):
    """Regression: only the first event was covered by the deadline; a model that streamed one
    token and then stalled held the connection open indefinitely."""
    container = client.app.state.container

    async def stalls(*args, **kwargs):
        yield ChatStarted("conv", "q", ["q"], [], "fast")
        await asyncio.sleep(30)
        yield None  # pragma: no cover

    monkeypatch.setattr(container.chat, "stream", stalls)
    started = time.time()
    with client.stream("POST", "/api/v1/chat/stream", json={"message": "hello"}) as response:
        text = "".join(response.iter_text())
    assert time.time() - started < 10
    blocks = [b for b in text.strip().split("\n\n") if b]
    assert blocks[0].startswith("event: start") and blocks[-1].startswith("event: error")
    assert json.loads(blocks[-1].split("data: ", 1)[1])["code"] == "timeout"


@pytest.mark.parametrize("mode", ["each", "interval"])
async def test_ingest_refresh_mode_decides_who_makes_new_chunks_searchable(
    make_settings, rag_config, run_id, tmp_path, mode
):
    """`each` refreshes the touched indices before reporting success (searchable immediately);
    `interval` leaves it to Elasticsearch's own refresh interval, which keeps the per-document
    cost of a bulk load low. Either way a search that runs *before* the data is visible must not
    leave an empty answer cached under the new corpus version."""
    container = await Container.build(
        make_settings(ingest_refresh=mode, es_refresh_interval="3s"),
        role="api",
        with_ingestion=True,
        config=rag_config,
    )
    await container.start()
    try:
        refreshes = 0
        writer = container.ingestion._writer
        real_refresh = writer.refresh

        async def counting(indices):
            nonlocal refreshes
            refreshes += 1
            await real_refresh(indices)

        writer.refresh = counting  # type: ignore[method-assign]
        note = tmp_path / "note.txt"
        note.write_text(
            "Quarterly revenue grew twelve percent driven by cloud subscriptions in every region."
        )
        summary = await container.ingestion.ingest(JobSpec(collection="alpha", path=str(note)))
        assert summary.text_chunks >= 1
        assert refreshes == (1 if mode == "each" else 0)

        scope = container.retrieval.scope(["alpha"])
        result = await container.retrieval.retrieve("quarterly revenue", scope)  # possibly too early
        if mode == "each":
            assert result.documents, "refreshed before the job reported success"
        deadline = time.time() + 15
        while not result.documents and time.time() < deadline:
            await asyncio.sleep(0.5)
            result = await container.retrieval.retrieve("quarterly revenue", scope)
        assert result.documents and "revenue" in result.documents[0].content
    finally:
        await drop_indices(container, run_id)
        await container.close()


async def test_interval_mode_still_sweeps_a_generation_written_moments_ago(
    make_settings, rag_config, run_id, tmp_path
):
    """The stale-generation sweep only sees searchable chunks. With a long refresh interval, re-ingesting
    a changed file straight away must still remove the previous version's chunks."""
    container = await Container.build(
        make_settings(ingest_refresh="interval", es_refresh_interval="30s"),
        role="api",
        with_ingestion=True,
        config=rag_config,
    )
    await container.start()
    try:
        note = tmp_path / "note.txt"
        spec = JobSpec(collection="alpha", path=str(note))
        note.write_text("First version of the note about quarterly revenue and cloud subscriptions.")
        await container.ingestion.ingest(spec)
        note.write_text("Second, rewritten version of the note about headcount and infrastructure spending.")
        second = await container.ingestion.ingest(spec)
        assert second.reindexed is not False and second.document_id

        index = container.config.collection("alpha").index_names()["text"]
        await container.elastic.client.indices.refresh(index=index)
        found = await container.elastic.client.search(
            index=index,
            query={"term": {"source": str(note)}},
            size=50,
            source_includes=["document_id", "content"],
        )
        generations = {hit["_source"]["document_id"] for hit in found["hits"]["hits"]}
        assert generations == {second.document_id}, "chunks of the first version were left behind"
    finally:
        await drop_indices(container, run_id)
        await container.close()
