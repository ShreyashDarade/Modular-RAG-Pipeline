"""The scale-out topology: several stateless API replicas and a separate ingestion worker that
share nothing but Redis and Elasticsearch."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
import redis.asyncio as aioredis
from fastapi.testclient import TestClient
from src.api.server import create_app
from src.core.container import Container

from tests.conftest import REDIS_URL
from tests.helpers import make_pdf

pytestmark = [pytest.mark.integration, pytest.mark.needs_redis]

V1 = ["Quarterly revenue grew twelve percent driven by cloud subscriptions across all regions."]
V2 = ["Gardening season brings soil temperature changes and seed germination across the northern fields."]


@pytest.fixture
async def topology(make_settings, rag_toml: Path, run_id: str, tmp_path: Path):
    """Two API replicas (queue only, no OCR stack, no embedded worker) + one worker + shared state."""
    shared = dict(
        rag_config=rag_toml,
        ingest_backend="redis",
        ingest_embedded_worker=False,
        cache_backend="tiered",
        rate_limit_backend="redis",
        chat_store="redis",
        redis_namespace=f"t{run_id}",
        job_visibility_timeout_seconds=2,
        ingest_concurrency=2,
        data_dir=tmp_path / "shared-data",
    )
    api_settings = make_settings(rate_limit_per_minute=1000, **shared)
    replica_a = TestClient(create_app(api_settings))
    replica_b = TestClient(create_app(api_settings))
    replica_a.__enter__(), replica_b.__enter__()

    worker = await Container.build(make_settings(**shared), role="worker")
    await worker.start()
    stop = asyncio.Event()
    task = asyncio.create_task(worker.jobs.run_worker(worker.handle_job, concurrency=2, stop=stop))
    try:
        yield replica_a, replica_b, worker
    finally:
        stop.set()
        await asyncio.wait_for(task, 20)
        names = list(await worker.elastic.client.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
        if names:
            await worker.elastic.client.indices.delete(index=names, ignore_unavailable=True)
        await worker.close()
        replica_a.__exit__(None, None, None)
        replica_b.__exit__(None, None, None)
        client = aioredis.from_url(REDIS_URL)
        keys = [k async for k in client.scan_iter(f"t{run_id}:*")]
        if keys:
            await client.delete(*keys)
        await client.aclose()


def upload(client: TestClient, path: Path, **params):
    with path.open("rb") as handle:
        return client.post("/api/v1/ingest", files={"file": (path.name, handle)}, params=params)


def wait_for_job(client: TestClient, job_id: str, timeout: float = 30) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get(f"/api/v1/jobs/{job_id}").json()
        if state["status"] in ("succeeded", "failed"):
            return state
        time.sleep(0.1)
    raise AssertionError("job did not finish")


async def test_replicas_and_a_worker_cooperate_through_shared_state(topology, tmp_path: Path):
    a, b, worker = topology
    assert worker.ingestion is not None
    assert a.app.state.container.ingestion is None and b.app.state.container.ingestion is None, (
        "API replicas carry no ingestion stack"
    )
    assert (
        a.get("/ready").json()["checks"]["jobs"] == "ok" and b.get("/ready").json()["checks"]["cache"] == "ok"
    )

    # ingest through replica A: queued, picked up by the separate worker ...
    queued = upload(a, make_pdf(tmp_path / "doc.pdf", V1), wait="false")
    assert queued.status_code == 202
    # ... and visible from replica B, which never saw the upload
    state = await asyncio.to_thread(wait_for_job, b, queued.json()["job_id"])
    assert state["status"] == "succeeded" and state["result"]["text_chunks"] >= 1

    found = b.post("/api/v1/retrieve", json={"query": "quarterly revenue growth"}).json()
    assert found["documents"] and "revenue" in found["documents"][0]["content"]

    # a conversation started on A continues on B (Redis-backed history)
    first = a.post("/api/v1/chat", json={"message": "how did revenue do?"}).json()
    second = b.post(
        "/api/v1/chat", json={"message": "and margin?", "conversation_id": first["conversation_id"]}
    )
    assert second.status_code == 200 and second.json()["standalone_query"].startswith("STANDALONE")


async def test_changing_a_document_invalidates_cached_results_on_every_replica(topology, tmp_path: Path):
    a, b, _ = topology
    doc = tmp_path / "doc.pdf"
    make_pdf(doc, V1)
    first = await asyncio.to_thread(wait_for_job, a, upload(a, doc, wait="false").json()["job_id"])
    assert first["status"] == "succeeded"
    before = b.post("/api/v1/retrieve", json={"query": "revenue cloud subscriptions"}).json()
    assert "revenue" in before["documents"][0]["content"]  # now cached on replica B

    make_pdf(doc, V2)
    second = await asyncio.to_thread(wait_for_job, a, upload(a, doc, wait="false").json()["job_id"])
    assert second["status"] == "succeeded" and second["result"]["reindexed"] is True
    await asyncio.sleep(1.2)  # the corpus version is memoised for ~1s per replica
    after = b.post("/api/v1/retrieve", json={"query": "revenue cloud subscriptions"}).json()
    assert all("revenue" not in d["content"] for d in after["documents"]), (
        "replica B must not serve the stale cached answer"
    )
    assert (
        "gardening"
        in b.post("/api/v1/retrieve", json={"query": "gardening soil"})
        .json()["documents"][0]["content"]
        .lower()
    )


async def test_rate_limit_is_shared_across_replicas(make_settings, rag_toml: Path, run_id: str):
    settings = make_settings(
        rag_config=rag_toml, rate_limit_per_minute=4, rate_limit_backend="redis", redis_namespace=f"t{run_id}"
    )
    with TestClient(create_app(settings)) as a, TestClient(create_app(settings)) as b:
        codes = [
            client.post("/api/v1/retrieve", json={"query": "x"}).status_code for client in (a, b, a, b, a, b)
        ]
        assert codes == [200, 200, 200, 200, 429, 429], "4 per minute in total, not 4 per replica"
    client = aioredis.from_url(REDIS_URL)
    keys = [k async for k in client.scan_iter(f"t{run_id}:*")]
    if keys:
        await client.delete(*keys)
    await client.aclose()


async def test_concurrent_uploads_of_one_file_from_two_replicas_end_up_consistent(topology, tmp_path: Path):
    a, b, worker = topology
    doc = tmp_path / "same.pdf"
    make_pdf(doc, V1)
    jobs = [upload(client, doc, wait="false").json()["job_id"] for client in (a, b, a, b)]
    states = [await asyncio.to_thread(wait_for_job, a, job) for job in jobs]
    assert all(s["status"] == "succeeded" for s in states)
    reindexed = [s["result"]["reindexed"] for s in states]
    assert reindexed.count(True) >= 1 and reindexed.count(False) >= 1, (
        "the per-document lock serialises runs; later ones see an up-to-date ledger"
    )
    spec = worker.config.collection("alpha")
    for index in spec.index_names().values():
        await worker.elastic.client.indices.refresh(index=index)
    total = 0
    for index in spec.index_names().values():
        total += (await worker.elastic.client.count(index=index))["count"]
    one_run = max(
        s["result"]["text_chunks"] + s["result"]["table_chunks"] + s["result"]["image_chunks"] for s in states
    )
    assert total == one_run, "exactly one generation of chunks remains"
