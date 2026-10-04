"""Both job backends. Redis tests exercise the real failure modes of a distributed queue."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
import redis.asyncio as aioredis
from src.core.config import Settings
from src.core.errors import InvalidRequestError, QueueFullError, UpstreamError
from src.core.types import JobSpec
from src.jobs.inprocess import InProcessJobBackend
from src.jobs.redis_streams import RedisStreamsJobBackend

from tests.conftest import REDIS_URL

pytestmark = pytest.mark.integration

SPEC = JobSpec(collection="alpha", path="/data/a.pdf")
Handler = Callable[[JobSpec], Awaitable[dict[str, Any]]]


def settings(**overrides) -> Settings:
    base = dict(
        redis_url=REDIS_URL,
        ingest_max_attempts=3,
        job_visibility_timeout_seconds=1,
        ingest_queue_max_size=1000,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


class Worker:
    """Runs ``backend.run_worker`` in the background; ``crash()`` cancels it without draining."""

    def __init__(self, backend, handler: Handler, concurrency: int = 1) -> None:
        self.stop = asyncio.Event()
        self.task = asyncio.create_task(backend.run_worker(handler, concurrency=concurrency, stop=self.stop))

    async def shutdown(self) -> None:
        self.stop.set()
        await asyncio.wait_for(self.task, 10)

    async def crash(self) -> None:
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task


# --- behaviour every backend must have (Liskov: the same tests run on both) -----------------
@pytest.fixture(params=["inprocess", pytest.param("redis", marks=pytest.mark.needs_redis)])
async def backend(request, run_id) -> AsyncIterator[Any]:
    if request.param == "inprocess":
        instance: Any = InProcessJobBackend(settings())
    else:
        instance = RedisStreamsJobBackend(settings(), namespace=f"t{run_id}")
    await instance.start()
    yield instance
    if request.param == "redis":
        client = aioredis.from_url(REDIS_URL)
        keys = [k async for k in client.scan_iter(f"t{run_id}:*")]
        if keys:
            await client.delete(*keys)
        await client.aclose()
    await instance.close()


async def test_job_runs_and_reports_result(backend):
    async def handler(spec: JobSpec) -> dict[str, Any]:
        return {"source": spec.path, "text_chunks": 3}

    worker = Worker(backend, handler)
    record = await backend.submit(SPEC)
    assert record.status == "queued"
    done = await backend.wait(record.id, 10)
    assert done.status == "succeeded" and done.result == {"source": "/data/a.pdf", "text_chunks": 3}
    assert done.attempts == 1 and done.finished_at and (await backend.get(record.id)).status == "succeeded"
    assert await backend.get("missing") is None
    await worker.shutdown()


async def test_permanent_failure_keeps_code_and_status_and_is_not_retried(backend):
    calls = 0

    async def handler(spec: JobSpec) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise InvalidRequestError("bad input")

    worker = Worker(backend, handler)
    done = await backend.wait((await backend.submit(SPEC)).id, 10)
    assert done.status == "failed" and calls == 1
    assert done.error_code == "invalid_request" and done.error_status == 400 and "bad input" in done.error
    await worker.shutdown()


async def test_transient_failure_is_retried_then_succeeds(backend):
    calls = 0

    async def handler(spec: JobSpec) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise UpstreamError("elasticsearch hiccup")
        return {"ok": True}

    worker = Worker(backend, handler)
    done = await backend.wait((await backend.submit(SPEC)).id, 15)
    assert done.status == "succeeded" and calls == 3 and done.attempts == 3
    await worker.shutdown()


async def test_retries_are_capped(backend):
    async def handler(spec: JobSpec) -> dict[str, Any]:
        raise UpstreamError("down")

    worker = Worker(backend, handler)
    done = await backend.wait((await backend.submit(SPEC)).id, 15)
    assert done.status == "failed" and done.attempts == 3 and done.error_code == "upstream_error"
    await worker.shutdown()


async def test_wait_returns_current_state_on_timeout(backend):
    record = await backend.submit(SPEC)  # no worker running
    waited = await backend.wait(record.id, 0.3)
    assert waited.status == "queued"


async def test_lock_is_mutually_exclusive(backend):
    order: list[str] = []

    async def critical(name: str) -> None:
        async with backend.lock("alpha:/data/a.pdf"):
            order.append(f"{name}-in")
            await asyncio.sleep(0.2)
            order.append(f"{name}-out")

    await asyncio.gather(critical("a"), critical("b"))
    assert order in (["a-in", "a-out", "b-in", "b-out"], ["b-in", "b-out", "a-in", "a-out"])


async def test_queue_backpressure(run_id):
    small = InProcessJobBackend(settings(ingest_queue_max_size=2))
    await small.submit(SPEC), await small.submit(SPEC)
    with pytest.raises(QueueFullError) as info:
        await small.submit(SPEC)
    assert info.value.status_code == 503 and info.value.retry_after >= 1
    assert await small.depth() == 2


# --- what only a distributed queue has to get right -----------------------------------------
@pytest.mark.needs_redis
async def test_redis_queue_backpressure(run_id):
    backend = RedisStreamsJobBackend(settings(ingest_queue_max_size=2), namespace=f"t{run_id}")
    await backend.start()
    try:
        await backend.submit(SPEC), await backend.submit(SPEC)
        with pytest.raises(QueueFullError):
            await backend.submit(SPEC)
        assert await backend.depth() == 2
    finally:
        client = aioredis.from_url(REDIS_URL)
        await client.delete(*[k async for k in client.scan_iter(f"t{run_id}:*")])
        await client.aclose()
        await backend.close()


@pytest.fixture
async def redis_ns(run_id) -> AsyncIterator[str]:
    yield f"t{run_id}"
    client = aioredis.from_url(REDIS_URL)
    keys = [k async for k in client.scan_iter(f"t{run_id}:*")]
    if keys:
        await client.delete(*keys)
    await client.aclose()


@pytest.mark.needs_redis
async def test_jobs_are_handled_exactly_once_across_competing_workers(redis_ns):
    handled: list[str] = []

    async def handler(spec: JobSpec) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        handled.append(spec.path)
        return {}

    backends = [RedisStreamsJobBackend(settings(), namespace=redis_ns) for _ in range(3)]
    workers = [Worker(b, handler, concurrency=2) for b in backends]
    paths = [f"/data/{i}.pdf" for i in range(30)]
    records = [await backends[0].submit(JobSpec(collection="alpha", path=p)) for p in paths]
    finals = [await backends[0].wait(r.id, 30) for r in records]
    assert all(f.status == "succeeded" for f in finals)
    assert sorted(handled) == sorted(paths), "every job exactly once, none duplicated or lost"
    assert await backends[0].depth() == 0, "acknowledged entries are removed from the stream"
    for w in workers:
        await w.shutdown()
    for b in backends:
        await b.close()


@pytest.mark.needs_redis
async def test_a_job_orphaned_by_a_crashed_worker_is_taken_over(redis_ns):
    started = asyncio.Event()

    async def hangs_forever(spec: JobSpec) -> dict[str, Any]:
        started.set()
        await asyncio.sleep(3600)
        return {}

    async def finishes(spec: JobSpec) -> dict[str, Any]:
        return {"recovered": True}

    victim = RedisStreamsJobBackend(settings(), namespace=redis_ns)
    rescuer = RedisStreamsJobBackend(settings(), namespace=redis_ns)
    record = await victim.submit(SPEC)
    doomed = Worker(victim, hangs_forever)
    await asyncio.wait_for(started.wait(), 10)
    await doomed.crash()  # no ack, no result: the entry is left pending

    survivor = Worker(rescuer, finishes)
    done = await rescuer.wait(record.id, 20)
    assert done.status == "succeeded" and done.result == {"recovered": True}
    assert done.attempts == 2, "the take-over counts as a second attempt"
    await survivor.shutdown()
    await victim.close(), await rescuer.close()


@pytest.mark.needs_redis
async def test_heartbeat_stops_a_long_running_job_from_being_taken_over(redis_ns):
    calls: list[str] = []

    async def slow(spec: JobSpec) -> dict[str, Any]:
        calls.append("run")
        await asyncio.sleep(3.0)  # three times the 1s visibility timeout
        return {"done": True}

    a = RedisStreamsJobBackend(settings(), namespace=redis_ns)
    b = RedisStreamsJobBackend(settings(), namespace=redis_ns)
    record = await a.submit(SPEC)
    workers = [Worker(a, slow), Worker(b, slow)]
    done = await a.wait(record.id, 20)
    assert done.status == "succeeded" and calls == ["run"], "heartbeats kept the entry from being re-claimed"
    for w in workers:
        await w.shutdown()
    await a.close(), await b.close()


@pytest.mark.needs_redis
async def test_a_job_that_keeps_killing_workers_is_failed_not_retried_forever(redis_ns):
    backend = RedisStreamsJobBackend(settings(ingest_max_attempts=2), namespace=redis_ns)
    record = await backend.submit(SPEC)
    exhausted = await backend.get(record.id)
    exhausted.attempts = 2  # two workers already started it and died
    await backend._save(exhausted)  # noqa: SLF001
    ran = False

    async def handler(spec: JobSpec) -> dict[str, Any]:
        nonlocal ran
        ran = True
        return {}

    worker = Worker(backend, handler)
    done = await backend.wait(record.id, 10)
    assert done.status == "failed" and "gave up after 2 attempts" in done.error and not ran
    await worker.shutdown()
    await backend.close()


@pytest.mark.needs_redis
async def test_redis_lock_excludes_other_processes_and_survives_longer_than_its_ttl(redis_ns):
    a = RedisStreamsJobBackend(settings(), namespace=redis_ns)
    b = RedisStreamsJobBackend(settings(), namespace=redis_ns)
    held = asyncio.Event()
    release = asyncio.Event()

    async def holder() -> None:
        async with a.lock("doc"):
            held.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await held.wait()
    acquired = asyncio.Event()

    async def contender() -> None:
        async with b.lock("doc"):
            acquired.set()

    contender_task = asyncio.create_task(contender())
    await asyncio.sleep(0.6)
    assert not acquired.is_set(), "second process must wait while the first holds the lock"
    release.set()
    await asyncio.wait_for(acquired.wait(), 5)
    await asyncio.gather(task, contender_task)
    await a.close(), await b.close()
