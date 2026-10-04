from __future__ import annotations

import asyncio

import pytest
from src.core.errors import InvalidRequestError, RequestTimeoutError, UpstreamError
from src.runtime import cache as cache_module
from src.runtime import ratelimit as ratelimit_module
from src.runtime.cache import CachedCall, CorpusVersion, MemoryCache, TieredCache
from src.runtime.concurrency import Bulkhead, SingleFlight, deadline_iter, run_all
from src.runtime.ratelimit import MemoryRateLimiter


# --- caches (the same contract for every backend) -------------------------------------------
async def test_memory_cache_get_set_delete_and_ttl(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: now[0])
    cache = MemoryCache(10)
    await cache.set("k", b"v", ttl=5)
    assert await cache.get("k") == b"v"
    now[0] += 6
    assert await cache.get("k") is None, "expired entries are not served"
    await cache.set("k", b"v2", ttl=5)
    await cache.delete("k")
    assert await cache.get("k") is None


async def test_memory_cache_is_a_bounded_lru():
    cache = MemoryCache(2)
    await cache.set("a", b"1", 60)
    await cache.set("b", b"2", 60)
    await cache.get("a")  # touch a: b is now the least recently used
    await cache.set("c", b"3", 60)
    assert await cache.get("b") is None and await cache.get("a") == b"1" and await cache.get("c") == b"3"


async def test_counters_are_separate_from_values_and_never_regress():
    cache = MemoryCache(10)
    assert await cache.counter("n") == 0
    assert [await cache.incr("n"), await cache.incr("n")] == [1, 2]
    assert await cache.counter("n") == 2


async def test_tiered_cache_promotes_to_l1_and_reads_counters_from_l2():
    l1, l2 = MemoryCache(10), MemoryCache(10)
    tiered = TieredCache(l1, l2)
    await l2.set("k", b"shared", 60)
    assert await l1.get("k") is None
    assert await tiered.get("k") == b"shared" and await l1.get("k") == b"shared", "L2 hit populates L1"
    await tiered.incr("version")
    assert (
        await tiered.counter("version") == 1
        and await l2.counter("version") == 1
        and await l1.counter("version") == 0
    )


async def test_corpus_version_bump_changes_the_version_everyone_sees():
    shared = MemoryCache(10)
    replica_a, replica_b = CorpusVersion(shared, refresh_seconds=0), CorpusVersion(shared, refresh_seconds=0)
    assert await replica_a.current() == await replica_b.current() == 0
    await replica_a.bump()
    assert await replica_b.current() == 1, "a bump on one replica is visible on the others"


async def test_corpus_version_reads_are_memoised():
    class Counting(MemoryCache):
        reads = 0

        async def counter(self, key: str) -> int:
            Counting.reads += 1
            return await super().counter(key)

    version = CorpusVersion(Counting(10), refresh_seconds=60)
    await version.current(), await version.current(), await version.current()
    assert Counting.reads == 1, "hot path costs no extra round trips"


async def test_corpus_is_settling_for_a_while_after_a_bump_on_every_replica():
    shared = MemoryCache(10)
    writer = CorpusVersion(shared, refresh_seconds=0, settle_seconds=0.4)
    reader = CorpusVersion(shared, refresh_seconds=0, settle_seconds=0.4)
    assert await reader.state() == (0, False)
    await writer.bump()
    assert await reader.state() == (1, True), "new version, but its data may not be searchable yet"
    await asyncio.sleep(0.5)
    assert await reader.state() == (1, False)


async def test_a_corpus_without_a_settle_window_never_reports_settling():
    version = CorpusVersion(MemoryCache(10), refresh_seconds=0)
    await version.bump()
    assert await version.state() == (1, False)


async def test_cached_call_computes_once_for_concurrent_identical_requests():
    calls = 0

    async def compute() -> bytes:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return b"result"

    cached = CachedCall(MemoryCache(10), "test")
    results = await asyncio.gather(*(cached.get_or_compute("key", 60, compute) for _ in range(20)))
    assert results == [b"result"] * 20 and calls == 1, "cache stampede protection"
    assert await cached.get_or_compute("key", 60, compute) == b"result" and calls == 1


async def test_single_flight_survives_a_cancelled_waiter():
    flight = SingleFlight()
    finished = asyncio.Event()

    async def work() -> str:
        await asyncio.sleep(0.1)
        finished.set()
        return "done"

    first = asyncio.create_task(flight.do("k", work))
    second = asyncio.create_task(flight.do("k", work))
    await asyncio.sleep(0.01)
    first.cancel()  # e.g. a client disconnect
    assert await second == "done" and finished.is_set()


async def test_single_flight_propagates_errors_to_everyone_and_forgets_them():
    flight = SingleFlight()
    attempts = 0

    async def fails() -> str:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.02)
        raise UpstreamError("boom")

    results = await asyncio.gather(*(flight.do("k", fails) for _ in range(3)), return_exceptions=True)
    assert all(isinstance(r, UpstreamError) for r in results) and attempts == 1
    with pytest.raises(UpstreamError):
        await flight.do("k", fails)
    assert attempts == 2, "a failed flight is not cached"


# --- concurrency primitives -----------------------------------------------------------------
async def test_bulkhead_bounds_concurrency():
    bulkhead = Bulkhead(2)
    active = peak = 0

    async def use() -> None:
        nonlocal active, peak
        async with bulkhead:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(*(use() for _ in range(10)))
    assert peak == 2 and bulkhead.in_flight == 0


async def test_run_all_returns_in_order_and_raises_the_original_error_cancelling_the_rest():
    assert await run_all(asyncio.sleep(0.01 * (3 - i), result=i) for i in range(3)) == [0, 1, 2]

    cancelled = asyncio.Event()

    async def slow() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def bad() -> None:
        await asyncio.sleep(0.01)
        raise InvalidRequestError("typed, not wrapped in an ExceptionGroup")

    with pytest.raises(InvalidRequestError):
        await run_all([slow(), bad()])
    assert cancelled.is_set()


# --- rate limiter ---------------------------------------------------------------------------
async def test_memory_rate_limiter_counts_per_key_and_window(monkeypatch):
    now = [10_000.0]
    monkeypatch.setattr(ratelimit_module.time, "time", lambda: now[0])
    limiter = MemoryRateLimiter()
    decisions = [await limiter.hit("client-1", 3, 60) for _ in range(5)]
    assert [d.allowed for d in decisions] == [True, True, True, False, False]
    assert [d.remaining for d in decisions] == [2, 1, 0, 0, 0] and decisions[3].retry_after >= 1
    assert (await limiter.hit("client-2", 3, 60)).allowed, "keys are independent"
    now[0] += 61
    assert (await limiter.hit("client-1", 3, 60)).allowed, "a new window starts fresh"


# --- deadline_iter: only time spent waiting on the source counts -----------------------------------------------------
async def test_a_consumer_that_pauses_between_items_does_not_eat_the_deadline():
    async def source():
        for i in range(3):
            await asyncio.sleep(0.01)
            yield i

    seen = []
    async for item in deadline_iter(source(), 0.15):
        seen.append(item)
        await asyncio.sleep(0.12)  # a slow consumer: 3 x 0.12s is well past the 0.15s budget
    assert seen == [0, 1, 2]


async def test_a_source_that_stalls_is_cut_off_with_the_typed_error_and_closed():
    closed = []

    async def source():
        try:
            yield 1
            await asyncio.sleep(5)
            yield 2
        finally:
            closed.append(True)

    got = []
    with pytest.raises(RequestTimeoutError, match="0.1s"):
        async for item in deadline_iter(source(), 0.1):
            got.append(item)
    assert got == [1] and closed == [True]


async def test_waiting_time_accumulates_across_items():
    async def source():
        for i in range(10):
            await asyncio.sleep(0.03)
            yield i

    got = []
    with pytest.raises(RequestTimeoutError):
        async for item in deadline_iter(source(), 0.1):
            got.append(item)
    assert 1 <= len(got) < 10
