"""Liskov in practice: one set of assertions per port, run against every implementation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
import redis.asyncio as aioredis
from src.chat.stores import MemoryConversationStore, RedisConversationStore
from src.core.types import ChatMessage
from src.runtime.cache import MemoryCache, RedisCache, TieredCache
from src.runtime.ratelimit import MemoryRateLimiter, RedisRateLimiter

from tests.conftest import REDIS_URL

redis_only = pytest.mark.needs_redis
pytestmark = pytest.mark.integration


@pytest.fixture
async def namespace(run_id: str) -> AsyncIterator[str]:
    yield f"t{run_id}"
    client = aioredis.from_url(REDIS_URL)
    keys = [k async for k in client.scan_iter(f"t{run_id}:*")]
    if keys:
        await client.delete(*keys)
    await client.aclose()


# --- Cache ----------------------------------------------------------------------------------
@pytest.fixture(
    params=["memory", pytest.param("redis", marks=redis_only), pytest.param("tiered", marks=redis_only)]
)
async def cache(request, namespace) -> AsyncIterator[Any]:
    if request.param == "memory":
        instance: Any = MemoryCache(100)
    elif request.param == "redis":
        instance = RedisCache(REDIS_URL, namespace=namespace)
    else:
        instance = TieredCache(MemoryCache(100), RedisCache(REDIS_URL, namespace=namespace))
    yield instance
    await instance.close()


async def test_cache_roundtrip_delete_and_binary_safety(cache):
    assert await cache.get("missing") is None
    payload = bytes(range(256)) + "हिंदी".encode()
    await cache.set("k", payload, ttl=30)
    assert await cache.get("k") == payload
    await cache.delete("k")
    assert await cache.get("k") is None
    await cache.ping()


async def test_cache_counters_are_atomic_across_concurrent_callers(cache):
    results = await asyncio.gather(*(cache.incr("version") for _ in range(50)))
    assert sorted(results) == list(range(1, 51)) and await cache.counter("version") == 50
    assert await cache.counter("never-used") == 0


@redis_only
async def test_redis_cache_entries_expire(namespace):
    cache = RedisCache(REDIS_URL, namespace=namespace)
    await cache.set("short", b"v", ttl=1)
    assert await cache.get("short") == b"v"
    await asyncio.sleep(1.3)
    assert await cache.get("short") is None
    await cache.close()


@redis_only
async def test_redis_cache_failures_are_errors_not_misses():
    from src.core.errors import UpstreamError

    broken = RedisCache("redis://127.0.0.1:1/0")
    with pytest.raises(UpstreamError):
        await broken.get("k")
    with pytest.raises(UpstreamError):
        await broken.ping()
    await broken.close()


# --- RateLimiter ----------------------------------------------------------------------------
@pytest.fixture(params=["memory", pytest.param("redis", marks=redis_only)])
async def limiter(request, namespace) -> AsyncIterator[Any]:
    instance: Any = (
        MemoryRateLimiter() if request.param == "memory" else RedisRateLimiter(REDIS_URL, namespace=namespace)
    )
    yield instance
    await instance.close()


async def _inside_one_window(seconds: int = 60, margin: float = 2.0) -> None:
    """Fixed-window limiters count per wall-clock window: wait out a boundary so a test burst cannot straddle it."""
    left = seconds - (time.time() % seconds)
    if left < margin:
        await asyncio.sleep(left + 0.1)


async def test_limiter_allows_up_to_the_limit_then_blocks_per_key(limiter):
    await _inside_one_window()
    decisions = [await limiter.hit("client-a", 3, 60) for _ in range(5)]
    assert [d.allowed for d in decisions] == [True, True, True, False, False]
    assert [d.remaining for d in decisions] == [2, 1, 0, 0, 0]
    assert all(1 <= d.retry_after <= 61 for d in decisions)
    assert (await limiter.hit("client-b", 3, 60)).allowed


async def test_limiter_is_exact_under_concurrency(limiter):
    await _inside_one_window()
    decisions = await asyncio.gather(*(limiter.hit("burst", 10, 60) for _ in range(40)))
    assert sum(d.allowed for d in decisions) == 10, "no over- or under-admission when requests race"


@redis_only
async def test_redis_limiter_failure_is_an_error_not_fail_open():
    from src.core.errors import UpstreamError

    broken = RedisRateLimiter("redis://127.0.0.1:1/0")
    with pytest.raises(UpstreamError):
        await broken.hit("k", 5, 60)
    await broken.close()


# --- ConversationStore ----------------------------------------------------------------------
@pytest.fixture(params=["memory", pytest.param("redis", marks=redis_only)])
async def store(request, namespace) -> AsyncIterator[Any]:
    if request.param == "memory":
        instance: Any = MemoryConversationStore(max_conversations=3, ttl=3600)
    else:
        instance = RedisConversationStore(REDIS_URL, ttl=3600, namespace=namespace)
    yield instance
    await instance.close()


def turn(i: int) -> list[ChatMessage]:
    return [ChatMessage("user", f"question {i} हिंदी"), ChatMessage("assistant", f"answer {i}")]


async def test_store_keeps_order_returns_the_most_recent_and_isolates_conversations(store):
    assert await store.load("unknown", 10) == []
    for i in range(4):
        await store.append("c1", turn(i))
    await store.append("c2", turn(99))
    recent = await store.load("c1", 4)
    assert [m.content for m in recent] == ["question 2 हिंदी", "answer 2", "question 3 हिंदी", "answer 3"]
    assert len(await store.load("c1", 100)) == 8 and await store.load("c1", 0) == []
    assert [m.content for m in await store.load("c2", 10)] == ["question 99 हिंदी", "answer 99"]


async def test_store_delete_reports_whether_anything_was_deleted(store):
    await store.append("c1", turn(1))
    assert await store.delete("c1") is True and await store.delete("c1") is False
    assert await store.load("c1", 10) == []
    await store.ping()


async def test_memory_store_evicts_least_recently_used_conversations():
    store = MemoryConversationStore(max_conversations=2, ttl=3600)
    await store.append("a", turn(1))
    await store.append("b", turn(2))
    await store.load("a", 10)  # a is now fresher than b
    await store.append("c", turn(3))
    assert await store.load("b", 10) == [] and await store.load("a", 10) and await store.load("c", 10)


@redis_only
async def test_redis_store_caps_conversation_length_and_sets_a_ttl(namespace):
    store = RedisConversationStore(REDIS_URL, ttl=3600, namespace=namespace)
    for i in range(130):
        await store.append("long", turn(i))
    assert len(await store.load("long", 1000)) == 200, "a conversation cannot grow without bound"
    raw = aioredis.from_url(REDIS_URL)
    assert 0 < await raw.ttl(f"{namespace}:chat:long") <= 3600
    await raw.aclose()
    await store.close()
