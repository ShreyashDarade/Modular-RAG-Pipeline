"""Cache backends (``Cache`` port) and the helpers built on them.

Backends fail loudly: a Redis outage is an ``UpstreamError``, not a silent cache miss.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, cast

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from src.core.errors import UpstreamError
from src.core.registry import Registries
from src.ports.runtime import Cache
from src.runtime.concurrency import SingleFlight
from src.runtime.metrics import CACHE_REQUESTS

if TYPE_CHECKING:
    from src.core.config import Settings


class MemoryCache:
    """Per-process LRU with per-entry TTL."""

    def __init__(self, max_entries: int) -> None:
        self._max = max_entries
        self._data: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self._counters: dict[str, int] = {}

    async def get(self, key: str) -> bytes | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if expires <= time.monotonic():
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    async def set(self, key: str, value: bytes, ttl: int) -> None:
        self._data[key] = (time.monotonic() + ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def incr(self, key: str) -> int:
        self._counters[key] = self._counters.get(key, 0) + 1
        return self._counters[key]

    async def counter(self, key: str) -> int:
        return self._counters.get(key, 0)

    async def ping(self) -> None:
        return None

    async def close(self) -> None:
        self._data.clear()


class RedisCache:
    """Shared across replicas. One connection pool per process."""

    def __init__(self, url: str, *, namespace: str = "rag") -> None:
        self._client: aioredis.Redis = aioredis.from_url(
            url, decode_responses=False, health_check_interval=30
        )
        self._ns = namespace

    def _k(self, key: str) -> str:
        return f"{self._ns}:cache:{key}"

    async def get(self, key: str) -> bytes | None:
        try:
            value = await self._client.get(self._k(key))
            return cast("bytes | None", value)  # the client is created with decode_responses=False
        except RedisError as exc:
            raise UpstreamError(f"redis cache get failed: {exc}") from exc

    async def set(self, key: str, value: bytes, ttl: int) -> None:
        try:
            await self._client.set(self._k(key), value, ex=ttl)
        except RedisError as exc:
            raise UpstreamError(f"redis cache set failed: {exc}") from exc

    async def delete(self, key: str) -> None:
        try:
            await self._client.delete(self._k(key))
        except RedisError as exc:
            raise UpstreamError(f"redis cache delete failed: {exc}") from exc

    async def incr(self, key: str) -> int:
        try:
            return int(await self._client.incr(self._k(key)))
        except RedisError as exc:
            raise UpstreamError(f"redis cache incr failed: {exc}") from exc

    async def counter(self, key: str) -> int:
        try:
            raw = await self._client.get(self._k(key))
        except RedisError as exc:
            raise UpstreamError(f"redis cache counter failed: {exc}") from exc
        return int(raw) if raw else 0

    async def ping(self) -> None:
        try:
            await self._client.ping()
        except RedisError as exc:
            raise UpstreamError(f"redis ping failed: {exc}") from exc

    async def close(self) -> None:
        await self._client.aclose()


class TieredCache:
    """Small, short-lived in-process L1 in front of a shared L2. Counters live in L2 only so every
    replica agrees on them."""

    def __init__(self, l1: Cache, l2: Cache, l1_ttl: int = 30) -> None:
        self._l1, self._l2, self._l1_ttl = l1, l2, l1_ttl

    async def get(self, key: str) -> bytes | None:
        value = await self._l1.get(key)
        if value is not None:
            return value
        value = await self._l2.get(key)
        if value is not None:
            await self._l1.set(key, value, self._l1_ttl)
        return value

    async def set(self, key: str, value: bytes, ttl: int) -> None:
        await self._l2.set(key, value, ttl)
        await self._l1.set(key, value, min(ttl, self._l1_ttl))

    async def delete(self, key: str) -> None:
        await self._l1.delete(key)
        await self._l2.delete(key)

    async def incr(self, key: str) -> int:
        return await self._l2.incr(key)

    async def counter(self, key: str) -> int:
        return await self._l2.counter(key)

    async def ping(self) -> None:
        await self._l2.ping()

    async def close(self) -> None:
        await self._l1.close()
        await self._l2.close()


def build_memory_cache(settings: Settings) -> Cache:
    return MemoryCache(settings.cache_memory_entries)


def build_redis_cache(settings: Settings) -> Cache:
    return RedisCache(settings.redis_url, namespace=settings.redis_namespace)


def build_tiered_cache(settings: Settings) -> Cache:
    return TieredCache(
        MemoryCache(settings.cache_memory_entries),
        RedisCache(settings.redis_url, namespace=settings.redis_namespace),
    )


class CorpusVersion:
    """Monotonic counter folded into corpus-dependent cache keys. Bumping it (after an ingest or a
    delete) retires every cached retrieval result on every replica at once. Reads are memoised
    for ``refresh_seconds`` so the hot path costs no extra round trip."""

    KEY = "corpus-version"

    def __init__(self, cache: Cache, refresh_seconds: float = 1.0) -> None:
        self._cache = cache
        self._refresh = refresh_seconds
        self._value = 0
        self._read_at = float("-inf")

    async def current(self) -> int:
        now = time.monotonic()
        if now - self._read_at >= self._refresh:
            self._value = await self._cache.counter(self.KEY)
            self._read_at = now
        return self._value

    async def bump(self) -> int:
        self._value = await self._cache.incr(self.KEY)
        self._read_at = time.monotonic()
        return self._value


class CachedCall:
    """``get -> miss -> compute -> set`` with stampede protection and hit/miss metrics."""

    def __init__(self, cache: Cache, name: str) -> None:
        self._cache = cache
        self._name = name
        self._flight = SingleFlight()

    async def get_or_compute(self, key: str, ttl: int, compute: Callable[[], Awaitable[bytes]]) -> bytes:
        cached = await self._cache.get(key)
        if cached is not None:
            CACHE_REQUESTS.labels(self._name, "hit").inc()
            return cached
        CACHE_REQUESTS.labels(self._name, "miss").inc()

        async def fill() -> bytes:
            value = await compute()
            await self._cache.set(key, value, ttl)
            return value

        return await self._flight.do(key, fill)


def register_builtin_caches(registries: Registries) -> None:
    registries.caches.register("memory", build_memory_cache)
    registries.caches.register("redis", build_redis_cache)
    registries.caches.register("tiered", build_tiered_cache)
