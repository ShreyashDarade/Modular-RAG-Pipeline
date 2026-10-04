"""Rate limiters (``RateLimiter`` port). Fixed window, one atomic round trip on Redis.

Fully async (unlike libraries built on a synchronous storage layer, a Redis round trip never
blocks the event loop). Backend failures propagate - the limiter never "fails open" silently.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from src.core.errors import UpstreamError
from src.core.registry import Registries
from src.ports.runtime import RateDecision

if TYPE_CHECKING:
    from src.core.config import Settings

_LUA = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return {count, redis.call('TTL', KEYS[1])}
"""


class MemoryRateLimiter:
    """Per-process limits: with N replicas the effective limit is N times higher. Use the Redis
    backend when limits must hold across replicas."""

    def __init__(self) -> None:
        self._windows: dict[str, tuple[int, int]] = {}
        self._last_sweep = time.monotonic()

    async def hit(self, key: str, limit: int, window_seconds: int) -> RateDecision:
        now = time.time()
        window = int(now // window_seconds)
        retry_after = max(1, int((window + 1) * window_seconds - now) + 1)
        current_window, count = self._windows.get(key, (window, 0))
        if current_window != window:
            count = 0
        count += 1
        self._windows[key] = (window, count)
        self._sweep(window)
        return RateDecision(count <= limit, limit, max(0, limit - count), retry_after)

    def _sweep(self, window: int) -> None:
        if time.monotonic() - self._last_sweep < 60:
            return
        self._last_sweep = time.monotonic()
        self._windows = {k: v for k, v in self._windows.items() if v[0] >= window}

    async def ping(self) -> None:
        return None

    async def close(self) -> None:
        self._windows.clear()


class RedisRateLimiter:
    def __init__(self, url: str, *, namespace: str = "rag") -> None:
        self._client: aioredis.Redis = aioredis.from_url(url, decode_responses=True, health_check_interval=30)
        self._script = self._client.register_script(_LUA)
        self._ns = namespace

    async def hit(self, key: str, limit: int, window_seconds: int) -> RateDecision:
        window = int(time.time() // window_seconds)
        try:
            count, ttl = await self._script(keys=[f"{self._ns}:rl:{key}:{window}"], args=[window_seconds + 1])
        except RedisError as exc:
            raise UpstreamError(f"redis rate limiter failed: {exc}") from exc
        return RateDecision(int(count) <= limit, limit, max(0, limit - int(count)), max(1, int(ttl)))

    async def ping(self) -> None:
        try:
            await self._client.ping()
        except RedisError as exc:
            raise UpstreamError(f"redis ping failed: {exc}") from exc

    async def close(self) -> None:
        await self._client.aclose()


def build_memory_limiter(settings: Settings) -> MemoryRateLimiter:
    return MemoryRateLimiter()


def build_redis_limiter(settings: Settings) -> RedisRateLimiter:
    return RedisRateLimiter(settings.redis_url, namespace=settings.redis_namespace)


def register_builtin_limiters(registries: Registries) -> None:
    registries.rate_limiters.register("memory", build_memory_limiter)
    registries.rate_limiters.register("redis", build_redis_limiter)
