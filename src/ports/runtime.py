from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from src.core.types import ChatMessage, JobRecord, JobSpec


class Cache(Protocol):
    """Byte-valued cache shared by replicas when backed by Redis. Errors propagate."""

    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl: int) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def incr(self, key: str) -> int:
        """Atomically increment a shared counter and return the new value."""

    async def counter(self, key: str) -> int:
        """Current value of a counter (0 if never incremented), never served from a local tier."""

    async def ping(self) -> None: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RateDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int


class RateLimiter(Protocol):
    async def hit(self, key: str, limit: int, window_seconds: int) -> RateDecision: ...

    async def ping(self) -> None: ...

    async def close(self) -> None: ...


class ConversationStore(Protocol):
    async def load(self, conversation_id: str, limit: int) -> list[ChatMessage]:
        """The most recent ``limit`` messages, oldest first. Empty list for an unknown id."""

    async def append(self, conversation_id: str, messages: Sequence[ChatMessage]) -> None: ...

    async def delete(self, conversation_id: str) -> bool: ...

    async def ping(self) -> None: ...

    async def close(self) -> None: ...


JobHandler = Callable[[JobSpec], Awaitable[dict[str, Any]]]


class JobBackend(Protocol):
    """Queue + status store for ingestion jobs. At-least-once delivery: handlers are idempotent."""

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def submit(self, spec: JobSpec) -> JobRecord:
        """Raises ``QueueFullError`` when the backlog is at capacity."""

    async def get(self, job_id: str) -> JobRecord | None: ...

    async def wait(self, job_id: str, wait_seconds: float) -> JobRecord:
        """Block until the job finishes or ``wait_seconds`` elapse (then return its current state)."""

    async def run_worker(self, handler: JobHandler, *, concurrency: int, stop: asyncio.Event) -> None:
        """Consume jobs until ``stop`` is set, then drain in-flight jobs and return."""

    async def depth(self) -> int: ...

    def lock(self, key: str) -> AbstractAsyncContextManager[None]:
        """Mutual exclusion across every worker/replica using this backend (per-document locking)."""

    async def ping(self) -> None: ...
