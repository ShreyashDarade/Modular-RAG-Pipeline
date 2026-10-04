"""Small asyncio primitives for staying healthy under load."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Any

from src.core.errors import RequestTimeoutError


class Bulkhead:
    """Bounds concurrent calls to one dependency. Excess callers wait their turn rather than
    stampeding the dependency (and timing each other out)."""

    def __init__(self, limit: int, name: str = "") -> None:
        self.name = name
        self.limit = limit
        self._semaphore = asyncio.Semaphore(limit)

    async def __aenter__(self) -> None:
        await self._semaphore.acquire()

    async def __aexit__(self, *exc_info: object) -> None:
        self._semaphore.release()

    @property
    def in_flight(self) -> int:
        return self.limit - self._semaphore._value


class SingleFlight:
    """Collapse concurrent identical work into one execution (cache-stampede protection).

    The first caller for a key runs ``factory``; everyone arriving while it runs awaits the same
    result. The work is shielded: a cancelled caller (e.g. a client disconnect) does not abort
    it for the others.
    """

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Task[Any]] = {}

    async def do[T](self, key: str, factory: Callable[[], Awaitable[T]]) -> T:
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(factory())
            self._inflight[key] = task

            def forget(_done: asyncio.Future[Any], key: str = key) -> None:
                self._inflight.pop(key, None)

            task.add_done_callback(forget)
        return await asyncio.shield(task)


async def run_all[T](coroutines: Iterable[Awaitable[T]]) -> list[T]:
    """Run concurrently and return results in order. On the first failure the remaining work is
    cancelled and *that* exception is raised as-is (a bare ``TaskGroup`` would wrap it in an
    ``ExceptionGroup``, hiding the typed errors callers switch on)."""
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(_await(c)) for c in coroutines]
    except ExceptionGroup as group_error:
        raise group_error.exceptions[0] from None
    return [task.result() for task in tasks]


async def _await[T](awaitable: Awaitable[T]) -> T:
    return await awaitable


@asynccontextmanager
async def deadline(seconds: float) -> AsyncIterator[None]:
    """Bound the enclosed work to ``seconds``; beyond that it is a ``RequestTimeoutError`` (HTTP 504)."""
    try:
        async with asyncio.timeout(seconds):
            yield
    except TimeoutError:
        raise RequestTimeoutError(f"request did not finish within {seconds:g}s") from None


async def deadline_iter[T](source: AsyncIterator[T], seconds: float) -> AsyncIterator[T]:
    """Yield from ``source``, allowing it ``seconds`` in total (a ``RequestTimeoutError`` beyond that).

    Only the time spent *waiting for the source* counts: a consumer that pauses between items - a slow HTTP
    client, an application doing work per event - does not eat the budget, and a stalled source is still cut
    off. Each pull is bounded on its own, so the iterator may be consumed from a different task than the one
    that created it (as a streaming HTTP response does).
    """
    loop = asyncio.get_running_loop()
    spent = 0.0
    try:
        while True:
            remaining = seconds - spent
            if remaining <= 0:
                raise RequestTimeoutError(f"request did not finish within {seconds:g}s")
            started = loop.time()
            try:
                item = await asyncio.wait_for(anext(source), remaining)
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise RequestTimeoutError(f"request did not finish within {seconds:g}s") from None
            spent += loop.time() - started
            yield item
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            await aclose()
