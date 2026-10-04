from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from src.core.errors import QueueFullError
from src.core.logger import logger
from src.core.registry import Registries
from src.core.types import JobRecord, JobSpec
from src.jobs.common import failure, is_retryable
from src.ports.runtime import JobHandler
from src.runtime.metrics import INGEST_JOBS, INGEST_QUEUE_DEPTH, INGEST_SECONDS

if TYPE_CHECKING:
    from src.core.config import Settings

RETAINED_JOBS = 5000


class InProcessJobBackend:
    """Queue and status live in this process: right for one server process (development, small
    deployments). With several API processes or replicas use the Redis backend - a job's status
    must be visible to whichever replica is asked about it."""

    def __init__(self, settings: Settings) -> None:
        self._max_attempts = settings.ingest_max_attempts
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.ingest_queue_max_size)
        self._jobs: OrderedDict[str, JobRecord] = OrderedDict()
        self._done: dict[str, asyncio.Event] = {}
        self._running = 0
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def submit(self, spec: JobSpec) -> JobRecord:
        record = JobRecord(id=uuid.uuid4().hex, spec=spec)
        try:
            self._queue.put_nowait(record.id)
        except asyncio.QueueFull:
            raise QueueFullError(
                f"ingestion queue is full ({self._queue.maxsize} jobs waiting)", retry_after=5
            ) from None
        self._jobs[record.id] = record
        self._done[record.id] = asyncio.Event()
        while len(self._jobs) > RETAINED_JOBS:  # forget the oldest *finished* jobs
            oldest_id, oldest = next(iter(self._jobs.items()))
            if not oldest.finished:
                break
            del self._jobs[oldest_id]
            self._done.pop(oldest_id, None)
        INGEST_QUEUE_DEPTH.set(self._queue.qsize() + self._running)
        return record

    async def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    async def wait(self, job_id: str, wait_seconds: float) -> JobRecord:
        record = self._jobs[job_id]
        event = self._done.get(job_id)
        if event is not None and not record.finished:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(event.wait(), wait_seconds)
        return record

    async def depth(self) -> int:
        return self._queue.qsize() + self._running

    async def ping(self) -> None:
        return None

    @asynccontextmanager
    async def lock(self, key: str) -> AsyncIterator[None]:
        entry = self._locks.get(key)
        lock, users = entry if entry else (asyncio.Lock(), 0)
        self._locks[key] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            lock, users = self._locks[key]
            if users <= 1:
                del self._locks[key]
            else:
                self._locks[key] = (lock, users - 1)

    async def run_worker(self, handler: JobHandler, *, concurrency: int, stop: asyncio.Event) -> None:
        async def loop() -> None:
            while not stop.is_set():
                try:
                    job_id = await asyncio.wait_for(self._queue.get(), 0.5)
                except TimeoutError:
                    continue
                await self._process(job_id, handler)

        async with asyncio.TaskGroup() as group:
            for _ in range(concurrency):
                group.create_task(loop())

    async def _process(self, job_id: str, handler: JobHandler) -> None:
        record = self._jobs[job_id]
        record.status, record.attempts = "running", record.attempts + 1
        record.started_at = record.started_at or time.time()
        self._running += 1
        started = time.perf_counter()
        try:
            record.result = await handler(record.spec)
            record.status = "succeeded"
        except Exception as exc:
            logger.error("job %s failed (attempt %d)", job_id, record.attempts, exc_info=exc)
            if is_retryable(exc) and record.attempts < self._max_attempts:
                record.status = "queued"
                self._queue.put_nowait(job_id)
                INGEST_JOBS.labels("retried").inc()
                return
            failure(record, exc)
        finally:
            self._running -= 1
            INGEST_QUEUE_DEPTH.set(self._queue.qsize() + self._running)
        record.finished_at = time.time()
        INGEST_SECONDS.observe(time.perf_counter() - started)
        INGEST_JOBS.labels(record.status).inc()
        self._done[job_id].set()


def build_inprocess_backend(settings: Settings) -> InProcessJobBackend:
    return InProcessJobBackend(settings)


def register_inprocess(registries: Registries) -> None:
    registries.job_backends.register("inprocess", build_inprocess_backend)
