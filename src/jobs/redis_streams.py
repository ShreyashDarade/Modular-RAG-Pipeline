"""Distributed ingestion queue on Redis Streams.

* API replicas ``submit`` (XADD) and read status; any number of worker processes consume through
  one consumer group, so each job goes to exactly one worker at a time.
* A worker that dies mid-job leaves the entry pending; once idle longer than the visibility
  timeout another worker takes it over (XAUTOCLAIM). Workers heartbeat running jobs so a long
  OCR job is not mistaken for a dead one.
* Delivery is at-least-once and ingestion is idempotent, so a take-over is safe. Retries are
  capped (``ingest_max_attempts``) - a job that keeps killing its worker ends up ``failed``.
* Per-document locks (SET NX PX with renewal) stop two workers ingesting the same file at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import redis.asyncio as aioredis
from redis.exceptions import RedisError, ResponseError

from src.core.errors import QueueFullError, UpstreamError
from src.core.logger import logger
from src.core.registry import Registries
from src.core.types import JobRecord, JobSpec
from src.jobs.common import decode, encode, failure, is_retryable
from src.ports.runtime import JobHandler
from src.runtime.metrics import INGEST_JOBS, INGEST_QUEUE_DEPTH, INGEST_SECONDS

if TYPE_CHECKING:
    from src.core.config import Settings

GROUP = "workers"
LOCK_TTL_MS = 60_000
_RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
_RENEW = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"


def _wrap(exc: RedisError) -> UpstreamError:
    return UpstreamError(f"redis job queue failed: {exc}")


class RedisStreamsJobBackend:
    def __init__(self, settings: Settings, *, namespace: str = "rag") -> None:
        self._s = settings
        self._client: aioredis.Redis = aioredis.from_url(
            settings.redis_url, decode_responses=True, health_check_interval=30
        )
        self._stream = f"{namespace}:ingest:stream"
        self._job_prefix = f"{namespace}:job:"
        self._lock_prefix = f"{namespace}:lock:"
        self._consumer = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self._release = self._client.register_script(_RELEASE)
        self._renew = self._client.register_script(_RENEW)

    # --- lifecycle -------------------------------------------------------------------------
    async def start(self) -> None:
        try:
            await self._client.xgroup_create(self._stream, GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise _wrap(exc) from exc
        except RedisError as exc:
            raise _wrap(exc) from exc

    async def close(self) -> None:
        await self._client.aclose()

    async def ping(self) -> None:
        try:
            await self._client.ping()
        except RedisError as exc:
            raise _wrap(exc) from exc

    # --- producer side ---------------------------------------------------------------------
    async def submit(self, spec: JobSpec) -> JobRecord:
        record = JobRecord(id=uuid.uuid4().hex, spec=spec)
        try:
            if await self._client.xlen(self._stream) >= self._s.ingest_queue_max_size:
                raise QueueFullError(
                    f"ingestion queue is full ({self._s.ingest_queue_max_size} jobs)", retry_after=5
                )
            await self._save(record)
            await self._client.xadd(self._stream, {"job_id": record.id})
        except RedisError as exc:
            raise _wrap(exc) from exc
        return record

    async def get(self, job_id: str) -> JobRecord | None:
        try:
            raw = await self._client.get(self._job_prefix + job_id)
        except RedisError as exc:
            raise _wrap(exc) from exc
        return decode(raw) if raw else None

    async def wait(self, job_id: str, wait_seconds: float) -> JobRecord:
        deadline = time.monotonic() + wait_seconds
        delay = 0.1
        while True:
            record = await self.get(job_id)
            if record is None:
                raise UpstreamError(f"job {job_id} disappeared from the queue store")
            if record.finished or time.monotonic() >= deadline:
                return record
            await asyncio.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 1.5, 1.0)

    async def depth(self) -> int:
        try:
            return int(await self._client.xlen(self._stream))
        except RedisError as exc:
            raise _wrap(exc) from exc

    async def _save(self, record: JobRecord) -> None:
        await self._client.set(self._job_prefix + record.id, encode(record), ex=self._s.job_ttl_seconds)

    # --- locks -----------------------------------------------------------------------------
    @asynccontextmanager
    async def lock(self, key: str) -> AsyncIterator[None]:
        name, token = self._lock_prefix + key, uuid.uuid4().hex
        try:
            while not await self._client.set(name, token, nx=True, px=LOCK_TTL_MS):  # noqa: ASYNC110 - polling a remote lock
                await asyncio.sleep(0.25)
        except RedisError as exc:
            raise _wrap(exc) from exc

        async def keep_alive() -> None:
            while True:
                await asyncio.sleep(LOCK_TTL_MS / 3000)
                if not await self._renew(keys=[name], args=[token, LOCK_TTL_MS]):
                    logger.error("lost lock %s while still holding it", key)
                    return

        renewer = asyncio.create_task(keep_alive())
        try:
            yield
        finally:
            renewer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renewer
            with contextlib.suppress(RedisError):  # the TTL frees it anyway
                await self._release(keys=[name], args=[token])

    # --- consumer side ---------------------------------------------------------------------
    async def run_worker(self, handler: JobHandler, *, concurrency: int, stop: asyncio.Event) -> None:
        await self.start()
        slots = asyncio.Semaphore(concurrency)
        running: set[asyncio.Task[None]] = set()
        visibility_ms = self._s.job_visibility_timeout_seconds * 1000
        last_claim = 0.0
        logger.info("worker %s consuming %s (concurrency=%d)", self._consumer, self._stream, concurrency)
        try:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(slots.acquire(), 0.5)
                except TimeoutError:
                    continue
                entries: Any = []
                try:
                    if time.monotonic() - last_claim >= self._s.job_visibility_timeout_seconds / 3:
                        last_claim = time.monotonic()
                        _, claimed, _ = await self._client.xautoclaim(
                            self._stream,
                            GROUP,
                            self._consumer,
                            min_idle_time=visibility_ms,
                            start_id="0-0",
                            count=1,
                        )
                        entries = claimed
                    if not entries:
                        read: Any = await self._client.xreadgroup(
                            GROUP, self._consumer, {self._stream: ">"}, count=1, block=1000
                        )
                        entries = read[0][1] if read else []
                except RedisError as exc:
                    slots.release()
                    logger.error("job queue read failed: %s", exc)
                    if "NOGROUP" in str(exc) or "UNBLOCKED" in str(
                        exc
                    ):  # stream or group was deleted under us
                        await self.start()
                    await asyncio.sleep(1)
                    continue
                if not entries:
                    slots.release()
                    continue
                entry_id, fields = entries[0]
                task = asyncio.create_task(self._process(entry_id, fields["job_id"], handler, slots))
                running.add(task)
                task.add_done_callback(running.discard)
            if running:
                logger.info("draining %d running job(s)", len(running))
                await asyncio.gather(*running, return_exceptions=True)
        except asyncio.CancelledError:
            # Cancelled (e.g. the process is going down): stop in-flight jobs *without* acking
            # them, so another worker takes them over after the visibility timeout.
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            raise

    async def _process(
        self, entry_id: str, job_id: str, handler: JobHandler, slots: asyncio.Semaphore
    ) -> None:
        try:
            record = await self.get(job_id)
            if record is None:  # expired while queued
                await self._ack(entry_id)
                return
            if record.attempts >= self._s.ingest_max_attempts:
                record.status, record.error = (
                    "failed",
                    f"gave up after {record.attempts} attempts (worker kept dying?)",
                )
                record.finished_at = time.time()
                await self._save(record)
                INGEST_JOBS.labels("failed").inc()
                await self._ack(entry_id)
                return
            record.status, record.attempts = "running", record.attempts + 1
            record.started_at = record.started_at or time.time()
            await self._save(record)
            heartbeat = asyncio.create_task(self._heartbeat(entry_id))
            started = time.perf_counter()
            try:
                record.result = await handler(record.spec)
                record.status = "succeeded"
            except Exception as exc:
                logger.error("job %s failed (attempt %d)", job_id, record.attempts, exc_info=exc)
                if is_retryable(exc) and record.attempts < self._s.ingest_max_attempts:
                    record.status = "queued"
                    await self._save(record)
                    await self._client.xadd(self._stream, {"job_id": job_id})
                    INGEST_JOBS.labels("retried").inc()
                    await self._ack(entry_id)
                    return
                failure(record, exc)
            finally:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
            record.finished_at = time.time()
            await self._save(record)
            INGEST_SECONDS.observe(time.perf_counter() - started)
            INGEST_JOBS.labels(record.status).inc()
            await self._ack(entry_id)
        except RedisError as exc:
            # leave the entry pending: another worker takes it over after the visibility timeout
            logger.error("job %s bookkeeping failed, leaving it for take-over: %s", job_id, exc)
        finally:
            slots.release()
            with contextlib.suppress(RedisError):
                INGEST_QUEUE_DEPTH.set(await self._client.xlen(self._stream))

    async def _heartbeat(self, entry_id: str) -> None:
        """Reset the entry's idle time so it is not taken over while we are still working on it."""
        interval = self._s.job_visibility_timeout_seconds / 3
        while True:
            await asyncio.sleep(interval)
            try:
                await self._client.xclaim(
                    self._stream, GROUP, self._consumer, min_idle_time=0, message_ids=[entry_id]
                )
            except RedisError as exc:
                logger.warning("heartbeat failed: %s", exc)

    async def _ack(self, entry_id: str) -> None:
        await self._client.xack(self._stream, GROUP, entry_id)
        await self._client.xdel(self._stream, entry_id)


def build_redis_backend(settings: Settings) -> RedisStreamsJobBackend:
    return RedisStreamsJobBackend(settings, namespace=settings.redis_namespace)


def register_redis(registries: Registries) -> None:
    registries.job_backends.register("redis", build_redis_backend)
