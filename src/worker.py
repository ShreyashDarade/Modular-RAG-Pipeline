"""Standalone ingestion worker: ``rag-worker`` / ``python -m src.worker``.

Run any number of these (one per GPU/CPU pool) against the same Redis and Elasticsearch; each
job goes to exactly one of them. API replicas then need none of the OCR stack.
"""

from __future__ import annotations

import asyncio
import signal

from prometheus_client import start_http_server

from src.core.config import get_settings
from src.core.container import Container
from src.core.errors import ConfigError
from src.core.logger import configure_logging, logger


async def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    if settings.ingest_backend != "redis":
        raise ConfigError(
            "the standalone worker consumes a shared queue: set INGEST_BACKEND=redis (and REDIS_URL). "
            "With INGEST_BACKEND=inprocess the API process runs ingestion itself."
        )
    container = await Container.build(settings, role="worker")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows: Ctrl-C then surfaces as KeyboardInterrupt instead
            logger.warning("signal handlers are unavailable on this platform; use Ctrl-C to stop")
            break
    try:
        await container.start()
        if settings.worker_metrics_port:
            start_http_server(settings.worker_metrics_port)
            logger.info("worker metrics on :%d", settings.worker_metrics_port)
        await container.start_watchers()
        await container.jobs.run_worker(
            container.handle_job, concurrency=settings.ingest_concurrency, stop=stop
        )
    finally:
        await container.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
