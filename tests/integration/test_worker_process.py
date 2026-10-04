"""``rag-worker`` as a real OS process: consumes the shared queue and shuts down cleanly on SIGTERM."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import redis.asyncio as aioredis
from src.core.types import JobSpec
from src.jobs.redis_streams import RedisStreamsJobBackend

from tests.conftest import ES_URL, REDIS_URL

pytestmark = [pytest.mark.integration, pytest.mark.needs_redis]


async def test_the_worker_process_consumes_jobs_and_exits_cleanly_on_sigterm(
    make_settings, rag_toml: Path, run_id: str, tmp_path: Path
):
    data = tmp_path / "data"
    (data / "alpha").mkdir(parents=True)
    note = data / "alpha" / "note.txt"
    note.write_text("The worker process indexes this note, which is long enough to become a chunk.")
    env = {
        **os.environ,
        "ES_HOST": ES_URL,
        "ES_NUMBER_OF_REPLICAS": "0",
        "ES_REFRESH_INTERVAL": "1s",
        "ES_INDEX_REGISTRY": f"t{run_id}-registry",
        "RAG_CONFIG": str(rag_toml),
        "PLUGINS": '["tests.fake_plugin"]',
        "DATA_DIR": str(data),
        "OCR_ENABLED": "false",
        "INGEST_BACKEND": "redis",
        "REDIS_URL": REDIS_URL,
        "REDIS_NAMESPACE": f"t{run_id}",
        "WORKER_METRICS_PORT": "0",
        "LOG_LEVEL": "WARNING",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "src.worker"],
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    queue = RedisStreamsJobBackend(make_settings(redis_url=REDIS_URL), namespace=f"t{run_id}")
    try:
        await queue.start()
        record = await queue.submit(JobSpec(collection="alpha", path=str(note)))
        done = await queue.wait(record.id, 60)
        assert done.status == "succeeded" and done.result["text_chunks"] >= 1, (
            process.stdout.read1().decode() if process.poll() else done
        )
        process.send_signal(signal.SIGTERM)
        assert await asyncio.to_thread(process.wait, 30) == 0, "SIGTERM drains and exits with status 0"
    finally:
        if process.poll() is None:
            process.kill()
        import elasticsearch

        es = elasticsearch.AsyncElasticsearch(ES_URL)
        names = list(await es.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
        if names:
            await es.indices.delete(index=names, ignore_unavailable=True)
        await es.close()
        redis_client = aioredis.from_url(REDIS_URL)
        for key in [k async for k in redis_client.scan_iter(f"t{run_id}:*")]:
            await redis_client.delete(key)
        await redis_client.aclose()
        await queue.close()


async def test_the_worker_refuses_to_start_without_the_shared_queue(rag_toml: Path):
    env = {
        **os.environ,
        "INGEST_BACKEND": "inprocess",
        "RAG_CONFIG": str(rag_toml),
        "PLUGINS": '["tests.fake_plugin"]',
        "ES_HOST": ES_URL,
    }
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "src.worker"],
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0 and "INGEST_BACKEND=redis" in result.stderr
