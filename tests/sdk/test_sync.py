"""The blocking SDK against a real server process (RagClient) and against an embedded engine (Rag)."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from turinton_rag import Rag, RagClient
from turinton_rag.errors import ConfigError, ConnectionFailedError, NotFoundError, UsageError

from tests.conftest import ES_URL
from tests.sdk.conftest import TEXT

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def live_server(rag_toml: Path, run_id: str, tmp_path: Path) -> Iterator[str]:
    """`python -m src.api.main` as its own process, as in production."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "HOST": "127.0.0.1",
        "PORT": str(port),
        "ES_HOST": ES_URL,
        "ES_NUMBER_OF_REPLICAS": "0",
        "ES_REFRESH_INTERVAL": "1s",
        "ES_INDEX_REGISTRY": f"t{run_id}-registry",
        "RAG_CONFIG": str(rag_toml),
        "PLUGINS": '["tests.fake_plugin"]',
        "DATA_DIR": str(tmp_path / "data"),
        "OCR_ENABLED": "false",
        "CACHE_BACKEND": "memory",
        "RATE_LIMIT_PER_MINUTE": "0",
        "LOG_LEVEL": "WARNING",
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "src.api.main"], cwd=ROOT, env=env, stderr=subprocess.PIPE
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 60
        while True:
            try:
                if httpx.get(f"{url}/ready", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if server.poll() is not None or time.time() > deadline:
                raise RuntimeError(
                    "server did not start: " + (server.stderr.read().decode() if server.stderr else "")
                )
            time.sleep(0.3)
        yield url
    finally:
        server.terminate()
        try:
            server.wait(20)
        except subprocess.TimeoutExpired:
            server.kill()
        import elasticsearch

        es = elasticsearch.Elasticsearch(ES_URL)
        names = list(es.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
        if names:
            es.indices.delete(index=names, ignore_unavailable=True)
        es.close()


def test_the_blocking_client_covers_the_whole_interface_against_a_real_server(
    live_server: str, tmp_path: Path
):
    report = tmp_path / "finance.txt"
    report.write_text(TEXT)
    with RagClient(live_server) as rag:
        ingested = rag.documents.ingest(report)
        assert ingested.status == "succeeded" and ingested.status_url and ingested.text_chunks >= 1
        assert rag.documents.list().total == 1
        assert rag.jobs.wait(ingested.job_id).status == "succeeded"

        top = rag.retrieve("quarterly revenue growth").documents[0]
        assert "revenue" in top.content and top.source and top.source.endswith("finance.txt")
        answer = rag.ask("what happened to revenue", model="smart")
        assert answer.model == "smart" and answer.context

        turn = rag.chat.send("How did revenue change?")
        events = list(rag.chat.stream("And the margin?", conversation_id=turn.conversation_id))
        assert type(events[0]).__name__ == "ChatStartEvent" and type(events[-1]).__name__ == "ChatEndEvent"
        assert [m.role for m in rag.chat.get(turn.conversation_id).messages] == ["user", "assistant"] * 2
        rag.chat.delete(turn.conversation_id)

        assert {c.name for c in rag.collections.list()} == {"alpha", "beta"} and rag.models().chat
        with pytest.raises(NotFoundError) as caught:
            rag.retrieve("x", collections=["missing"])
        assert caught.value.code == "not_found" and caught.value.status_code == 404
        assert caught.value.request_id, "the server's request id travels with the error"
        assert rag.documents.delete(ingested.source).deleted_count >= 1
    with pytest.raises(UsageError, match="closed"):
        rag.retrieve("x")


def test_abandoning_a_blocking_stream_early_leaves_the_client_usable(live_server: str, tmp_path: Path):
    report = tmp_path / "finance.txt"
    report.write_text(TEXT)
    with RagClient(live_server) as rag:
        rag.documents.ingest(report)
        for event in rag.chat.stream("Summarise revenue"):
            if type(event).__name__ == "ChatDeltaEvent":
                break
        assert rag.retrieve("revenue").documents, (
            "the connection from the abandoned stream did not wedge the client"
        )


def test_an_unreachable_server_is_a_typed_error_not_a_hang():
    with RagClient("http://127.0.0.1:9", max_retries=1, timeout=2) as rag:
        with pytest.raises(ConnectionFailedError, match="cannot reach"):
            rag.retrieve("q")


async def test_the_blocking_client_refuses_to_run_inside_an_event_loop(live_server: str):
    rag = await asyncio.to_thread(RagClient, live_server)  # building it is fine from a worker thread
    try:
        with pytest.raises(UsageError, match="running an event loop"):
            rag.retrieve("q")
    finally:
        await asyncio.to_thread(rag.close)


def test_the_embedded_blocking_engine(sdk_settings, tmp_path: Path):
    report = tmp_path / "finance.txt"
    report.write_text(TEXT)
    with Rag(sdk_settings) as rag:
        try:
            assert rag.documents.ingest(report).status == "succeeded"
            assert rag.retrieve("quarterly revenue").documents[0].source.endswith("finance.txt")
            assert [type(e).__name__ for e in rag.chat.stream("revenue?")][-1] == "ChatEndEvent"
            assert rag.engine.config.default_collection == "alpha", (
                "the composition root is reachable for apps that need it"
            )

            cases = tmp_path / "cases.jsonl"
            cases.write_text(
                json.dumps(
                    {"id": "q1", "query": "quarterly revenue growth", "relevant": [{"source": "finance.txt"}]}
                )
                + "\n"
            )
            evaluation = rag.evaluate(cases, ks=[1, 3], name="sdk")
            assert (
                evaluation.name == "sdk"
                and evaluation.metrics["hit@3"].mean == 1.0
                and evaluation.errors == 0
            )
        finally:
            rag.documents.delete(str(next(iter((tmp_path / "data").rglob("finance.txt")), report)))
    with pytest.raises(UsageError, match="closed"):
        rag.models()


async def test_an_engine_without_ingestion_says_so_instead_of_queueing_forever(
    sdk_settings, run_id, tmp_path
):
    from turinton_rag import AsyncRag

    from tests.sdk.conftest import _drop_indices

    engine = await AsyncRag.create(sdk_settings, ingestion=False)
    try:
        with pytest.raises(ConfigError, match="no ingestion worker"):
            await engine.documents.ingest(b"x", filename="a.txt")
        assert (await engine.retrieve("anything")).documents == [], "search and chat still work"
    finally:
        await engine.aclose()
        await _drop_indices(sdk_settings, run_id)
