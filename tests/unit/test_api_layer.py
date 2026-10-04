"""Middleware, error mapping, storage hardening and the slim-image guarantee."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.errors import install_error_handlers
from src.api.middleware import RequestContextMiddleware
from src.core.errors import (
    InvalidRequestError,
    ModelError,
    NotFoundError,
    OverloadedError,
    PayloadTooLargeError,
    RateLimitedError,
    RequestTimeoutError,
    UpstreamError,
)
from src.core.logger import request_id_var
from src.core.specs import CollectionSpec
from src.ingestion.storage import DataStore, compute_checksum


# --- middleware -----------------------------------------------------------------------------
def make_app(max_in_flight: int = 0) -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)
    app.add_middleware(RequestContextMiddleware, max_in_flight=max_in_flight)

    @app.get("/slow")
    async def slow():
        await asyncio.sleep(0.2)
        return {"request_id": request_id_var.get()}

    @app.get("/items/{item_id}")
    async def item(item_id: int):
        return {"id": item_id}

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("secret internal detail: password=hunter2")

    return app


async def client_for(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t"
    )


async def test_request_ids_are_generated_propagated_and_visible_in_the_request():
    async with await client_for(make_app()) as client:
        generated = await client.get("/items/1")
        assert len(generated.headers["x-request-id"]) == 32
        supplied = await client.get("/slow", headers={"X-Request-ID": "trace-abc-123"})
        assert (
            supplied.headers["x-request-id"] == "trace-abc-123"
            and supplied.json()["request_id"] == "trace-abc-123"
        )


async def test_load_shedding_rejects_excess_work_with_retry_after_but_never_ops_endpoints():
    async with await client_for(make_app(max_in_flight=2)) as client:
        responses = await asyncio.gather(*(client.get("/slow") for _ in range(6)), client.get("/health"))
    slow = responses[:-1]
    assert sorted(r.status_code for r in slow) == [200, 200, 503, 503, 503, 503]
    shed = next(r for r in slow if r.status_code == 503)
    assert shed.headers["retry-after"] == "1" and shed.json()["code"] == "overloaded"
    assert responses[-1].status_code == 200, "health checks bypass admission control"


async def test_shedding_is_released_when_requests_finish():
    async with await client_for(make_app(max_in_flight=1)) as client:
        assert (await client.get("/items/1")).status_code == 200
        assert (await client.get("/items/2")).status_code == 200


async def test_metrics_use_route_templates_not_raw_paths():
    from prometheus_client import generate_latest

    async with await client_for(make_app()) as client:
        for i in (1, 2, 3):
            await client.get(f"/items/{i}")
        await client.get("/does-not-exist")
    text = generate_latest().decode()
    assert 'route="/items/{item_id}"' in text and "/items/1" not in text, "bounded label cardinality"
    assert 'route="unmatched"' in text


async def test_unexpected_errors_never_leak_details():
    async with await client_for(make_app()) as client:
        response = await client.get("/boom")
    assert response.status_code == 500 and response.json() == {
        "detail": "Internal server error",
        "code": "internal_error",
    }
    assert "hunter2" not in response.text


# --- error mapping --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (InvalidRequestError("bad"), 400, "invalid_request"),
        (NotFoundError("gone"), 404, "not_found"),
        (PayloadTooLargeError("big"), 413, "payload_too_large"),
        (RateLimitedError("slow down", retry_after=7), 429, "rate_limited"),
        (OverloadedError("busy", retry_after=3), 503, "overloaded"),
        (RequestTimeoutError("late"), 504, "timeout"),
        (UpstreamError("raw dependency text sk-123"), 502, "upstream_error"),
        (ModelError("gpt: AuthenticationError sk-123"), 502, "model_error"),
    ],
)
def test_typed_errors_map_to_http(error, status, code):
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/x")
    async def x():
        raise error

    response = TestClient(app).get("/x")
    assert response.status_code == status and response.json()["code"] == code
    assert "sk-123" not in response.text, "upstream error text is withheld from callers"
    if hasattr(error, "retry_after"):
        assert response.headers["retry-after"] == str(error.retry_after)


# --- storage --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("/abs/path/file.txt", "file.txt"),
        ("C:\\Users\\x\\file.docx", "file.docx"),
        ("dir/sub/../file.csv", "file.csv"),
        ("name with spaces.pdf", "name with spaces.pdf"),
        ("ünïcode-हिंदी.pdf", "ünïcode-हिंदी.pdf"),
    ],
)
def test_safe_name_reduces_to_a_plain_file_name(given, expected):
    assert DataStore.safe_name(given) == expected


@pytest.mark.parametrize("bad", [None, "", "   ", ".hidden", "..", ".", "a\x00b.pdf", "x" * 300, "dir/"])
def test_safe_name_rejects_the_unacceptable(bad):
    with pytest.raises(InvalidRequestError):
        DataStore.safe_name(bad)


def test_store_saves_atomically_in_the_collection_directory(tmp_path):
    store = DataStore(tmp_path / "data", max_bytes=1000)
    spec = CollectionSpec(name="hr", embedding_model="e")
    path = store.save(spec, "a.txt", io.BytesIO(b"hello"))
    assert (
        path == (tmp_path / "data" / "hr" / "a.txt").resolve() or path == tmp_path / "data" / "hr" / "a.txt"
    )
    assert path.read_bytes() == b"hello"
    store.save(spec, "a.txt", io.BytesIO(b"replaced"))
    assert path.read_bytes() == b"replaced" and [p.name for p in path.parent.iterdir()] == ["a.txt"], (
        "no temp files left"
    )


def test_store_enforces_the_size_limit_while_streaming_and_cleans_up(tmp_path):
    store = DataStore(tmp_path / "data", max_bytes=100)
    spec = CollectionSpec(name="hr", embedding_model="e")
    store.save(spec, "keep.txt", io.BytesIO(b"x" * 50))
    with pytest.raises(PayloadTooLargeError):
        store.save(spec, "keep.txt", io.BytesIO(b"y" * 101))
    assert (tmp_path / "data" / "hr" / "keep.txt").read_bytes() == b"x" * 50, (
        "a rejected upload must not clobber the existing file"
    )
    assert [p.name for p in (tmp_path / "data" / "hr").iterdir()] == ["keep.txt"]


def test_collection_directories_cannot_escape_the_data_dir(tmp_path):
    store = DataStore(tmp_path / "data", max_bytes=10)
    with pytest.raises(InvalidRequestError, match="outside the data directory"):
        store.directory(CollectionSpec(name="evil", embedding_model="e", data_subdir="../../outside"))
    assert (
        store.directory(CollectionSpec(name="legacy", embedding_model="e", data_subdir=""))
        == (tmp_path / "data").resolve()
    )
    assert store.within_data_dir(tmp_path / "data" / "x.pdf") and not store.within_data_dir(
        tmp_path / "x.pdf"
    )


def test_checksum_is_stable_and_content_sensitive(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x" * 3_000_000)
    b.write_bytes(b"x" * 2_999_999 + b"y")
    assert compute_checksum(a) == compute_checksum(a) != compute_checksum(b)


# --- the slim API image ---------------------------------------------------------------------
def test_api_imports_and_wires_without_the_worker_stack():
    """API replicas must not need numpy, OpenCV, torch, scikit-learn, PyMuPDF, pandas or watchdog."""
    script = textwrap.dedent(
        """
        import sys
        for name in ("numpy", "cv2", "torch", "easyocr", "sklearn", "scipy", "pymupdf", "fitz", "pandas",
                     "watchdog", "PIL", "docx", "openpyxl", "tabulate"):
            sys.modules[name] = None          # any import of these now raises ImportError
        from src.api.server import create_app
        from src.core.bootstrap import build_registries
        from src.core.config import Settings
        from src.parsing.registry import ParserSet
        settings = Settings(_env_file=None, plugins=[])
        app = create_app(settings)
        extensions = ParserSet(build_registries(settings), settings).extensions
        assert ".pdf" in extensions and ".docx" in extensions and ".png" in extensions
        import src.mcp_server.server, src.cli.app, src.worker
        print("SLIM-OK", len(extensions))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0 and "SLIM-OK" in result.stdout, result.stderr[-2000:]


def test_the_worker_refuses_to_start_against_the_in_process_queue(monkeypatch):
    from src import worker
    from src.core.config import get_settings
    from src.core.errors import ConfigError

    monkeypatch.setenv("INGEST_BACKEND", "inprocess")
    get_settings.cache_clear()
    try:
        with pytest.raises(ConfigError, match="INGEST_BACKEND=redis"):
            asyncio.run(worker.run())
    finally:
        get_settings.cache_clear()


def test_job_error_text_is_safe_for_callers():
    from src.core.types import JobRecord, JobSpec
    from src.jobs.common import decode, encode, failure

    record = JobRecord(id="1", spec=JobSpec(collection="c", path="/p", kinds=("text", "table")))
    failure(record, UpstreamError("elasticsearch said: api_key=SECRET"))
    assert (
        record.error_status == 502 and record.error_code == "upstream_error" and "SECRET" not in record.error
    )
    failure(record, RuntimeError("password=hunter2"))
    assert record.error_status == 500 and "hunter2" not in record.error
    roundtrip = decode(encode(record))
    assert roundtrip == record and roundtrip.spec.kinds == ("text", "table"), (
        "tuples survive the JSON round trip"
    )
    assert json.loads(encode(record))["spec"]["kinds"] == ["text", "table"]


async def test_cached_retrievals_are_keyed_by_the_settings_that_shape_them():
    """A shared cache outlives deploys; results computed under other settings must not be served."""
    from src.core.config import Settings
    from src.core.types import RetrievalScope
    from src.retrieval.pipeline import RetrievalPipeline
    from src.retrieval.rerank import IdentityReranker
    from src.runtime.cache import CachedCall, CorpusVersion, MemoryCache

    class Retriever:
        def resolve_scope(self, *a):
            return RetrievalScope(("c",))

        async def retrieve_many(self, queries, scope):
            return [[] for _ in queries]

    class Expander:
        async def expand(self, query):
            return [query]

    shared = MemoryCache(100)
    keys = []
    for alpha, fingerprint in (
        (0.5, "llm/heuristic/c"),
        (0.9, "llm/heuristic/c"),
        (0.5, "identity/heuristic/c"),
    ):
        before = set(shared._data)  # noqa: SLF001
        pipeline = RetrievalPipeline(
            expander=Expander(),
            retriever=Retriever(),
            reranker=IdentityReranker(),
            cache=CachedCall(shared, "t"),
            corpus=CorpusVersion(shared),
            settings=Settings(_env_file=None, hybrid_alpha=alpha),
            fingerprint=fingerprint,
        )
        await pipeline.retrieve("same query", RetrievalScope(("c",)))
        (new_key,) = set(shared._data) - before  # noqa: SLF001
        keys.append(new_key)
    assert len(set(keys)) == 3, "different alpha or expander/reranker choice => different cache entries"
