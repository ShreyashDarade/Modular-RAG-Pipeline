"""SDK fixtures: the same engine reached through each transport.

``rag`` is parametrised over the two transports, so every test that uses it runs twice - once with the engine
embedded in the test process, once through the real FastAPI application over an in-process ASGI transport.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from ai_rag_info import AsyncRag, AsyncRagAPI, AsyncRagClient
from src.api.server import create_app
from src.core.config import Settings

MODES = ["embedded", "remote"]


@pytest.fixture
def sdk_settings(make_settings, rag_toml: Path) -> Settings:
    return make_settings(rag_config=rag_toml, ingest_embedded_worker=True)


async def _drop_indices(settings: Settings, run_id: str) -> None:
    import elasticsearch

    es = elasticsearch.AsyncElasticsearch(settings.es_host)
    try:
        names = list(await es.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
        if names:
            await es.indices.delete(index=names, ignore_unavailable=True)
    finally:
        await es.close()


@pytest.fixture(params=MODES)
async def rag(request, sdk_settings: Settings, run_id: str) -> AsyncIterator[AsyncRagAPI]:
    mode: str = request.param
    if mode == "embedded":
        engine = await AsyncRag.create(sdk_settings)
        try:
            yield engine
        finally:
            await engine.aclose()
            await _drop_indices(sdk_settings, run_id)
        return
    app = create_app(sdk_settings)
    async with app.router.lifespan_context(app):
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://rag.test")
        async with AsyncRagClient("http://rag.test", http_client=http, max_retries=0) as client:
            yield client
        await http.aclose()
    await _drop_indices(sdk_settings, run_id)


@pytest.fixture
async def both(sdk_settings: Settings, run_id: str) -> AsyncIterator[tuple[AsyncRag, AsyncRagClient]]:
    """One engine, reached both ways at once: in-process and over HTTP. Whatever one transport says, the
    other must say too."""
    app = create_app(sdk_settings)
    async with app.router.lifespan_context(app):
        embedded = AsyncRag.from_container(app.state.container)
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://rag.test")
        async with AsyncRagClient("http://rag.test", http_client=http, max_retries=0) as remote:
            yield embedded, remote
        await http.aclose()
    await _drop_indices(sdk_settings, run_id)


TEXT = (
    "Quarterly revenue grew twelve percent driven by cloud subscriptions across all regions. "
    "Operating margin improved to eighteen percent after cost reductions in the second half of the year."
)


@pytest.fixture
def report(tmp_path: Path) -> Path:
    path = tmp_path / "finance.txt"
    path.write_text(TEXT)
    return path


async def eventually(check, *, timeout: float = 15.0, interval: float = 0.2):
    """Poll an async predicate until it returns something truthy."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        value = await check()
        if value or asyncio.get_running_loop().time() > deadline:
            return value
        await asyncio.sleep(interval)
