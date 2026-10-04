"""The MCP server, exercised with the SDK's own client against a live uvicorn process."""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn
from mcp.client import Client
from src.api.server import create_app

from tests.helpers import make_pdf

pytestmark = pytest.mark.integration

FINANCE = ["Quarterly revenue grew twelve percent driven by cloud subscriptions across all regions."]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def live_server(make_settings, rag_toml: Path, run_id: str, request) -> Iterator[str]:
    extra = getattr(request, "param", {})
    app = create_app(make_settings(rag_config=rag_toml, mcp_enabled=True, **extra))
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    # clean up this run's indices through ES directly
    import elasticsearch

    es = elasticsearch.Elasticsearch(make_settings().es_host)
    names = list(es.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
    if names:
        es.indices.delete(index=names, ignore_unavailable=True)
    es.close()
    server.should_exit = True
    thread.join(10)


async def test_mcp_tools_over_streamable_http(live_server: str, tmp_path: Path):
    pdf = make_pdf(tmp_path / "finance.pdf", FINANCE)
    with pdf.open("rb") as handle:
        uploaded = httpx.post(
            f"{live_server}/api/v1/ingest", files={"file": ("finance.pdf", handle)}, timeout=60
        )
    assert uploaded.status_code == 200, uploaded.text

    async with Client(f"{live_server}/mcp") as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == {
            "list_collections",
            "list_models",
            "list_documents",
            "search_documents",
            "ask_question",
        }
        assert all(t.annotations and t.annotations.read_only_hint for t in tools.values())
        assert "collections" in tools["search_documents"].input_schema["properties"]

        collections = (await client.call_tool("list_collections", {})).structured_content
        names = {c["name"]: c for c in collections["result"]}
        assert names["alpha"]["default"] and names["beta"]["kinds"] == ["text", "table"]

        found = (
            await client.call_tool("search_documents", {"query": "quarterly revenue growth", "limit": 2})
        ).structured_content
        assert (
            found["documents"][0]["source"].endswith("finance.pdf")
            and found["documents"][0]["collection"] == "alpha"
        )
        assert len(found["documents"]) <= 2

        answer = (
            await client.call_tool("ask_question", {"question": "what happened to revenue", "model": "smart"})
        ).structured_content
        assert answer["model"] == "smart" and "finance.pdf" in answer["answer"]

        listing = (await client.call_tool("list_documents", {})).structured_content
        assert listing["total"] == 1

        failed = await client.call_tool("search_documents", {"query": "x", "collections": ["nope"]})
        assert failed.is_error and "unknown collection" in failed.content[0].text


async def test_mcp_is_stateless_so_any_replica_can_answer(live_server: str):
    """No session id is issued or required: a request needs nothing from an earlier one."""
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    first = httpx.post(f"{live_server}/mcp", json=call, headers=headers)
    second = httpx.post(f"{live_server}/mcp", json=call, headers=headers)
    assert first.status_code == 200 and second.status_code == 200
    assert "mcp-session-id" not in {k.lower() for k in first.headers}


@pytest.mark.parametrize("live_server", [{"mcp_allowed_hosts": ["rag.example.com"]}], indirect=True)
def test_mcp_rejects_unlisted_host_headers(live_server: str):
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    refused = httpx.post(f"{live_server}/mcp", json=call, headers=headers)  # Host: 127.0.0.1:port
    assert refused.status_code in (400, 403, 421)
    allowed = httpx.post(f"{live_server}/mcp", json=call, headers={**headers, "Host": "rag.example.com"})
    assert allowed.status_code == 200


def test_mcp_defaults_to_localhost_only_host_headers(live_server: str):
    """With MCP_ALLOWED_HOSTS unset the SDK's DNS-rebinding protection stays on."""
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    assert httpx.post(f"{live_server}/mcp", json=call, headers=headers).status_code == 200
    spoofed = httpx.post(f"{live_server}/mcp", json=call, headers={**headers, "Host": "evil.example.com"})
    assert spoofed.status_code in (400, 403, 421)


async def test_mcp_over_stdio(make_settings, rag_toml: Path, run_id: str, tmp_path: Path):
    """``rag-mcp`` as a desktop MCP client would launch it."""
    import os

    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    env = {
        **os.environ,
        "ES_HOST": make_settings().es_host,
        "ES_NUMBER_OF_REPLICAS": "0",
        "ES_INDEX_REGISTRY": f"t{run_id}-registry",
        "RAG_CONFIG": str(rag_toml),
        "PLUGINS": '["tests.fake_plugin"]',
        "DATA_DIR": str(tmp_path / "data"),
        "CACHE_BACKEND": "memory",
        "LOG_LEVEL": "ERROR",
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server.main"],
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    try:
        async with Client(params) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert {"search_documents", "ask_question", "list_collections"} <= tools
            collections = (await client.call_tool("list_collections", {})).structured_content
            assert {c["name"] for c in collections["result"]} == {"alpha", "beta"}
            empty = (await client.call_tool("search_documents", {"query": "anything"})).structured_content
            assert empty["documents"] == []
    finally:
        import elasticsearch

        es = elasticsearch.Elasticsearch(make_settings().es_host)
        names = list(es.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
        if names:
            es.indices.delete(index=names, ignore_unavailable=True)
        es.close()


@pytest.mark.parametrize(
    "live_server", [{"rate_limit_per_minute": 3, "rate_limit_backend": "memory"}], indirect=True
)
def test_mcp_calls_are_rate_limited_per_client_like_rest(live_server: str):
    """Regression: the MCP mount bypassed the limiter, so ask_question could be called without
    limit while the equivalent REST endpoint was limited."""
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    codes = [httpx.post(f"{live_server}/mcp", json=call, headers=headers).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    limited = httpx.post(f"{live_server}/mcp", json=call, headers=headers)
    assert int(limited.headers["retry-after"]) >= 1 and limited.json()["code"] == "rate_limited"
    assert httpx.get(f"{live_server}/health").status_code == 200


async def test_mcp_tool_calls_carry_the_request_deadline(container, monkeypatch):
    from mcp.client import Client
    from src.mcp_server.server import build_mcp_server

    async def slow(*args, **kwargs):
        await asyncio.sleep(30)

    container.settings.request_timeout_seconds = 1
    monkeypatch.setattr(container.retrieval, "retrieve", slow)
    async with Client(build_mcp_server(lambda: container)) as client:
        result = await client.call_tool("search_documents", {"query": "anything"})
    assert result.is_error and "timeout" in result.content[0].text
