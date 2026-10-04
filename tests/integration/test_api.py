"""The HTTP surface, through the real app and its real lifespan (TOML config, live Elasticsearch)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.api.server import create_app

from tests.helpers import make_pdf

pytestmark = pytest.mark.integration

FINANCE = [
    "Quarterly revenue grew twelve percent driven by cloud subscriptions across all regions.\n"
    "Operating margin improved to eighteen percent after cost reductions in the second half.",
]


@pytest.fixture
def client(make_settings, rag_toml: Path, run_id: str):
    settings = make_settings(rag_config=rag_toml)
    with TestClient(create_app(settings)) as test_client:
        yield test_client
        es = test_client.app.state.container.elastic.client

        async def cleanup():
            names = list(await es.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
            if names:
                await es.indices.delete(index=names, ignore_unavailable=True)

        test_client.portal.call(cleanup)  # runs on the app's own event loop


def upload(client: TestClient, path: Path, **params):
    data = {k: v for k, v in params.items() if k in ("collection", "image_language", "kinds")}
    query = {k: v for k, v in params.items() if k in ("force", "wait")}
    with path.open("rb") as handle:
        return client.post(
            "/api/v1/ingest", files={"file": (params.get("name", path.name), handle)}, data=data, params=query
        )


def test_health_ready_status_and_catalog(client: TestClient):
    assert client.get("/health").json()["status"] == "healthy"
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["checks"]["elasticsearch"] == "ok"

    status = client.get("/api/v1/status").json()
    assert (
        status["default_collection"] == "alpha" and "pdf" in status["parsers"] and "docx" in status["parsers"]
    )
    assert "api_key" not in json.dumps(status).lower()

    collections = {c["name"]: c for c in client.get("/api/v1/collections").json()}
    assert collections["alpha"]["default"] and collections["beta"]["kinds"] == ["text", "table"]
    models = client.get("/api/v1/models").json()
    assert {m["name"] for m in models["chat"]} == {"fast", "smart"}
    assert {m["name"]: m["dimensions"] for m in models["embedding"]} == {"hash64": 64, "hash32": 32}


def test_ingest_search_ask_chat_delete_flow(client: TestClient, tmp_path: Path):
    pdf = make_pdf(tmp_path / "finance.pdf", FINANCE)
    response = upload(client, pdf)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "succeeded" and body["text_chunks"] >= 1 and body["collection"] == "alpha"
    assert client.get(body["status_url"]).json()["status"] == "succeeded"

    again = upload(client, pdf).json()
    assert again["skipped_reason"] == "no_changes_detected" and again["reindexed"] is False

    docs = client.get("/api/v1/documents", params={"collection": "alpha"}).json()
    assert docs["total"] == 1 and docs["documents"][0]["parser"] == "pdf"

    retrieved = client.post("/api/v1/retrieve", json={"query": "quarterly revenue growth"})
    assert retrieved.status_code == 200
    payload = retrieved.json()
    assert payload["documents"][0]["collection"] == "alpha" and payload["documents"][0]["type"] == "pdf_text"
    assert retrieved.headers["x-request-id"]

    asked = client.post("/api/v1/ask", json={"query": "revenue growth", "model": "smart"}).json()
    assert asked["model"] == "smart" and "finance.pdf" in asked["answer"] and asked["context"][0]["rank"] == 1

    turn1 = client.post("/api/v1/chat", json={"message": "how did revenue do?"}).json()
    turn2 = client.post(
        "/api/v1/chat", json={"message": "and margin?", "conversation_id": turn1["conversation_id"]}
    ).json()
    assert turn2["standalone_query"].startswith("STANDALONE")
    convo = client.get(f"/api/v1/chat/{turn1['conversation_id']}").json()
    assert len(convo["messages"]) == 4
    assert client.delete(f"/api/v1/chat/{turn1['conversation_id']}").status_code == 204
    assert client.get(f"/api/v1/chat/{turn1['conversation_id']}").status_code == 404

    deleted = client.request("DELETE", "/api/v1/documents", params={"source": body["source"]}).json()
    assert deleted["success"] and deleted["deleted_count"] >= 1
    assert (
        client.post("/api/v1/retrieve", json={"query": "quarterly revenue growth"}).json()["documents"] == []
    )


def test_chat_streaming_sse(client: TestClient, tmp_path: Path):
    upload(client, make_pdf(tmp_path / "finance.pdf", FINANCE))
    with client.stream("POST", "/api/v1/chat/stream", json={"message": "revenue growth"}) as response:
        assert response.status_code == 200 and response.headers["content-type"].startswith(
            "text/event-stream"
        )
        raw = "".join(response.iter_text())
    events = [
        (
            block.split("\n")[0].removeprefix("event: "),
            json.loads(block.split("\n")[1].removeprefix("data: ")),
        )
        for block in raw.strip().split("\n\n")
    ]
    kinds = [e[0] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "end" and "delta" in kinds
    assert events[0][1]["conversation_id"] and events[0][1]["context"]
    assert "".join(e[1]["text"] for e in events if e[0] == "delta").strip() == events[-1][1]["answer"].strip()

    # an unknown conversation is a real 404, not an error event inside a 200 stream
    bad = client.post("/api/v1/chat/stream", json={"message": "x", "conversation_id": "nope"})
    assert bad.status_code == 404 and bad.json()["code"] == "not_found"


def test_async_ingest_returns_202_and_job_can_be_polled(client: TestClient, tmp_path: Path):
    response = upload(client, make_pdf(tmp_path / "later.pdf", FINANCE), wait="false")
    assert response.status_code == 202
    job = response.json()
    assert job["status"] in ("queued", "running", "succeeded")
    deadline = time.time() + 30
    while time.time() < deadline:
        state = client.get(f"/api/v1/jobs/{job['job_id']}").json()
        if state["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.1)
    assert state["status"] == "succeeded" and state["result"]["text_chunks"] >= 1
    assert client.get("/api/v1/jobs/unknown").status_code == 404


def test_upload_validation_and_hardening(client: TestClient, tmp_path: Path):
    exe = tmp_path / "program.exe"
    exe.write_bytes(b"MZ")
    rejected = upload(client, exe)
    assert rejected.status_code == 415 and rejected.json()["code"] == "unsupported_type"

    pdf = make_pdf(tmp_path / "ok.pdf", FINANCE)
    assert upload(client, pdf, collection="nope").status_code == 404
    assert upload(client, pdf, image_language="klingon").status_code == 400
    assert upload(client, pdf, kinds="text,video").status_code == 400
    assert (
        upload(client, pdf, collection="beta", name="x.docx").status_code == 415
    )  # beta takes pdf/text/csv only

    # a hostile file name is reduced to its base name and stays inside the data directory
    hostile = upload(client, pdf, name="../../../../tmp/evil.pdf")
    assert hostile.status_code == 200
    stored = Path(hostile.json()["source"])
    data_dir = client.app.state.container.settings.data_dir
    assert stored.parent == data_dir / "alpha" and stored.name == "evil.pdf"  # per-collection directory
    assert not Path("/tmp/evil.pdf").exists()
    assert upload(client, pdf, name=".hidden.pdf").status_code == 400


def test_oversized_upload_is_rejected_while_streaming(
    make_settings, rag_toml: Path, tmp_path: Path, run_id: str
):
    settings = make_settings(rag_config=rag_toml, max_upload_mb=1)
    with TestClient(create_app(settings)) as client:
        big = tmp_path / "big.txt"
        big.write_bytes(b"x" * (1024 * 1024 + 10))
        response = upload(client, big)
        assert response.status_code == 413 and response.json()["code"] == "payload_too_large"
        data_dir = settings.data_dir
        assert not list(data_dir.glob("*")) or all(
            p.name.startswith(".") is False for p in data_dir.glob("*")
        ), "no partial file left behind"
        assert not any(p.name.startswith(".upload-") for p in data_dir.iterdir()), "temp file cleaned up"


def test_rate_limit_returns_429_with_retry_after(make_settings, rag_toml: Path, run_id: str):
    settings = make_settings(rag_config=rag_toml, rate_limit_per_minute=3, rate_limit_backend="memory")
    with TestClient(create_app(settings)) as client:
        codes = [client.post("/api/v1/retrieve", json={"query": "anything"}).status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
        limited = client.post("/api/v1/retrieve", json={"query": "anything"})
        assert int(limited.headers["retry-after"]) >= 1 and limited.json()["code"] == "rate_limited"
        assert client.get("/health").status_code == 200  # ops endpoints are not limited


def test_metrics_endpoint(client: TestClient):
    client.get("/health")
    text = client.get("/metrics").text
    assert 'rag_http_requests_total{method="GET",route="/health",status="200"}' in text
    assert "rag_http_request_seconds_bucket" in text and "rag_upstream_seconds" in text
