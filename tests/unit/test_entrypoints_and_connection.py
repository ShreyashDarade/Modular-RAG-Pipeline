"""Connection settings, structured logging and start-up failure handling."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient
from src.api.server import create_app
from src.core.config import Settings
from src.core.errors import ConfigError, UpstreamError
from src.core.logger import _JsonFormatter, request_id_var
from src.indexing import elastic


@pytest.fixture
def captured(monkeypatch):
    seen: dict = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(elastic, "AsyncElasticsearch", FakeClient)
    return seen


def connect(**settings) -> None:
    elastic.ElasticConnection(Settings(_env_file=None, **settings))


def test_elastic_cloud_with_an_api_key(captured):
    connect(es_cloud_id="deploy:abc", es_api_key="key123")
    assert (
        captured["cloud_id"] == "deploy:abc" and captured["api_key"] == "key123" and "hosts" not in captured
    )
    assert (
        captured["verify_certs"] is True
        and captured["http_compress"] is True
        and captured["retry_on_timeout"] is True
    )
    assert captured["connections_per_node"] == 32 and captured["max_retries"] == 3


def test_self_hosted_nodes_with_basic_auth_and_a_custom_ca(captured):
    connect(
        es_host="https://es1:9200, https://es2:9200",
        es_username="elastic",
        es_password="pw",
        es_ca_certs="/ca.pem",
        es_verify_certs=False,
    )
    assert captured["hosts"] == ["https://es1:9200", "https://es2:9200"]
    assert (
        captured["basic_auth"] == ("elastic", "pw")
        and captured["ca_certs"] == "/ca.pem"
        and captured["verify_certs"] is False
    )
    assert "api_key" not in captured and "cloud_id" not in captured


def test_an_api_key_wins_over_basic_auth(captured):
    connect(es_host="http://es:9200", es_api_key="key", es_username="u", es_password="p")
    assert captured["api_key"] == "key" and "basic_auth" not in captured


def test_pool_and_retry_settings_reach_the_client(captured):
    connect(es_host="http://es:9200", es_connections_per_node=7, es_max_retries=1, es_request_timeout=9.0)
    assert (captured["connections_per_node"], captured["max_retries"], captured["request_timeout"]) == (
        7,
        1,
        9.0,
    )


def test_unconfigured_elasticsearch_is_an_explicit_error(captured):
    with pytest.raises(ConfigError, match="ES_CLOUD_ID.*ES_HOST"):
        connect()


def test_json_log_lines_carry_the_request_id_and_extra_fields():
    token = request_id_var.set("req-42")
    try:
        record = logging.LogRecord(
            "ai-rag-info", logging.WARNING, __file__, 1, "model call failed", None, None
        )
        record.request_id = request_id_var.get()
        record.model = "gpt-x"
        line = json.loads(_JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert line["level"] == "WARNING" and line["message"] == "model call failed"
    assert line["request_id"] == "req-42" and line["model"] == "gpt-x" and line["ts"]


def test_a_failed_startup_surfaces_the_error_and_does_not_hang(tmp_path):
    """Elasticsearch unreachable: start-up fails fast with a typed error (and the container is closed)."""
    settings = Settings(
        _env_file=None,
        es_host="http://127.0.0.1:1",
        es_max_retries=0,
        es_request_timeout=2.0,
        plugins=["tests.fake_plugin"],
        data_dir=tmp_path / "data",
        ocr_enabled=False,
        ingest_embedded_worker=False,
        rag_config=None,
        openai_api_key="sk-unused",
    )
    with pytest.raises(UpstreamError), TestClient(create_app(settings)):
        pass  # pragma: no cover
