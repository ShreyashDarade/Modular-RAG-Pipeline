from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from src.core.config import Settings
from src.core.container import Container
from src.core.specs import RagConfig

ES_URL = os.environ.get("RAG_TEST_ES_URL", "http://localhost:9200")
REDIS_URL = os.environ.get("RAG_TEST_REDIS_URL", "redis://localhost:6390/5")


def _reachable_es() -> bool:
    try:
        return httpx.get(ES_URL, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def _reachable_redis() -> bool:
    import redis

    try:
        return bool(redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2).ping())
    except redis.RedisError:
        return False


def pytest_collection_modifyitems(config, items):
    es, rd = _reachable_es(), _reachable_redis()
    for item in items:
        if "integration" in item.keywords:
            if not es:
                item.add_marker(pytest.mark.skip(reason=f"Elasticsearch not reachable at {ES_URL}"))
            elif "needs_redis" in item.keywords and not rd:
                item.add_marker(pytest.mark.skip(reason=f"Redis not reachable at {REDIS_URL}"))


@pytest.fixture
def run_id() -> str:
    return uuid.uuid4().hex[:8]


@pytest.fixture
def make_settings(tmp_path: Path, run_id: str):
    def build(**overrides) -> Settings:
        values = dict(
            es_host=ES_URL,
            es_number_of_replicas=0,
            es_refresh_interval="1s",
            es_index_registry=f"t{run_id}-registry",
            data_dir=tmp_path / "data",
            plugins=["tests.fake_plugin"],
            redis_url=REDIS_URL,
            rate_limit_per_minute=0,
            ocr_enabled=False,
            watch_data_dir=False,
            log_level="WARNING",
        )
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return build


@pytest.fixture
def rag_config(run_id: str) -> RagConfig:
    return RagConfig.model_validate(
        {
            "default_chat_model": "fast",
            "default_collection": "alpha",
            "utility_model": "fast",
            "chat_models": {
                "fast": {"provider": "fake", "model": "fast"},
                "smart": {"provider": "fake", "model": "smart"},
            },
            "embedding_models": {
                "hash64": {"provider": "fake", "model": "hash", "dimensions": 64},
                "hash32": {"provider": "fake", "model": "hash", "dimensions": 32},
            },
            "collections": {
                "alpha": {
                    "embedding_model": "hash64",
                    "description": "Alpha corpus",
                    "index_prefix": f"t{run_id}-alpha",
                    "chunker": {"chunk_size": 400, "chunk_overlap": 50, "min_chunk_size": 20},
                },
                "beta": {
                    "embedding_model": "hash32",
                    "index_prefix": f"t{run_id}-beta",
                    "kinds": ["text", "table"],
                    "parsers": ["pdf", "text", "csv"],
                    "chunker": {"chunk_size": 400, "chunk_overlap": 50, "min_chunk_size": 20},
                },
            },
        }
    )


@pytest.fixture
async def container(make_settings, rag_config: RagConfig, run_id: str) -> AsyncIterator[Container]:
    """A fully wired container (ingestion included) against live Elasticsearch, with fake models."""
    settings = make_settings()
    built = await Container.build(settings, role="api", with_ingestion=True, config=rag_config)
    await built.start()
    try:
        yield built
    finally:
        try:
            names = list(
                await built.elastic.client.indices.get(index=f"t{run_id}-*", ignore_unavailable=True)
            )
            if names:  # ES refuses wildcard deletes by default, so delete by resolved name
                await built.elastic.client.indices.delete(index=names, ignore_unavailable=True)
        finally:
            await built.close()


@pytest.fixture
def rag_toml(tmp_path: Path, run_id: str) -> Path:
    """The same two-collection setup as ``rag_config``, but as the TOML file operators write."""
    path = tmp_path / "rag.toml"
    path.write_text(
        f"""
default_chat_model = "fast"
default_collection = "alpha"
utility_model = "fast"

[chat_models.fast]
provider = "fake"
model = "fast"

[chat_models.smart]
provider = "fake"
model = "smart"

[embedding_models.hash64]
provider = "fake"
model = "hash"
dimensions = 64

[embedding_models.hash32]
provider = "fake"
model = "hash"
dimensions = 32

[collections.alpha]
embedding_model = "hash64"
description = "Alpha corpus"
index_prefix = "t{run_id}-alpha"
[collections.alpha.chunker]
chunk_size = 400
chunk_overlap = 50
min_chunk_size = 20

[collections.beta]
embedding_model = "hash32"
index_prefix = "t{run_id}-beta"
kinds = ["text", "table"]
parsers = ["pdf", "text", "csv"]
[collections.beta.chunker]
chunk_size = 400
chunk_overlap = 50
min_chunk_size = 20
"""
    )
    return path
