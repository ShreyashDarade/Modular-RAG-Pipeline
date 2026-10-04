from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from src.core.bootstrap import build_registries
from src.core.config import Settings, load_rag_config
from src.core.errors import ConfigError, UnknownComponentError
from src.core.registry import Registries, Registry
from src.core.specs import ChunkerSpec, CollectionSpec, RagConfig

VALID = {
    "default_chat_model": "c",
    "default_collection": "a",
    "chat_models": {"c": {"provider": "openai", "model": "m"}},
    "embedding_models": {"e": {"provider": "openai", "model": "text-embedding-3-small"}},
    "collections": {"a": {"embedding_model": "e"}},
}


def config(**changes) -> dict:
    return {**VALID, **changes}


# --- Settings -------------------------------------------------------------------------------
def test_legacy_redis_cache_flag_maps_to_tiered_unless_overridden():
    assert Settings(_env_file=None, use_redis_cache=True).cache_backend == "tiered"
    assert Settings(_env_file=None, use_redis_cache=True, cache_backend="memory").cache_backend == "memory"
    assert Settings(_env_file=None).cache_backend == "memory"


def test_settings_validate_ranges():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hybrid_alpha=1.5)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, es_connections_per_node=0)


def test_secrets_are_not_shown_in_repr():
    s = Settings(_env_file=None, es_api_key="super-secret", openai_api_key="sk-secret")
    assert "super-secret" not in repr(s) and "sk-secret" not in repr(s)


def test_default_rag_config_is_synthesised_from_classic_env_vars():
    s = Settings(
        _env_file=None,
        openai_model="gpt-x",
        openai_embedding_model="text-embedding-3-large",
        chunk_size=500,
        chunk_overlap=50,
        es_index_text="my-text",
        es_index_tables="my-tables",
        es_index_images="my-images",
    )
    cfg = load_rag_config(s)
    assert cfg.chat_models["default"].model == "gpt-x"
    assert cfg.embedding_models["default"].model == "text-embedding-3-large"
    collection = cfg.collection("default")
    assert collection.chunker.chunk_size == 500
    assert collection.index_names() == {"text": "my-text", "table": "my-tables", "image": "my-images"}
    assert collection.subdir == "", "legacy layout: files directly in DATA_DIR"
    assert cfg.query_expander == "llm" and cfg.reranker == "heuristic"


def test_disabling_expansion_and_rerank_selects_the_identity_strategies():
    cfg = load_rag_config(Settings(_env_file=None, query_expansion_enabled=False, rerank_enabled=False))
    assert cfg.query_expander == "identity" and cfg.reranker == "identity"


# --- RagConfig ------------------------------------------------------------------------------
def test_valid_config_and_naming():
    cfg = RagConfig.model_validate(VALID)
    assert cfg.collection("a").name == "a"
    assert cfg.collection("a").index_names() == {"text": "a-text", "table": "a-tables", "image": "a-images"}
    assert cfg.utility_model == "c", "utility model defaults to the default chat model"
    assert cfg.collection("a").subdir == "a"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"default_collection": "zzz"}, "default_collection 'zzz' is not defined"),
        ({"default_chat_model": "zzz"}, "default_chat_model 'zzz' is not defined"),
        ({"collections": {"a": {"embedding_model": "nope"}}}, "unknown embedding model 'nope'"),
        (
            {"collections": {"Bad Name": {"embedding_model": "e"}}, "default_collection": "Bad Name"},
            "invalid name",
        ),
        ({"chat_models": {"c": {"provider": "openai", "model": "m", "surprise": 1}}}, "surprise"),
    ],
)
def test_invalid_configs_are_rejected_with_a_useful_message(changes, message):
    with pytest.raises(ValidationError, match=message):
        RagConfig.model_validate(config(**changes))


def test_two_collections_cannot_share_an_index():
    shared = {
        "a": {"embedding_model": "e", "index_prefix": "x"},
        "b": {"embedding_model": "e", "index_prefix": "x"},
    }
    with pytest.raises(ValidationError, match="used by both"):
        RagConfig.model_validate(config(collections=shared))


def test_chunk_overlap_must_be_smaller_than_size():
    with pytest.raises(ValidationError, match="chunk_overlap"):
        ChunkerSpec(chunk_size=100, chunk_overlap=100)


def test_collection_must_index_something():
    with pytest.raises(ValidationError):
        CollectionSpec(embedding_model="e", kinds=())


def test_from_toml_errors_name_the_file(tmp_path: Path):
    missing = tmp_path / "missing.toml"
    with pytest.raises(ConfigError, match="missing.toml"):
        RagConfig.from_toml(missing)
    broken = tmp_path / "broken.toml"
    broken.write_text("this is = = not toml")
    with pytest.raises(ConfigError, match="broken.toml"):
        RagConfig.from_toml(broken)
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        'default_chat_model = "x"\ndefault_collection = "y"\n[chat_models]\n[embedding_models]\n[collections]\n'
    )
    with pytest.raises(ConfigError, match="invalid.toml"):
        RagConfig.from_toml(invalid)


def test_example_config_in_the_repo_is_valid():
    path = Path(__file__).resolve().parents[2] / "config" / "rag.example.toml"
    cfg = RagConfig.from_toml(path)
    assert len(cfg.collections) >= 2 and len(cfg.chat_models) >= 2


# --- registries and plug-ins ----------------------------------------------------------------
def test_registry_creates_by_name_and_refuses_duplicates_and_unknowns():
    registry: Registry[str] = Registry("widget")
    registry.register("a", lambda x: f"a:{x}")
    assert registry.create("a", 1) == "a:1" and "a" in registry and registry.names() == ["a"]
    with pytest.raises(ConfigError, match="already registered"):
        registry.register("a", lambda x: x)
    registry.register("a", lambda x: "replaced", replace=True)
    with pytest.raises(UnknownComponentError, match=r"Unknown widget 'zzz'. Registered: a"):
        registry.create("zzz")


def test_builtin_registries_cover_every_extension_point():
    registries = build_registries(Settings(_env_file=None))
    assert {"openai", "azure_openai", "anthropic", "google", "ollama"} <= set(
        registries.chat_providers.names()
    )
    assert {"openai", "azure_openai", "google", "ollama"} <= set(registries.embedding_providers.names())
    assert {"pdf", "image", "text", "html", "csv", "xlsx", "docx"} <= set(registries.parsers.names())
    assert registries.chunkers.names() == ["recursive"] and registries.ocr_engines.names() == ["easyocr"]
    assert set(registries.query_expanders.names()) == {"llm", "identity"}
    assert set(registries.rerankers.names()) == {"heuristic", "identity"}
    assert set(registries.caches.names()) == {"memory", "redis", "tiered"}
    assert set(registries.job_backends.names()) == {"inprocess", "redis"}
    assert set(registries.conversation_stores.names()) == {"memory", "redis"}
    assert set(registries.rate_limiters.names()) == {"memory", "redis"}


def test_plugin_adds_a_component_without_touching_core():
    registries = build_registries(Settings(_env_file=None, plugins=["tests.fake_plugin"]))
    assert "fake" in registries.chat_providers and "fake" in registries.embedding_providers


def test_broken_plugins_fail_loudly_at_startup():
    with pytest.raises(ConfigError, match="could not be imported"):
        build_registries(Settings(_env_file=None, plugins=["no.such.module"]))
    with pytest.raises(ConfigError, match=r"no register\(registries\)"):
        build_registries(Settings(_env_file=None, plugins=["json"]))


def test_registries_are_independent_instances():
    a, b = Registries(), Registries()
    a.parsers.register("x", lambda s: s)
    assert "x" not in b.parsers, "no global state: each container gets its own registries"


def test_refresh_interval_must_be_an_elasticsearch_time_value():
    with pytest.raises(ValueError, match="ES_REFRESH_INTERVAL"):
        Settings(_env_file=None, es_refresh_interval="soon")
    assert Settings(_env_file=None, es_refresh_interval="500ms").es_refresh_interval == "500ms"


def test_search_settle_window_follows_the_refresh_mode():
    assert (
        Settings(_env_file=None, ingest_refresh="each", es_refresh_interval="5s").search_settle_seconds == 0.0
    )
    assert (
        Settings(_env_file=None, ingest_refresh="interval", es_refresh_interval="5s").search_settle_seconds
        == 6.0
    )
    assert (
        Settings(_env_file=None, ingest_refresh="interval", es_refresh_interval="1m").search_settle_seconds
        == 61.0
    )


def test_interval_mode_refuses_a_disabled_refresh():
    with pytest.raises(ValueError, match="never refreshes"):
        Settings(_env_file=None, ingest_refresh="interval", es_refresh_interval="-1")
