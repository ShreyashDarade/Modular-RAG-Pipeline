from __future__ import annotations

import asyncio
import importlib

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import SecretStr
from src.core.config import Settings
from src.core.errors import ConfigError, ModelError, ProviderUnavailableError
from src.core.registry import Registries
from src.core.specs import ChatModelSpec, EmbeddingModelSpec
from src.core.types import ChatMessage
from src.models.adapters import CachingEmbedder, LangChainChatModel, LangChainEmbedder
from src.models.providers import register_builtin_providers
from src.runtime.cache import MemoryCache
from src.runtime.concurrency import Bulkhead

KEYS = dict(
    openai_api_key=SecretStr("sk-x"),
    anthropic_api_key=SecretStr("a"),
    google_api_key=SecretStr("g"),
    azure_openai_api_key=SecretStr("z"),
    azure_openai_endpoint="https://x.openai.azure.com",
    azure_openai_api_version="2025-04-01-preview",
)


def registries() -> Registries:
    r = Registries()
    register_builtin_providers(r)
    return r


# --- providers ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("provider", "model", "klass"),
    [
        ("openai", "gpt-4o-mini", "ChatOpenAI"),
        ("azure_openai", "deploy", "AzureChatOpenAI"),
        ("anthropic", "claude-sonnet-5-5", "ChatAnthropic"),
        ("google", "gemini-2.5-flash", "ChatGoogleGenerativeAI"),
        ("ollama", "llama3", "ChatOllama"),
    ],
)
def test_every_chat_provider_builds_from_a_spec(provider, model, klass):
    chat = registries().chat_providers.create(
        provider,
        "m",
        ChatModelSpec(provider=provider, model=model, temperature=0.2, max_tokens=64),
        Settings(_env_file=None, **KEYS),
    )
    assert type(chat._chat).__name__ == klass and chat.model_id == "m"  # noqa: SLF001


@pytest.mark.parametrize(
    ("provider", "model", "dims", "expected"),
    [
        ("openai", "text-embedding-3-small", None, 1536),
        ("openai", "text-embedding-3-large", None, 3072),
        ("openai", "text-embedding-3-large", 1024, 1024),
        ("azure_openai", "dep", 1536, 1536),
        ("google", "gemini-embedding-001", 768, 768),
        ("ollama", "nomic-embed-text", 768, 768),
    ],
)
def test_embedding_dimensions_are_known_up_front(provider, model, dims, expected):
    e = registries().embedding_providers.create(
        provider,
        "e",
        EmbeddingModelSpec(provider=provider, model=model, dimensions=dims),
        Settings(_env_file=None, **KEYS),
    )
    assert e.dimensions == expected


def test_unknown_dimensions_must_be_declared_never_probed():
    with pytest.raises(ConfigError, match="set `dimensions` explicitly"):
        registries().embedding_providers.create(
            "ollama",
            "e",
            EmbeddingModelSpec(provider="ollama", model="mystery"),
            Settings(_env_file=None, **KEYS),
        )


def test_missing_credentials_fail_at_construction():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY is required"):
        registries().chat_providers.create(
            "openai", "m", ChatModelSpec(provider="openai", model="x"), Settings(_env_file=None)
        )
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        registries().chat_providers.create(
            "anthropic", "m", ChatModelSpec(provider="anthropic", model="x"), Settings(_env_file=None)
        )


def test_a_missing_optional_package_names_the_extra_to_install(monkeypatch):
    real = importlib.import_module

    def fake_import(name, *a, **k):
        if name == "langchain_anthropic":
            raise ImportError("not installed")
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ProviderUnavailableError, match=r"turinton-rag\[anthropic\]"):
        registries().chat_providers.create(
            "anthropic", "m", ChatModelSpec(provider="anthropic", model="x"), Settings(_env_file=None, **KEYS)
        )


def test_anthropic_has_no_embedding_provider_so_asking_for_one_is_an_error():
    from src.core.errors import UnknownComponentError

    with pytest.raises(UnknownComponentError, match="Registered:"):
        registries().embedding_providers.create(
            "anthropic",
            "e",
            EmbeddingModelSpec(provider="anthropic", model="x", dimensions=8),
            Settings(_env_file=None, **KEYS),
        )


# --- chat adapter ---------------------------------------------------------------------------
async def test_chat_adapter_maps_roles_and_extracts_text():
    chat = LangChainChatModel(
        "m", FakeListChatModel(responses=["hello there"]), service="t", bulkhead=Bulkhead(2)
    )
    assert (
        await chat.complete(
            [ChatMessage("system", "s"), ChatMessage("user", "u"), ChatMessage("assistant", "a")]
        )
        == "hello there"
    )
    chunks = [
        c
        async for c in LangChainChatModel(
            "m", FakeListChatModel(responses=["abc"]), service="t", bulkhead=Bulkhead(2)
        ).stream([ChatMessage("user", "u")])
    ]
    assert "".join(chunks) == "abc"


class Exploding:
    async def ainvoke(self, *_):
        raise RuntimeError("Incorrect API key provided: sk-abc...wxyz")

    async def astream(self, *_):
        raise RuntimeError("stream broke")
        yield  # pragma: no cover


async def test_model_failures_become_model_errors_whose_public_text_hides_the_detail():
    chat = LangChainChatModel("m", Exploding(), service="t", bulkhead=Bulkhead(2))
    with pytest.raises(ModelError) as info:
        await chat.complete([ChatMessage("user", "u")])
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "sk-abc" in str(info.value), "full detail is available to logs"
    assert "sk-abc" not in info.value.public_message and info.value.status_code == 502
    with pytest.raises(ModelError):
        [c async for c in chat.stream([ChatMessage("user", "u")])]


# --- embedder adapter -----------------------------------------------------------------------
class RecordingEmbeddings:
    def __init__(self, dims=4, delay=0.01):
        self.dims, self.delay, self.calls, self.active, self.peak = dims, delay, [], 0, 0

    async def aembed_documents(self, texts):
        self.calls.append(list(texts))
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        return [[float(len(t))] * self.dims for t in texts]


def embedder(raw, *, batch=3, limit=8, dims=4):
    return LangChainEmbedder(
        "e", raw, dimensions=dims, batch_size=batch, service="t", bulkhead=Bulkhead(limit)
    )


async def test_embedder_splits_into_batches_runs_them_concurrently_and_preserves_order():
    raw = RecordingEmbeddings()
    texts = ["x" * n for n in range(1, 11)]
    vectors = await embedder(raw).embed_documents(texts)
    assert [v[0] for v in vectors] == [float(len(t)) for t in texts], "order preserved"
    assert [len(c) for c in raw.calls] == [3, 3, 3, 1] and raw.peak > 1, (
        "batches overlap instead of running serially"
    )


async def test_embedder_respects_the_concurrency_limit():
    raw = RecordingEmbeddings()
    await embedder(raw, batch=1, limit=2).embed_documents(["a"] * 12)
    assert raw.peak == 2


async def test_embedder_rejects_vectors_of_the_wrong_size():
    with pytest.raises(ModelError, match="expected 2 vectors of 8 dimensions"):
        await embedder(RecordingEmbeddings(dims=4), dims=8).embed_documents(["a", "b"])


async def test_embedder_wraps_provider_errors():
    class Down:
        async def aembed_documents(self, texts):
            raise ConnectionError("connection refused")

    with pytest.raises(ModelError):
        await embedder(Down()).embed_queries(["a"])


# --- caching decorator ----------------------------------------------------------------------
async def test_embedding_cache_never_confuses_texts_with_a_shared_prefix():
    """Regression: the old cache keyed on the first 500 characters, so two chunks that began the
    same way silently received each other's vectors."""
    raw = RecordingEmbeddings()
    cached = CachingEmbedder(embedder(raw), max_entries=100)
    prefix = "p" * 600
    a, b = await cached.embed_documents([prefix + " ends with A", prefix + " ends with a much longer B"])
    assert a != b
    assert (await cached.embed_documents([prefix + " ends with A"]))[0] == a


async def test_embedding_cache_embeds_each_distinct_text_once_and_serves_repeats():
    raw = RecordingEmbeddings()
    cached = CachingEmbedder(embedder(raw, batch=10), max_entries=100)
    first = await cached.embed_documents(["alpha", "beta", "alpha", "alpha"])
    assert sum(len(c) for c in raw.calls) == 2 and first[0] == first[2] == first[3]
    await cached.embed_documents(["alpha", "beta"])
    assert sum(len(c) for c in raw.calls) == 2, "second call is entirely from cache"


async def test_embedding_cache_is_bounded():
    raw = RecordingEmbeddings()
    cached = CachingEmbedder(embedder(raw, batch=10), max_entries=2)
    await cached.embed_documents(["a", "b", "c"])
    assert len(cached._lru) == 2  # noqa: SLF001
    zero = CachingEmbedder(embedder(RecordingEmbeddings(), batch=10), max_entries=0)
    await zero.embed_documents(["a"])
    assert len(zero._lru) == 0  # noqa: SLF001


async def test_queries_use_the_shared_cache_documents_do_not():
    raw, shared = RecordingEmbeddings(), MemoryCache(100)
    one = CachingEmbedder(embedder(raw, batch=10), max_entries=10, shared=shared)
    two = CachingEmbedder(embedder(raw, batch=10), max_entries=10, shared=shared)  # another replica
    q = (await one.embed_queries(["what is revenue"]))[0]
    calls_after_first = len(raw.calls)
    assert (await two.embed_queries(["what is revenue"]))[0] == pytest.approx(q), (
        "replica two reuses replica one's work"
    )
    assert len(raw.calls) == calls_after_first
    await one.embed_documents(["a document chunk"])
    await two.embed_documents(["a document chunk"])
    assert len(raw.calls) == calls_after_first + 2, (
        "documents are not shared (they are written once, not asked repeatedly)"
    )
