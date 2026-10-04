"""Every built-in component - and the fake plug-in used across the suite - passes the conformance checks that
plug-in authors are given (``turinton_rag.testing``). And the checks themselves fail on implementations that
break the contract: a check that cannot fail proves nothing."""

from __future__ import annotations

import csv
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from src.core.bootstrap import build_registries
from src.core.config import Settings
from src.core.specs import EmbeddingModelSpec, RerankerSpec
from src.models.adapters import CachingEmbedder
from src.models.rerankers import ApiReranker, CrossEncoderReranker
from src.retrieval.rerank import HeuristicReranker, IdentityReranker
from src.runtime.cache import MemoryCache, TieredCache
from turinton_rag.testing import (
    ConformanceError,
    check_cache,
    check_chat_model,
    check_chunker,
    check_conversation_store,
    check_embedder,
    check_parser,
    check_query_expander,
    check_reranker,
)

from tests.fake_plugin import HashEmbedder, OverlapReranker, ScriptedChat
from tests.unit.tiny_models import save_tiny_bert

SETTINGS = Settings(_env_file=None, plugins=["tests.fake_plugin"])
REGISTRIES = build_registries(SETTINGS)


# --- embedders ----------------------------------------------------------------------------------------------------
async def test_embedders_conform(tmp_path):
    await check_embedder(HashEmbedder("hash", 32))
    await check_embedder(CachingEmbedder(HashEmbedder("hash", 32), max_entries=100, shared=MemoryCache(100)))
    pytest.importorskip("transformers")
    from src.models.local import HuggingFaceEmbedder

    path = save_tiny_bert(tmp_path / "emb", labels=None, hidden=16)
    spec = EmbeddingModelSpec(
        provider="huggingface", model=str(path), dimensions=16, options={"device": "cpu"}
    )
    await check_embedder(
        HuggingFaceEmbedder("tiny", spec, SETTINGS),
        samples=("revenue growth", "paris france", "revenue growth"),
    )


class _Embedder(HashEmbedder):
    pass


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        (
            type("WrongSize", (_Embedder,), {"_vector": lambda self, t: [1.0] * (self.dimensions + 1)}),
            "dimensions says",
        ),
        (type("AllZero", (_Embedder,), {"_vector": lambda self, t: [0.0] * self.dimensions}), "zero vector"),
        (
            type("NotFinite", (_Embedder,), {"_vector": lambda self, t: [float("nan")] * self.dimensions}),
            "NaN",
        ),
    ],
)
async def test_the_embedder_check_rejects_broken_embedders(broken, message):
    with pytest.raises(ConformanceError, match=message):
        await check_embedder(broken("broken", 8))


async def test_the_embedder_check_rejects_vectors_that_depend_on_the_batch():
    class BatchDependent(HashEmbedder):
        async def embed_documents(self, texts):
            vectors = await super().embed_documents(texts)
            return [[v + len(texts) for v in vec] for vec in vectors]

    with pytest.raises(ConformanceError, match="depends on what else is in the batch"):
        await check_embedder(BatchDependent("b", 8))


# --- rerankers -----------------------------------------------------------------------------------------------------
async def test_rerankers_conform(tmp_path):
    await check_reranker(IdentityReranker())
    await check_reranker(HeuristicReranker())
    await check_reranker(OverlapReranker())

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        n = len(json.loads(request.content)["documents"])
        return httpx.Response(
            200, json={"results": [{"index": i, "relevance_score": 1.0 / (i + 1)} for i in range(n)]}
        )

    api = ApiReranker(
        "cohere",
        "r",
        RerankerSpec(provider="cohere", model="m"),
        SETTINGS,
        SecretStr("k"),
        transport=httpx.MockTransport(handler),
    )
    await check_reranker(api)

    pytest.importorskip("transformers")
    path = save_tiny_bert(tmp_path / "ce", labels=1)
    await check_reranker(
        CrossEncoderReranker(
            "ce", RerankerSpec(provider="cross-encoder", model=str(path), options={"device": "cpu"}), SETTINGS
        )
    )


async def test_the_reranker_check_rejects_broken_rerankers():
    class NoScores(IdentityReranker):
        async def rerank(self, documents, query):
            return list(documents)

    class Copies(IdentityReranker):
        async def rerank(self, documents, query):
            import copy

            return [self._scored(copy.copy(d)) for d in documents]

        @staticmethod
        def _scored(doc):
            doc.rerank_score = 1.0
            return doc

    class Drops(IdentityReranker):
        async def rerank(self, documents, query):
            return (await super().rerank(documents, query))[:-1]

    class Mutates(IdentityReranker):
        async def rerank(self, documents, query):
            for d in documents:
                d.content = d.content.upper()
                d.rerank_score = 1.0
            return list(documents)

    class NotFinite(IdentityReranker):
        async def rerank(self, documents, query):
            for d in documents:
                d.rerank_score = float("inf")
            return list(documents)

    for broken, message in [
        (NoScores, "finite rerank_score"),
        (Copies, "not copies"),
        (Drops, "returned 2 documents for 3"),
        (Mutates, "must not alter"),
        (NotFinite, "finite rerank_score"),
    ]:
        with pytest.raises(ConformanceError, match=message):
            await check_reranker(broken())


# --- chat models and query expanders -----------------------------------------------------------------------------------
async def test_chat_model_and_expanders_conform():
    chat = ScriptedChat("fast")
    await check_chat_model(chat)
    cache = MemoryCache(100)
    for name in ("identity", "llm"):
        await check_query_expander(REGISTRIES.query_expanders.create(name, SETTINGS, chat, cache))


async def test_the_expander_check_rejects_expanders_that_lose_the_original_query():
    class Rewrites:
        async def expand(self, query):
            return [f"{query} explained", query]

    with pytest.raises(ConformanceError, match="first variant"):
        await check_query_expander(Rewrites())


# --- chunkers and parsers --------------------------------------------------------------------------------------------------
def test_the_chunker_conforms():
    from src.core.specs import ChunkerSpec

    check_chunker(
        REGISTRIES.chunkers.create(
            "recursive", ChunkerSpec(chunk_size=200, chunk_overlap=20, min_chunk_size=10)
        )
    )


def test_the_chunker_check_rejects_a_chunker_with_bad_numbering():
    from src.ports.parsing import ChunkDraft

    class Misnumbered:
        def split(self, text):
            return [] if not text else [ChunkDraft(text, 1, 1)]

    with pytest.raises(ConformanceError, match="0..n-1"):
        check_chunker(Misnumbered())


def test_the_built_in_parsers_conform(tmp_path: Path):
    from tests.helpers import make_docx, make_pdf

    text = "Quarterly revenue grew twelve percent."
    samples = {
        "text": [("note.txt", text + "\n\nMargins improved."), ("note.md", f"# Title\n\n{text}\n")],
        "html": [("page.html", f"<html><body><p>{text}</p></body></html>")],
    }
    for name, files in samples.items():
        for filename, content in files:
            path = tmp_path / filename
            path.write_text(content)
            check_parser(REGISTRIES.parsers.create(name, SETTINGS), path)
    table = tmp_path / "t.csv"
    with table.open("w", newline="") as handle:
        csv.writer(handle).writerows([["region", "q1"], ["north", "10"], ["south", "8"]])
    check_parser(REGISTRIES.parsers.create("csv", SETTINGS), table)
    check_parser(REGISTRIES.parsers.create("pdf", SETTINGS), make_pdf(tmp_path / "a.pdf", [text]))
    check_parser(
        REGISTRIES.parsers.create("docx", SETTINGS), make_docx(tmp_path / "a.docx", {"Results": text})
    )


def test_the_parser_check_rejects_a_parser_that_misreports_itself(tmp_path: Path):
    class Sloppy:
        name = "sloppy"
        extensions = frozenset({"txt"})  # missing the dot

        def iter_units(self, path):
            return iter(())

    sample = tmp_path / "a.txt"
    sample.write_text("x")
    with pytest.raises(ConformanceError, match="dot-prefixed"):
        check_parser(Sloppy(), sample)  # type: ignore[arg-type]


# --- caches and conversation stores ---------------------------------------------------------------------------------------
async def test_caches_conform():
    await check_cache(MemoryCache(100))
    await check_cache(TieredCache(MemoryCache(100), MemoryCache(100)))


@pytest.mark.integration
@pytest.mark.needs_redis
async def test_the_redis_cache_and_store_conform():
    from src.chat.stores import RedisConversationStore
    from src.runtime.cache import RedisCache

    from tests.conftest import REDIS_URL

    cache = RedisCache(REDIS_URL, namespace="conformance")
    store = RedisConversationStore(REDIS_URL, namespace="conformance", ttl=60)
    try:
        await check_cache(cache)
        await check_conversation_store(store)
    finally:
        await cache.close()
        await store.close()


async def test_memory_conversation_store_conforms():
    store = REGISTRIES.conversation_stores.create("memory", SETTINGS)
    await check_conversation_store(store)


async def test_the_cache_check_rejects_a_cache_that_never_expires_or_forgets_deletes():
    class Immortal(MemoryCache):
        async def set(self, key, value, ttl):
            await super().set(key, value, 10_000)

    with pytest.raises(ConformanceError, match="expire"):
        await check_cache(Immortal(10))
