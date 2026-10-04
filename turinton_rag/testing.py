"""Conformance checks for the extension points (``docs/adr/0007``).

Each ``check_*`` function states what *any* implementation of a port must do, and raises
:class:`ConformanceError` (an ``AssertionError``) saying which promise was broken. They are plain functions -
no pytest needed - so a plug-in author can call them from any test runner::

    async def test_my_embedder():
        await check_embedder(MyEmbedder(...))

The engine's own components pass the same checks in CI. A contract these functions do not express is a
convention, not a guarantee.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from pathlib import Path

from src.core.types import ChatMessage, RetrievedDocument
from src.ports.models import ChatModel, Embedder
from src.ports.parsing import Chunker, ParsedUnit, Parser
from src.ports.retrieval import QueryExpander, Reranker
from src.ports.runtime import Cache, ConversationStore

from turinton_rag._compat import experimental


class ConformanceError(AssertionError):
    """An implementation broke a promise its port makes."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ConformanceError(message)


def _close(a: Sequence[float], b: Sequence[float], tolerance: float = 1e-4) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b, strict=True))


@experimental
async def check_embedder(
    embedder: Embedder, samples: Sequence[str] = ("alpha beta", "gamma delta", "alpha beta")
) -> None:
    """Vectors have the declared size, are finite and non-zero, do not depend on batching, and equal texts get equal vectors."""
    _require(isinstance(embedder.model_id, str) and embedder.model_id, "model_id must be a non-empty string")
    _require(
        isinstance(embedder.dimensions, int) and embedder.dimensions > 0, "dimensions must be a positive int"
    )
    texts = list(samples)
    _require(await embedder.embed_documents([]) == [], "embed_documents([]) must return []")
    _require(await embedder.embed_queries([]) == [], "embed_queries([]) must return []")
    for role, embed in (("documents", embedder.embed_documents), ("queries", embedder.embed_queries)):
        vectors = await embed(texts)
        _require(
            len(vectors) == len(texts), f"embed_{role} returned {len(vectors)} vectors for {len(texts)} texts"
        )
        for vector in vectors:
            _require(
                len(vector) == embedder.dimensions,
                f"embed_{role}: a vector has {len(vector)} values, dimensions says {embedder.dimensions}",
            )
            _require(
                all(math.isfinite(x) for x in vector), f"embed_{role}: a vector contains NaN or infinity"
            )
            _require(
                any(x != 0.0 for x in vector), f"embed_{role}: a zero vector (cosine indices reject them)"
            )
        _require(
            _close(vectors[0], vectors[-1]) if texts[0] == texts[-1] else True,
            f"embed_{role}: equal texts got different vectors",
        )
        for text, vector in zip(texts, vectors, strict=True):
            alone = (await embed([text]))[0]
            _require(
                _close(alone, vector),
                f"embed_{role}: the vector for {text!r} depends on what else is in the batch",
            )


@experimental
async def check_chat_model(chat: ChatModel) -> None:
    """``complete`` returns text, ``stream`` yields text chunks, and a failure is an exception, never an empty answer."""
    _require(isinstance(chat.model_id, str) and chat.model_id, "model_id must be a non-empty string")
    messages = [ChatMessage("system", "Answer briefly."), ChatMessage("user", "Say hello.")]
    answer = await chat.complete(messages)
    _require(isinstance(answer, str), f"complete() must return str, got {type(answer).__name__}")
    chunks = [chunk async for chunk in chat.stream(messages)]
    _require(all(isinstance(c, str) for c in chunks), "stream() must yield str chunks")


def _candidates(texts: Sequence[str]) -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            content=text,
            metadata={"source": f"/s{i}.txt", "page": 1, "keywords": []},
            score=0.01 * (len(texts) - i),
            collection="c",
            kind="text",
            index="i",
        )
        for i, text in enumerate(texts)
    ]


@experimental
async def check_reranker(
    reranker: Reranker,
    query: str = "quarterly revenue growth",
    passages: Sequence[str] = (
        "Quarterly revenue grew twelve percent across all regions.",
        "The office plants need watering every week.",
        "Revenue was discussed at the board meeting.",
    ),
) -> None:
    """Every candidate comes back exactly once with a finite ``rerank_score``; nothing else about it changes."""
    await reranker.start()
    try:
        _require(await reranker.rerank([], query) == [], "rerank([]) must return []")
        docs = _candidates(passages)
        before = [(d.content, dict(d.metadata), d.score) for d in docs]
        out = await reranker.rerank(docs, query)
        _require(len(out) == len(docs), f"rerank returned {len(out)} documents for {len(docs)}")
        _require(
            {id(d) for d in out} == {id(d) for d in docs},
            "rerank must return the candidates it was given, not copies or others",
        )
        for doc in out:
            _require(
                doc.rerank_score is not None and math.isfinite(doc.rerank_score),
                "every candidate needs a finite rerank_score",
            )
        _require(
            before == [(d.content, dict(d.metadata), d.score) for d in docs],
            "rerank must not alter content, metadata or the first-stage score",
        )
        again = await reranker.rerank(_candidates(passages), query)
        _require(
            [round(d.final_score, 4) for d in again]
            == [round(d.final_score, 4) for d in sorted(out, key=lambda d: passages.index(d.content))],
            "scoring the same input twice gave different scores",
        )
    finally:
        await reranker.close()


@experimental
async def check_query_expander(expander: QueryExpander, query: str = "quarterly revenue growth") -> None:
    """Variants are non-empty strings and the original query comes first, unchanged."""
    variants = await expander.expand(query)
    _require(isinstance(variants, list) and variants, "expand() must return a non-empty list")
    _require(variants[0] == query, "the original query must be the first variant")
    _require(all(isinstance(v, str) and v.strip() for v in variants), "variants must be non-empty strings")


@experimental
def check_chunker(chunker: Chunker, text: str = ("Revenue grew twelve percent. " * 120)) -> None:
    """Chunks cover the text, are numbered 0..n-1 with a consistent total, and empty input yields no chunks."""
    _require(chunker.split("") == [], "split('') must return no chunks")
    chunks = chunker.split(text)
    _require(chunks, "a non-empty text must yield at least one chunk")
    _require([c.index for c in chunks] == list(range(len(chunks))), "chunk indices must be 0..n-1 in order")
    _require(
        all(c.total == len(chunks) for c in chunks), "every chunk's total must equal the number of chunks"
    )
    _require(all(c.content.strip() for c in chunks), "chunks must not be blank")
    joined = " ".join(c.content for c in chunks)
    _require("Revenue grew twelve percent" in joined, "chunks must carry the text's content")


@experimental
def check_parser(parser: Parser, sample: Path) -> None:
    """``name`` and ``extensions`` are well-formed; ``sample`` (a file the parser should accept) parses into numbered units with content."""
    _require(isinstance(parser.name, str) and parser.name, "name must be a non-empty string")
    _require(
        parser.extensions and all(e.startswith(".") and e == e.lower() for e in parser.extensions),
        "extensions must be lower-case, dot-prefixed",
    )
    _require(
        sample.suffix.lower() in parser.extensions,
        f"the sample {sample.name} is not one of the parser's extensions {sorted(parser.extensions)}",
    )
    units = list(parser.iter_units(sample))
    _require(units, "a valid sample must yield at least one unit")
    _require(all(isinstance(u, ParsedUnit) for u in units), "iter_units must yield ParsedUnit objects")
    _require(
        [u.unit for u in units] == sorted(u.unit for u in units) and units[0].unit >= 1,
        "units must be numbered from 1, ascending",
    )
    _require(any(u.texts or u.tables or u.images for u in units), "a valid sample must contain some content")


@experimental
async def check_cache(cache: Cache) -> None:
    """Stored bytes come back unchanged, misses are ``None``, ``delete`` removes, counters only go up."""
    await cache.ping()
    _require(await cache.get("conformance:missing") is None, "a missing key must read as None")
    await cache.set("conformance:k", b"value", 60)
    _require(await cache.get("conformance:k") == b"value", "a stored value must read back unchanged")
    await cache.delete("conformance:k")
    _require(await cache.get("conformance:k") is None, "a deleted key must read as None")
    await cache.delete("conformance:never-set")  # must not raise
    first = await cache.incr("conformance:counter")
    second = await cache.incr("conformance:counter")
    _require(second == first + 1, "incr must increase by one")
    _require(await cache.counter("conformance:counter") == second, "counter must report the current value")
    _require(await cache.counter("conformance:never-incremented") == 0, "an untouched counter reads 0")
    await cache.set("conformance:short", b"x", 1)
    await asyncio.sleep(1.2)
    _require(await cache.get("conformance:short") is None, "a value must expire after its ttl")


@experimental
async def check_conversation_store(store: ConversationStore) -> None:
    """Messages come back oldest first, ``limit`` keeps the most recent, unknown ids are empty, ``delete`` says whether anything was there."""
    await store.ping()
    cid = "conformance-conversation"
    await store.delete(cid)
    _require(await store.load(cid, 10) == [], "an unknown conversation must load as []")
    turns = [ChatMessage("user", "one"), ChatMessage("assistant", "two"), ChatMessage("user", "three")]
    await store.append(cid, turns[:2])
    await store.append(cid, turns[2:])
    _require(await store.load(cid, 10) == turns, "messages must come back in the order they were appended")
    _require(await store.load(cid, 2) == turns[-2:], "limit must keep the most recent messages, oldest first")
    _require(await store.delete(cid) is True, "delete must return True when the conversation existed")
    _require(await store.delete(cid) is False, "delete must return False when nothing was there")
    _require(await store.load(cid, 10) == [], "a deleted conversation must load as []")


__all__ = [
    "ConformanceError",
    "check_cache",
    "check_chat_model",
    "check_chunker",
    "check_conversation_store",
    "check_embedder",
    "check_parser",
    "check_query_expander",
    "check_reranker",
]
