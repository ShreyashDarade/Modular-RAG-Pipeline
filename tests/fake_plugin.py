"""Deterministic model providers, registered through the real plug-in hook.

Used by the tests (``PLUGINS=tests.fake_plugin``) - which also proves a new provider can be
added without editing any core module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import AsyncIterator, Sequence

from src.core.registry import Registries
from src.core.types import ChatMessage

_WORD = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)


class HashEmbedder:
    """Bag-of-words hashing into ``dimensions`` buckets, L2-normalised: texts sharing words are
    close in cosine space, so vector search behaves meaningfully without any model."""

    def __init__(self, model_id: str, dimensions: int) -> None:
        self.model_id = model_id
        self.dimensions = dimensions
        self.calls: list[tuple[str, int]] = []

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in _WORD.findall(text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:  # signed buckets can cancel exactly; a cosine index rejects zero vectors
            vector[0], norm = 1.0, 1.0
        return [v / norm for v in vector]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(("documents", len(texts)))
        return [self._vector(t) for t in texts]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(("queries", len(texts)))
        return [self._vector(t) for t in texts]


class ScriptedChat:
    """Replies depend on the role the prompt asks for, so the whole pipeline can be asserted on."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.calls: list[list[ChatMessage]] = []

    def _reply(self, messages: Sequence[ChatMessage]) -> str:
        system, last = messages[0].content, messages[-1].content
        if system.startswith("You are a search query rewriter"):
            query = last.removeprefix("Query: ")
            return f"- {query} explained\n- about {query}\n- {query} details"
        if system.startswith("You check whether an answer is supported"):
            # supported unless the answer was written from "No relevant context found."
            supported = "No relevant context found." not in last
            return json.dumps({"claims": [{"claim": "the answer", "supported": supported}]})
        if system.startswith("You compare an answer with a reference"):
            reference = re.search(r"Reference answer:\n(.*?)\n\nAnswer:", last, flags=re.S)
            words = set(_WORD.findall((reference.group(1) if reference else "").lower()))
            answer = last.rsplit("Answer:\n", 1)[-1].lower()
            hit = any(w in answer for w in words)
            return json.dumps({"verdict": "correct" if hit else "incorrect", "reason": "scripted"})
        if system.startswith("You decide whether an answer declines"):
            return json.dumps({"abstained": "answer with 0 sources" in last})
        if system.startswith("You write evaluation questions"):
            passage = last.removeprefix("Passage:\n").strip()
            words = passage.split()
            return json.dumps(
                {"question": "What is said about " + " ".join(words[:3]) + "?", "answer": " ".join(words[:8])}
            )
        if system.startswith("Rewrite the user's latest message"):
            return "STANDALONE " + last.rsplit("Latest message: ", 1)[-1]
        sources = re.findall(r"^\[(\d+)\] Source: (\S+) \| Page: (\S+)", last, flags=re.M)
        cited = "; ".join(f"[Source: {s.rsplit('/', 1)[-1]}, Page: {p}]" for _, s, p in sources[:3])
        return f"[{self.model_id}] answer with {len(sources)} sources {cited}".strip()

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        return self._reply(messages)

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        for word in self._reply(messages).split(" "):
            yield word + " "


class OverlapReranker:
    """Scores a passage by the share of query words it contains (a deterministic stand-in for a model)."""

    def __init__(self) -> None:
        self.calls = 0

    async def start(self) -> None: ...

    async def rerank(self, documents, query):
        self.calls += 1
        words = set(_WORD.findall(query.lower()))
        for doc in documents:
            have = set(_WORD.findall(doc.content.lower()))
            doc.rerank_score = len(words & have) / max(1, len(words))
        return list(documents)

    async def close(self) -> None: ...


def register(registries: Registries) -> None:
    registries.rerankers.register("overlap", lambda name, spec, settings: OverlapReranker())
    registries.chat_providers.register("fake", lambda model_id, spec, settings: ScriptedChat(model_id))
    registries.embedding_providers.register(
        "fake", lambda model_id, spec, settings: HashEmbedder(model_id, spec.dimensions or 64)
    )
