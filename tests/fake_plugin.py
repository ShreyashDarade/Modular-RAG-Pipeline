"""Deterministic model providers, registered through the real plug-in hook.

Used by the tests (``PLUGINS=tests.fake_plugin``) - which also proves a new provider can be
added without editing any core module.
"""

from __future__ import annotations

import hashlib
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


def register(registries: Registries) -> None:
    registries.chat_providers.register("fake", lambda model_id, spec, settings: ScriptedChat(model_id))
    registries.embedding_providers.register(
        "fake", lambda model_id, spec, settings: HashEmbedder(model_id, spec.dimensions or 64)
    )
